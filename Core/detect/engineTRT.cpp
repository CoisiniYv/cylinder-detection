#include "engineTRT.h"

#include "config.h"
#include "logging.h"

#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

Logger gLogger;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}

nvinfer1::Dims toDims32(const nvinfer1::Dims64& dims64) {
    nvinfer1::Dims dims{};
    dims.nbDims = static_cast<int>(dims64.nbDims);
    for (int i = 0; i < dims.nbDims && i < nvinfer1::Dims::MAX_DIMS; ++i) {
        dims.d[i] = static_cast<int>(dims64.d[i]);
    }
    return dims;
}

std::size_t getSizeByDim64(const nvinfer1::Dims64& dims) {
    std::size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        const std::int64_t dimension = dims.d[i];
        if (dimension == -1) {
            size *= MAX_NUM_PROMPTS;
        }
        else if (dimension > 0) {
            size *= static_cast<std::size_t>(dimension);
        }
        else {
            throw std::runtime_error("TensorRT tensor has an invalid dimension");
        }
    }
    return size;
}

bool hasOnnxExtension(const std::string& path) {
    std::string extension = std::filesystem::path(path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return extension == ".onnx";
}

} // namespace

EngineTRT::EngineTRT(
    const std::string& model_path,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    bool dynamic_shape,
    bool fp16) {
    if (model_path.empty()) throw std::invalid_argument("TensorRT model path is empty");

    if (hasOnnxExtension(model_path)) {
        build(model_path, input_names, output_names, dynamic_shape, fp16);
    }
    else {
        deserializeEngine(model_path, input_names, output_names);
    }
}

EngineTRT::~EngineTRT() {
    releaseBuffers();
    if (mCudaStream) {
        cudaStreamDestroy(mCudaStream);
        mCudaStream = nullptr;
    }
    delete mContext;
    delete mEngine;
    delete mRuntime;
    mContext = nullptr;
    mEngine = nullptr;
    mRuntime = nullptr;
}

void EngineTRT::releaseBuffers() noexcept {
    for (void*& buffer : mGpuBuffers) {
        if (buffer) {
            cudaFree(buffer);
            buffer = nullptr;
        }
    }
    for (float*& buffer : mCpuBuffers) {
        delete[] buffer;
        buffer = nullptr;
    }
    mGpuBuffers.clear();
    mCpuBuffers.clear();
    mBufferBindingBytes.clear();
    mBufferBindingSizes.clear();
    mInputDims.clear();
    mOutputDims.clear();
}

void EngineTRT::build(
    const std::string& onnx_path,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    bool dynamic_shape,
    bool fp16) {
    if (!std::filesystem::is_regular_file(onnx_path)) {
        throw std::runtime_error("ONNX file not found: " + onnx_path);
    }

    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    if (!builder) throw std::runtime_error("createInferBuilder failed");

    const auto explicit_batch =
        1U << static_cast<std::uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicit_batch);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    nvonnxparser::IParser* parser = network ? nvonnxparser::createParser(*network, gLogger) : nullptr;

    if (!network || !config || !parser) {
        delete parser;
        delete config;
        delete network;
        delete builder;
        throw std::runtime_error("failed to create TensorRT builder resources");
    }

    try {
        if (dynamic_shape) {
            if (input_names.size() < 3) {
                throw std::invalid_argument("dynamic SAM decoder requires at least three input names");
            }
            nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
            if (!profile) throw std::runtime_error("createOptimizationProfile failed");

            const bool coords_ok =
                profile->setDimensions(input_names[1].c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3{1, 1, 2}) &&
                profile->setDimensions(input_names[1].c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3{1, 2, 2}) &&
                profile->setDimensions(input_names[1].c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3{1, 10, 2});
            const bool labels_ok =
                profile->setDimensions(input_names[2].c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{1, 1}) &&
                profile->setDimensions(input_names[2].c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2{1, 2}) &&
                profile->setDimensions(input_names[2].c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2{1, 10});
            if (!coords_ok || !labels_ok || config->addOptimizationProfile(profile) < 0) {
                throw std::runtime_error("failed to configure TensorRT dynamic-shape profile");
            }
        }

        if (fp16) config->setFlag(nvinfer1::BuilderFlag::kFP16);

        if (!parser->parseFromFile(
                onnx_path.c_str(),
                static_cast<int>(gLogger.getReportableSeverity()))) {
            throw std::runtime_error("failed to parse ONNX model: " + onnx_path);
        }

        nvinfer1::IHostMemory* plan = builder->buildSerializedNetwork(*network, *config);
        if (!plan) throw std::runtime_error("buildSerializedNetwork failed");

        mRuntime = nvinfer1::createInferRuntime(gLogger);
        if (!mRuntime) {
            delete plan;
            throw std::runtime_error("createInferRuntime failed");
        }

        mEngine = mRuntime->deserializeCudaEngine(plan->data(), plan->size());
        delete plan;
        if (!mEngine) throw std::runtime_error("deserializeCudaEngine failed after ONNX build");

        mContext = mEngine->createExecutionContext();
        if (!mContext) throw std::runtime_error("createExecutionContext failed");
    }
    catch (...) {
        delete parser;
        delete config;
        delete network;
        delete builder;
        throw;
    }

    delete parser;
    delete config;
    delete network;
    delete builder;
    initialize(input_names, output_names);
}

void EngineTRT::deserializeEngine(
    const std::string& engine_name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names) {
    std::ifstream file(engine_name, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("unable to open TensorRT engine: " + engine_name);

    const std::streamsize file_size = file.tellg();
    if (file_size <= 0) throw std::runtime_error("TensorRT engine is empty: " + engine_name);
    file.seekg(0, std::ios::beg);

    std::vector<char> serialized_engine(static_cast<std::size_t>(file_size));
    if (!file.read(serialized_engine.data(), file_size)) {
        throw std::runtime_error("failed to read TensorRT engine: " + engine_name);
    }

    mRuntime = nvinfer1::createInferRuntime(gLogger);
    if (!mRuntime) throw std::runtime_error("createInferRuntime failed");

    mEngine = mRuntime->deserializeCudaEngine(serialized_engine.data(), serialized_engine.size());
    if (!mEngine) throw std::runtime_error("deserializeCudaEngine failed: " + engine_name);

    mContext = mEngine->createExecutionContext();
    if (!mContext) throw std::runtime_error("createExecutionContext failed");

    if (mEngine->getNbIOTensors() != static_cast<int>(input_names.size() + output_names.size())) {
        throw std::runtime_error("TensorRT engine IO count does not match configured tensor names");
    }
    initialize(input_names, output_names);
}

void EngineTRT::initialize(
    const std::vector<std::string>&,
    const std::vector<std::string>&) {
    if (!mEngine || !mContext) throw std::runtime_error("TensorRT engine is not initialized");

    releaseBuffers();
    const int tensor_count = mEngine->getNbIOTensors();
    mGpuBuffers.assign(static_cast<std::size_t>(tensor_count), nullptr);
    mCpuBuffers.assign(static_cast<std::size_t>(tensor_count), nullptr);
    mBufferBindingBytes.reserve(static_cast<std::size_t>(tensor_count));
    mBufferBindingSizes.reserve(static_cast<std::size_t>(tensor_count));

    for (int i = 0; i < tensor_count; ++i) {
        const char* tensor_name = mEngine->getIOTensorName(i);
        if (!tensor_name) throw std::runtime_error("TensorRT returned a null tensor name");

        const nvinfer1::Dims64 shape64 = mEngine->getTensorShape(tensor_name);
        const std::size_t element_count = getSizeByDim64(shape64);
        const std::size_t byte_count = element_count * sizeof(float);
        mBufferBindingSizes.push_back(element_count);
        mBufferBindingBytes.push_back(byte_count);

        mCpuBuffers[static_cast<std::size_t>(i)] = new float[element_count]{};
        checkCuda(
            cudaMalloc(&mGpuBuffers[static_cast<std::size_t>(i)], byte_count),
            "cudaMalloc TensorRT buffer");

        if (mEngine->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kINPUT) {
            mInputDims.push_back(toDims32(shape64));
        }
        else {
            mOutputDims.push_back(toDims32(shape64));
        }
    }

    checkCuda(cudaStreamCreate(&mCudaStream), "cudaStreamCreate");
}

void EngineTRT::saveEngine(const std::string& engine_file_path) const {
    if (!mEngine) throw std::runtime_error("cannot serialize an uninitialized TensorRT engine");

    nvinfer1::IHostMemory* serialized = mEngine->serialize();
    if (!serialized) throw std::runtime_error("TensorRT engine serialization failed");

    std::ofstream file(engine_file_path, std::ios::binary);
    if (!file) {
        delete serialized;
        throw std::runtime_error("unable to create engine file: " + engine_file_path);
    }
    file.write(reinterpret_cast<const char*>(serialized->data()), serialized->size());
    const bool write_ok = static_cast<bool>(file);
    delete serialized;
    if (!write_ok) throw std::runtime_error("failed to write engine file: " + engine_file_path);
}

bool EngineTRT::infer() {
    if (!mContext || !mEngine || !mCudaStream) return false;

    try {
        copyInputToDeviceAsync(mCudaStream);
        // executeV2 is synchronous and does not consume mCudaStream. Explicitly
        // complete H2D/profile work before calling it.
        checkCuda(cudaStreamSynchronize(mCudaStream), "TensorRT input stream synchronize");

        if (!mContext->executeV2(mGpuBuffers.data())) return false;

        copyOutputToHostAsync(mCudaStream);
        // getOutput reads host buffers immediately after infer() returns.
        checkCuda(cudaStreamSynchronize(mCudaStream), "TensorRT output stream synchronize");
        return true;
    }
    catch (const std::exception& e) {
        LOG_ERROR(gLogger) << "TensorRT infer failed: " << e.what() << std::endl;
        return false;
    }
}

void EngineTRT::copyInputToDeviceAsync(cudaStream_t stream) {
    memcpyBuffers(true, false, true, stream);
}

void EngineTRT::copyOutputToHostAsync(cudaStream_t stream) {
    memcpyBuffers(false, true, true, stream);
}

void EngineTRT::memcpyBuffers(
    bool copy_input,
    bool device_to_host,
    bool async,
    cudaStream_t stream) {
    if (!mEngine) throw std::runtime_error("TensorRT engine is not initialized");

    for (int i = 0; i < mEngine->getNbIOTensors(); ++i) {
        const char* tensor_name = mEngine->getIOTensorName(i);
        const bool is_input =
            mEngine->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kINPUT;
        if ((copy_input && !is_input) || (!copy_input && is_input)) continue;

        void* destination = device_to_host
            ? static_cast<void*>(mCpuBuffers[static_cast<std::size_t>(i)])
            : mGpuBuffers[static_cast<std::size_t>(i)];
        const void* source = device_to_host
            ? mGpuBuffers[static_cast<std::size_t>(i)]
            : static_cast<const void*>(mCpuBuffers[static_cast<std::size_t>(i)]);
        const std::size_t bytes = mBufferBindingBytes[static_cast<std::size_t>(i)];
        const cudaMemcpyKind kind = device_to_host
            ? cudaMemcpyDeviceToHost
            : cudaMemcpyHostToDevice;

        if (async) {
            checkCuda(cudaMemcpyAsync(destination, source, bytes, kind, stream), "cudaMemcpyAsync");
        }
        else {
            checkCuda(cudaMemcpy(destination, source, bytes, kind), "cudaMemcpy");
        }
    }
}

std::size_t EngineTRT::getSizeByDim(const nvinfer1::Dims& dims) {
    std::size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] == -1) size *= MAX_NUM_PROMPTS;
        else if (dims.d[i] > 0) size *= static_cast<std::size_t>(dims.d[i]);
        else throw std::runtime_error("TensorRT tensor has an invalid dimension");
    }
    return size;
}

void EngineTRT::setInput(const cv::Mat& image) {
    if (mInputDims.empty() || mCpuBuffers.empty()) {
        throw std::runtime_error("TensorRT input buffers are not initialized");
    }
    if (image.empty() || image.type() != CV_8UC3) {
        throw std::invalid_argument("TensorRT image input must be non-empty CV_8UC3");
    }

    const nvinfer1::Dims& input_dims = mInputDims.front();
    if (input_dims.nbDims < 4) throw std::runtime_error("unexpected image input dimensions");
    const int input_height = input_dims.d[2];
    const int input_width = input_dims.d[3];
    if (image.rows != input_height || image.cols != input_width) {
        throw std::invalid_argument("image size does not match TensorRT input dimensions");
    }

    const std::size_t plane_size =
        static_cast<std::size_t>(input_height) * static_cast<std::size_t>(input_width);
    if (mBufferBindingSizes.front() < plane_size * 3) {
        throw std::runtime_error("TensorRT image input buffer is too small");
    }

    float* target = mCpuBuffers.front();
    for (int row = 0; row < image.rows; ++row) {
        const cv::Vec3b* pixels = image.ptr<cv::Vec3b>(row);
        for (int col = 0; col < image.cols; ++col) {
            const std::size_t index =
                static_cast<std::size_t>(row) * image.cols + static_cast<std::size_t>(col);
            target[index] = (pixels[col][2] / 255.0f - 0.485f) / 0.229f;
            target[index + plane_size] = (pixels[col][1] / 255.0f - 0.456f) / 0.224f;
            target[index + 2 * plane_size] = (pixels[col][0] / 255.0f - 0.406f) / 0.225f;
        }
    }
}

void EngineTRT::setInput(
    const float* features,
    const float* image_point_coords,
    const float* image_point_labels,
    const float* mask_input,
    const float* has_mask_input,
    int num_points) {
    if (!features || !image_point_coords || !image_point_labels || !mask_input || !has_mask_input) {
        throw std::invalid_argument("SAM decoder input contains a null pointer");
    }
    if (num_points <= 0 || num_points > 10) {
        throw std::invalid_argument("SAM decoder num_points must be in [1, 10]");
    }
    if (!mContext || !mEngine || mCpuBuffers.size() < 5 || mGpuBuffers.size() < 5) {
        throw std::runtime_error("SAM decoder buffers are not initialized");
    }

    delete[] mCpuBuffers[1];
    delete[] mCpuBuffers[2];
    mCpuBuffers[1] = new float[static_cast<std::size_t>(num_points) * 2];
    mCpuBuffers[2] = new float[static_cast<std::size_t>(num_points)];

    if (mGpuBuffers[1]) checkCuda(cudaFree(mGpuBuffers[1]), "cudaFree point_coords");
    if (mGpuBuffers[2]) checkCuda(cudaFree(mGpuBuffers[2]), "cudaFree point_labels");
    mGpuBuffers[1] = nullptr;
    mGpuBuffers[2] = nullptr;

    const std::size_t coords_bytes = sizeof(float) * static_cast<std::size_t>(num_points) * 2;
    const std::size_t labels_bytes = sizeof(float) * static_cast<std::size_t>(num_points);
    checkCuda(cudaMalloc(&mGpuBuffers[1], coords_bytes), "cudaMalloc point_coords");
    checkCuda(cudaMalloc(&mGpuBuffers[2], labels_bytes), "cudaMalloc point_labels");
    mBufferBindingBytes[1] = coords_bytes;
    mBufferBindingBytes[2] = labels_bytes;
    mBufferBindingSizes[1] = static_cast<std::size_t>(num_points) * 2;
    mBufferBindingSizes[2] = static_cast<std::size_t>(num_points);

    std::memcpy(mCpuBuffers[0], features, mBufferBindingBytes[0]);
    std::memcpy(mCpuBuffers[1], image_point_coords, coords_bytes);
    std::memcpy(mCpuBuffers[2], image_point_labels, labels_bytes);
    std::memcpy(mCpuBuffers[3], mask_input, mBufferBindingBytes[3]);
    std::memcpy(mCpuBuffers[4], has_mask_input, mBufferBindingBytes[4]);

    if (!mContext->setOptimizationProfileAsync(0, mCudaStream)) {
        throw std::runtime_error("setOptimizationProfileAsync failed");
    }
    const char* coords_name = mEngine->getIOTensorName(1);
    const char* labels_name = mEngine->getIOTensorName(2);
    if (!mContext->setInputShape(coords_name, nvinfer1::Dims3{1, num_points, 2}) ||
        !mContext->setInputShape(labels_name, nvinfer1::Dims2{1, num_points})) {
        throw std::runtime_error("setInputShape failed");
    }
}

void EngineTRT::getOutput(float* features) const {
    if (!features || !mEngine) throw std::invalid_argument("invalid TensorRT output target");

    for (int i = 0; i < mEngine->getNbIOTensors(); ++i) {
        const char* tensor_name = mEngine->getIOTensorName(i);
        if (mEngine->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kOUTPUT) {
            std::memcpy(features, mCpuBuffers[static_cast<std::size_t>(i)],
                        mBufferBindingBytes[static_cast<std::size_t>(i)]);
            return;
        }
    }
    throw std::runtime_error("TensorRT engine has no output tensor");
}

void EngineTRT::getOutput(float* iou_prediction, float* low_resolution_masks) const {
    if (!iou_prediction || !low_resolution_masks || !mEngine) {
        throw std::invalid_argument("invalid SAM decoder output target");
    }

    std::vector<int> outputs;
    for (int i = 0; i < mEngine->getNbIOTensors(); ++i) {
        const char* tensor_name = mEngine->getIOTensorName(i);
        if (mEngine->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kOUTPUT) {
            outputs.push_back(i);
        }
    }
    if (outputs.size() != 2) {
        throw std::runtime_error("SAM decoder must expose exactly two output tensors");
    }

    const int first = outputs[0];
    const int second = outputs[1];
    const int iou_index =
        mBufferBindingBytes[static_cast<std::size_t>(first)] <=
                mBufferBindingBytes[static_cast<std::size_t>(second)]
            ? first
            : second;
    const int masks_index = iou_index == first ? second : first;

    std::memcpy(iou_prediction,
                mCpuBuffers[static_cast<std::size_t>(iou_index)],
                mBufferBindingBytes[static_cast<std::size_t>(iou_index)]);
    std::memcpy(low_resolution_masks,
                mCpuBuffers[static_cast<std::size_t>(masks_index)],
                mBufferBindingBytes[static_cast<std::size_t>(masks_index)]);
}
