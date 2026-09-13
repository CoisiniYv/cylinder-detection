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
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
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
    if (model_path.empty()) {
        throw std::invalid_argument("TensorRT model path is empty");
    }
    if (input_names.empty() || output_names.empty()) {
        throw std::invalid_argument("TensorRT input/output tensor names must not be empty");
    }

    try {
        if (hasOnnxExtension(model_path)) {
            build(model_path, input_names, output_names, dynamic_shape, fp16);
        }
        else {
            deserializeEngine(model_path, input_names, output_names);
        }
    }
    catch (...) {
        // Destructors are not invoked when a constructor throws. Explicitly
        // release any TensorRT/CUDA objects created before the failure point.
        releaseResources();
        throw;
    }
}

EngineTRT::~EngineTRT() {
    releaseResources();
}

void EngineTRT::releaseBuffers() noexcept {
    for (void*& buffer : mGpuBuffers) {
        if (buffer) {
            (void)cudaFree(buffer);
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

    mInputNames.clear();
    mOutputNames.clear();
    mInputIndices.clear();
    mOutputIndices.clear();
    mInputDims.clear();
    mOutputDims.clear();
}

void EngineTRT::releaseResources() noexcept {
    if (mCudaStream) {
        (void)cudaStreamSynchronize(mCudaStream);
    }

    releaseBuffers();

    if (mCudaStream) {
        (void)cudaStreamDestroy(mCudaStream);
        mCudaStream = nullptr;
    }

    delete mContext;
    delete mEngine;
    delete mRuntime;
    mContext = nullptr;
    mEngine = nullptr;
    mRuntime = nullptr;
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
        1U << static_cast<std::uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicit_batch);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    nvonnxparser::IParser* parser =
        network ? nvonnxparser::createParser(*network, gLogger) : nullptr;

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
                throw std::invalid_argument(
                    "dynamic SAM decoder requires at least three input names");
            }

            nvinfer1::IOptimizationProfile* profile =
                builder->createOptimizationProfile();
            if (!profile) {
                throw std::runtime_error("createOptimizationProfile failed");
            }

            const bool coords_ok =
                profile->setDimensions(
                    input_names[1].c_str(),
                    nvinfer1::OptProfileSelector::kMIN,
                    nvinfer1::Dims3{1, 1, 2}) &&
                profile->setDimensions(
                    input_names[1].c_str(),
                    nvinfer1::OptProfileSelector::kOPT,
                    nvinfer1::Dims3{1, 2, 2}) &&
                profile->setDimensions(
                    input_names[1].c_str(),
                    nvinfer1::OptProfileSelector::kMAX,
                    nvinfer1::Dims3{1, 10, 2});
            const bool labels_ok =
                profile->setDimensions(
                    input_names[2].c_str(),
                    nvinfer1::OptProfileSelector::kMIN,
                    nvinfer1::Dims2{1, 1}) &&
                profile->setDimensions(
                    input_names[2].c_str(),
                    nvinfer1::OptProfileSelector::kOPT,
                    nvinfer1::Dims2{1, 2}) &&
                profile->setDimensions(
                    input_names[2].c_str(),
                    nvinfer1::OptProfileSelector::kMAX,
                    nvinfer1::Dims2{1, 10});
            if (!coords_ok || !labels_ok || config->addOptimizationProfile(profile) < 0) {
                throw std::runtime_error(
                    "failed to configure TensorRT dynamic-shape profile");
            }
        }

        if (fp16) config->setFlag(nvinfer1::BuilderFlag::kFP16);

        if (!parser->parseFromFile(
                onnx_path.c_str(),
                static_cast<int>(gLogger.getReportableSeverity()))) {
            throw std::runtime_error("failed to parse ONNX model: " + onnx_path);
        }

        nvinfer1::IHostMemory* plan =
            builder->buildSerializedNetwork(*network, *config);
        if (!plan) throw std::runtime_error("buildSerializedNetwork failed");

        mRuntime = nvinfer1::createInferRuntime(gLogger);
        if (!mRuntime) {
            delete plan;
            throw std::runtime_error("createInferRuntime failed");
        }

        mEngine = mRuntime->deserializeCudaEngine(plan->data(), plan->size());
        delete plan;
        if (!mEngine) {
            throw std::runtime_error("deserializeCudaEngine failed after ONNX build");
        }

        mContext = mEngine->createExecutionContext();
        if (!mContext) {
            throw std::runtime_error("createExecutionContext failed");
        }
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
    if (!file) {
        throw std::runtime_error("unable to open TensorRT engine: " + engine_name);
    }

    const std::streamsize file_size = file.tellg();
    if (file_size <= 0) {
        throw std::runtime_error("TensorRT engine is empty: " + engine_name);
    }
    file.seekg(0, std::ios::beg);

    std::vector<char> serialized_engine(static_cast<std::size_t>(file_size));
    if (!file.read(serialized_engine.data(), file_size)) {
        throw std::runtime_error("failed to read TensorRT engine: " + engine_name);
    }

    mRuntime = nvinfer1::createInferRuntime(gLogger);
    if (!mRuntime) throw std::runtime_error("createInferRuntime failed");

    mEngine = mRuntime->deserializeCudaEngine(
        serialized_engine.data(),
        serialized_engine.size());
    if (!mEngine) {
        throw std::runtime_error("deserializeCudaEngine failed: " + engine_name);
    }

    mContext = mEngine->createExecutionContext();
    if (!mContext) throw std::runtime_error("createExecutionContext failed");

    initialize(input_names, output_names);
}

int EngineTRT::tensorIndex(const std::string& tensor_name) const {
    if (!mEngine || tensor_name.empty()) return -1;

    for (int index = 0; index < mEngine->getNbIOTensors(); ++index) {
        const char* current_name = mEngine->getIOTensorName(index);
        if (current_name && tensor_name == current_name) return index;
    }
    return -1;
}

int EngineTRT::inputIndex(std::size_t logical_index) const {
    if (logical_index >= mInputIndices.size()) {
        throw std::out_of_range("TensorRT logical input index is out of range");
    }
    return mInputIndices[logical_index];
}

int EngineTRT::outputIndex(std::size_t logical_index) const {
    if (logical_index >= mOutputIndices.size()) {
        throw std::out_of_range("TensorRT logical output index is out of range");
    }
    return mOutputIndices[logical_index];
}

void EngineTRT::initialize(
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names) {
    if (!mEngine || !mContext) {
        throw std::runtime_error("TensorRT engine is not initialized");
    }

    const int tensor_count = mEngine->getNbIOTensors();
    if (tensor_count != static_cast<int>(input_names.size() + output_names.size())) {
        throw std::runtime_error(
            "TensorRT engine IO count does not match configured tensor names");
    }

    if (mCudaStream) {
        checkCuda(cudaStreamSynchronize(mCudaStream), "cudaStreamSynchronize before reinitialize");
        checkCuda(cudaStreamDestroy(mCudaStream), "cudaStreamDestroy before reinitialize");
        mCudaStream = nullptr;
    }
    releaseBuffers();

    mGpuBuffers.assign(static_cast<std::size_t>(tensor_count), nullptr);
    mCpuBuffers.assign(static_cast<std::size_t>(tensor_count), nullptr);
    mBufferBindingBytes.assign(static_cast<std::size_t>(tensor_count), 0);
    mBufferBindingSizes.assign(static_cast<std::size_t>(tensor_count), 0);

    for (int index = 0; index < tensor_count; ++index) {
        const char* tensor_name = mEngine->getIOTensorName(index);
        if (!tensor_name) {
            throw std::runtime_error("TensorRT returned a null tensor name");
        }

        const nvinfer1::Dims64 shape = mEngine->getTensorShape(tensor_name);
        const std::size_t element_count = getSizeByDim64(shape);
        const std::size_t byte_count = element_count * sizeof(float);
        const std::size_t storage_index = static_cast<std::size_t>(index);

        mBufferBindingSizes[storage_index] = element_count;
        mBufferBindingBytes[storage_index] = byte_count;
        mCpuBuffers[storage_index] = new float[element_count]{};
        checkCuda(
            cudaMalloc(&mGpuBuffers[storage_index], byte_count),
            "cudaMalloc TensorRT buffer");
    }

    mInputNames = input_names;
    mOutputNames = output_names;
    mInputIndices.reserve(input_names.size());
    mOutputIndices.reserve(output_names.size());
    mInputDims.reserve(input_names.size());
    mOutputDims.reserve(output_names.size());

    auto register_tensor = [this](
                               const std::string& name,
                               nvinfer1::TensorIOMode expected_mode,
                               std::vector<int>& indices,
                               std::vector<nvinfer1::Dims>& dims) {
        const int index = tensorIndex(name);
        if (index < 0) {
            throw std::runtime_error("TensorRT tensor not found: " + name);
        }

        const char* engine_name = mEngine->getIOTensorName(index);
        if (!engine_name || mEngine->getTensorIOMode(engine_name) != expected_mode) {
            throw std::runtime_error("TensorRT tensor has unexpected IO mode: " + name);
        }
        if (std::find(indices.begin(), indices.end(), index) != indices.end()) {
            throw std::runtime_error("duplicate TensorRT tensor configured: " + name);
        }

        indices.push_back(index);
        dims.push_back(toDims32(mEngine->getTensorShape(name.c_str())));
    };

    for (const auto& name : input_names) {
        register_tensor(
            name,
            nvinfer1::TensorIOMode::kINPUT,
            mInputIndices,
            mInputDims);
    }
    for (const auto& name : output_names) {
        register_tensor(
            name,
            nvinfer1::TensorIOMode::kOUTPUT,
            mOutputIndices,
            mOutputDims);
    }

    checkCuda(cudaStreamCreate(&mCudaStream), "cudaStreamCreate");
}

void EngineTRT::saveEngine(const std::string& engine_file_path) const {
    if (!mEngine) {
        throw std::runtime_error("cannot serialize an uninitialized TensorRT engine");
    }

    nvinfer1::IHostMemory* serialized = mEngine->serialize();
    if (!serialized) {
        throw std::runtime_error("TensorRT engine serialization failed");
    }

    std::ofstream file(engine_file_path, std::ios::binary);
    if (!file) {
        delete serialized;
        throw std::runtime_error("unable to create engine file: " + engine_file_path);
    }

    file.write(
        reinterpret_cast<const char*>(serialized->data()),
        static_cast<std::streamsize>(serialized->size()));
    const bool write_ok = static_cast<bool>(file);
    delete serialized;

    if (!write_ok) {
        throw std::runtime_error("failed to write engine file: " + engine_file_path);
    }
}

bool EngineTRT::infer() {
    if (!mContext || !mEngine || !mCudaStream) return false;

    try {
        copyInputToDeviceAsync(mCudaStream);
        // executeV2 is synchronous and does not consume mCudaStream. Explicitly
        // complete H2D/profile work before calling it.
        checkCuda(
            cudaStreamSynchronize(mCudaStream),
            "TensorRT input stream synchronize");

        if (!mContext->executeV2(mGpuBuffers.data())) return false;

        copyOutputToHostAsync(mCudaStream);
        // getOutput reads host buffers immediately after infer() returns.
        checkCuda(
            cudaStreamSynchronize(mCudaStream),
            "TensorRT output stream synchronize");
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
    if (!mEngine) {
        throw std::runtime_error("TensorRT engine is not initialized");
    }

    for (int index = 0; index < mEngine->getNbIOTensors(); ++index) {
        const char* tensor_name = mEngine->getIOTensorName(index);
        if (!tensor_name) {
            throw std::runtime_error("TensorRT returned a null tensor name");
        }

        const bool is_input =
            mEngine->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kINPUT;
        if ((copy_input && !is_input) || (!copy_input && is_input)) continue;

        const std::size_t storage_index = static_cast<std::size_t>(index);
        void* destination = device_to_host
            ? static_cast<void*>(mCpuBuffers[storage_index])
            : mGpuBuffers[storage_index];
        const void* source = device_to_host
            ? mGpuBuffers[storage_index]
            : static_cast<const void*>(mCpuBuffers[storage_index]);
        const std::size_t bytes = mBufferBindingBytes[storage_index];
        const cudaMemcpyKind kind = device_to_host
            ? cudaMemcpyDeviceToHost
            : cudaMemcpyHostToDevice;

        if (async) {
            checkCuda(
                cudaMemcpyAsync(destination, source, bytes, kind, stream),
                "cudaMemcpyAsync");
        }
        else {
            checkCuda(cudaMemcpy(destination, source, bytes, kind), "cudaMemcpy");
        }
    }
}

void EngineTRT::setInput(const cv::Mat& image) {
    if (mInputDims.empty() || mInputIndices.empty()) {
        throw std::runtime_error("TensorRT input buffers are not initialized");
    }
    if (image.empty() || image.type() != CV_8UC3) {
        throw std::invalid_argument(
            "TensorRT image input must be non-empty CV_8UC3");
    }

    const int engine_index = inputIndex(0);
    const std::size_t storage_index = static_cast<std::size_t>(engine_index);
    const nvinfer1::Dims& input_dims = mInputDims.front();
    if (input_dims.nbDims < 4) {
        throw std::runtime_error("unexpected image input dimensions");
    }

    const int input_height = input_dims.d[2];
    const int input_width = input_dims.d[3];
    if (image.rows != input_height || image.cols != input_width) {
        throw std::invalid_argument(
            "image size does not match TensorRT input dimensions");
    }

    const std::size_t plane_size =
        static_cast<std::size_t>(input_height) *
        static_cast<std::size_t>(input_width);
    if (mBufferBindingSizes.at(storage_index) < plane_size * 3) {
        throw std::runtime_error("TensorRT image input buffer is too small");
    }

    float* target = mCpuBuffers.at(storage_index);
    for (int row = 0; row < image.rows; ++row) {
        const cv::Vec3b* pixels = image.ptr<cv::Vec3b>(row);
        for (int column = 0; column < image.cols; ++column) {
            const std::size_t index =
                static_cast<std::size_t>(row) *
                    static_cast<std::size_t>(image.cols) +
                static_cast<std::size_t>(column);
            target[index] = (pixels[column][2] / 255.0f - 0.485f) / 0.229f;
            target[index + plane_size] =
                (pixels[column][1] / 255.0f - 0.456f) / 0.224f;
            target[index + 2 * plane_size] =
                (pixels[column][0] / 255.0f - 0.406f) / 0.225f;
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
    if (!features || !image_point_coords || !image_point_labels ||
        !mask_input || !has_mask_input) {
        throw std::invalid_argument("SAM decoder input contains a null pointer");
    }
    if (num_points <= 0 || num_points > 10) {
        throw std::invalid_argument("SAM decoder num_points must be in [1, 10]");
    }
    if (!mContext || !mEngine || mInputIndices.size() < 5) {
        throw std::runtime_error("SAM decoder buffers are not initialized");
    }

    const int features_index = inputIndex(0);
    const int coords_index = inputIndex(1);
    const int labels_index = inputIndex(2);
    const int mask_index = inputIndex(3);
    const int has_mask_index = inputIndex(4);

    const std::size_t coords_bytes =
        sizeof(float) * static_cast<std::size_t>(num_points) * 2;
    const std::size_t labels_bytes =
        sizeof(float) * static_cast<std::size_t>(num_points);

    std::unique_ptr<float[]> new_coords_host(
        new float[static_cast<std::size_t>(num_points) * 2]{});
    std::unique_ptr<float[]> new_labels_host(
        new float[static_cast<std::size_t>(num_points)]{});
    void* new_coords_device = nullptr;
    void* new_labels_device = nullptr;

    try {
        checkCuda(
            cudaMalloc(&new_coords_device, coords_bytes),
            "cudaMalloc point_coords");
        checkCuda(
            cudaMalloc(&new_labels_device, labels_bytes),
            "cudaMalloc point_labels");
    }
    catch (...) {
        if (new_coords_device) (void)cudaFree(new_coords_device);
        if (new_labels_device) (void)cudaFree(new_labels_device);
        throw;
    }

    const auto replace_dynamic_buffer = [this](
                                            int engine_index,
                                            std::unique_ptr<float[]> host,
                                            void* device,
                                            std::size_t bytes,
                                            std::size_t elements) {
        const std::size_t index = static_cast<std::size_t>(engine_index);
        if (mGpuBuffers[index]) {
            checkCuda(cudaFree(mGpuBuffers[index]), "cudaFree dynamic TensorRT buffer");
        }
        delete[] mCpuBuffers[index];
        mGpuBuffers[index] = device;
        mCpuBuffers[index] = host.release();
        mBufferBindingBytes[index] = bytes;
        mBufferBindingSizes[index] = elements;
    };

    try {
        replace_dynamic_buffer(
            coords_index,
            std::move(new_coords_host),
            new_coords_device,
            coords_bytes,
            static_cast<std::size_t>(num_points) * 2);
        new_coords_device = nullptr;

        replace_dynamic_buffer(
            labels_index,
            std::move(new_labels_host),
            new_labels_device,
            labels_bytes,
            static_cast<std::size_t>(num_points));
        new_labels_device = nullptr;
    }
    catch (...) {
        if (new_coords_device) (void)cudaFree(new_coords_device);
        if (new_labels_device) (void)cudaFree(new_labels_device);
        throw;
    }

    const auto copy_to_host = [this](
                                  int engine_index,
                                  const void* source,
                                  std::size_t bytes) {
        const std::size_t index = static_cast<std::size_t>(engine_index);
        if (bytes > mBufferBindingBytes.at(index)) {
            throw std::runtime_error("SAM decoder input exceeds TensorRT host buffer");
        }
        std::memcpy(mCpuBuffers.at(index), source, bytes);
    };

    copy_to_host(
        features_index,
        features,
        mBufferBindingBytes.at(static_cast<std::size_t>(features_index)));
    copy_to_host(coords_index, image_point_coords, coords_bytes);
    copy_to_host(labels_index, image_point_labels, labels_bytes);
    copy_to_host(
        mask_index,
        mask_input,
        mBufferBindingBytes.at(static_cast<std::size_t>(mask_index)));
    copy_to_host(
        has_mask_index,
        has_mask_input,
        mBufferBindingBytes.at(static_cast<std::size_t>(has_mask_index)));

    if (!mContext->setOptimizationProfileAsync(0, mCudaStream)) {
        throw std::runtime_error("setOptimizationProfileAsync failed");
    }
    if (!mContext->setInputShape(
            mInputNames.at(1).c_str(),
            nvinfer1::Dims3{1, num_points, 2}) ||
        !mContext->setInputShape(
            mInputNames.at(2).c_str(),
            nvinfer1::Dims2{1, num_points})) {
        throw std::runtime_error("setInputShape failed");
    }
}

void EngineTRT::getOutput(float* features) const {
    if (!features || !mEngine || mOutputIndices.empty()) {
        throw std::invalid_argument("invalid TensorRT output target");
    }

    const int engine_index = outputIndex(0);
    const std::size_t index = static_cast<std::size_t>(engine_index);
    std::memcpy(
        features,
        mCpuBuffers.at(index),
        mBufferBindingBytes.at(index));
}

void EngineTRT::getOutput(
    float* iou_prediction,
    float* low_resolution_masks) const {
    if (!iou_prediction || !low_resolution_masks || !mEngine) {
        throw std::invalid_argument("invalid SAM decoder output target");
    }
    if (mOutputIndices.size() != 2) {
        throw std::runtime_error(
            "SAM decoder must expose exactly two configured output tensors");
    }

    // SpeedSam configures outputs as {"iou_predictions", "low_res_masks"}.
    // Preserve that explicit business meaning instead of inferring tensor roles
    // from enumeration order or buffer size.
    const int iou_engine_index = outputIndex(0);
    const int masks_engine_index = outputIndex(1);
    const std::size_t iou_index = static_cast<std::size_t>(iou_engine_index);
    const std::size_t masks_index = static_cast<std::size_t>(masks_engine_index);

    std::memcpy(
        iou_prediction,
        mCpuBuffers.at(iou_index),
        mBufferBindingBytes.at(iou_index));
    std::memcpy(
        low_resolution_masks,
        mCpuBuffers.at(masks_index),
        mBufferBindingBytes.at(masks_index));
}
