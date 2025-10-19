#include "engineTRT.h"
#include "logging.h"
#include "cuda_utils.h"
#include "config.h"
#include "macros.h"
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

static Logger gLogger;

// Helper to convert Dims64 to Dims (keeps header unchanged)
static Dims toDims32(const Dims64& d64)
{
    Dims d{};
    d.nbDims = static_cast<int>(d64.nbDims);
    for (int i = 0; i < d.nbDims && i < Dims::MAX_DIMS; ++i)
    {
        d.d[i] = static_cast<int>(d64.d[i]);
    }
    return d;
}

// Helper to compute size from Dims64
static size_t getSizeByDim64(const Dims64& dims)
{
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        const int64_t di = dims.d[i];
        if (di == -1)
            size *= MAX_NUM_PROMPTS;
        else
            size *= static_cast<size_t>(di);
    }
    return size;
}

std::string getFileExtension(const std::string& filePath) {
    // Find the position of the last dot in the file path
    size_t dotPos = filePath.find_last_of(".");
    // If a dot is found, extract and return the substring after the dot as the file extension
    if (dotPos != std::string::npos) {
        return filePath.substr(dotPos + 1);
    }
    // Return an empty string if no extension is found
    return "";
}

EngineTRT::EngineTRT(string modelPath, vector<string> inputNames, vector<string> outputNames, bool isDynamicShape, bool isFP16) {
    // Check if the model file has an ".onnx" extension
    if (getFileExtension(modelPath) == "onnx") {
        // If the file is an ONNX model, build the engine using the provided parameters
        cout << "Building Engine from " << modelPath << endl;
        build(modelPath, inputNames, outputNames, isDynamicShape, isFP16);
    }
    else {
        // If the file is not an ONNX model, deserialize an existing engine
        cout << "Deserializing Engine." << endl;
        deserializeEngine(modelPath, inputNames, outputNames);
    }
}

EngineTRT::~EngineTRT() {
    // Release the CUDA stream
    cudaStreamDestroy(mCudaStream);
    // Free GPU buffers allocated for inference
    for (int i = 0; i < mGpuBuffers.size(); i++)
        CUDA_CHECK(cudaFree(mGpuBuffers[i]));
    // Free CPU buffers
    for (int i = 0; i < mCpuBuffers.size(); i++)
        delete[] mCpuBuffers[i];

    // Clean up and destroy the TensorRT engine components
    delete mContext;  // Destroy the execution context
    delete mEngine;   // Destroy the engine
    delete mRuntime;  // Destroy the runtime
}

void EngineTRT::build(string onnxPath, vector<string> inputNames, vector<string> outputNames, bool isDynamicShape, bool isFP16)
{
    // Check if the ONNX file exists. If not, print an error message and return.
    if (!std::filesystem::exists(onnxPath)) {
        std::cerr << "ONNX file not found: " << onnxPath << std::endl;
        return;  // Early exit if the ONNX file is missing
    }

    // Create an inference builder for building the TensorRT engine.
    auto builder = createInferBuilder(gLogger);
    assert(builder != nullptr);  // Ensure the builder is created successfully

    // Use explicit batch size, which is needed for ONNX models.
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    assert(network != nullptr);  // Ensure the network is created successfully

    // Create a builder configuration object to set options like FP16 precision.
    IBuilderConfig* config = builder->createBuilderConfig();
    assert(config != nullptr);  // Ensure the config is created successfully

    // If dynamic shape support is needed, configure the optimization profile.
    if (isDynamicShape) // Only designed for NanoSAM mask decoder
    {
        // Create an optimization profile for dynamic input shapes.
        auto profile = builder->createOptimizationProfile();

        // Set the minimum, optimal, and maximum dimensions for the first input.
        profile->setDimensions(inputNames[1].c_str(), OptProfileSelector::kMIN, Dims3{ 1, 1, 2 });
        profile->setDimensions(inputNames[1].c_str(), OptProfileSelector::kOPT, Dims3{ 1, 1, 2 });
        profile->setDimensions(inputNames[1].c_str(), OptProfileSelector::kMAX, Dims3{ 1, 10, 2 });

        // Set the minimum, optimal, and maximum dimensions for the second input.
        profile->setDimensions(inputNames[2].c_str(), OptProfileSelector::kMIN, Dims2{ 1, 1 });
        profile->setDimensions(inputNames[2].c_str(), OptProfileSelector::kOPT, Dims2{ 1, 1 });
        profile->setDimensions(inputNames[2].c_str(), OptProfileSelector::kMAX, Dims2{ 1, 10 });

        // Add the optimization profile to the builder configuration.
        config->addOptimizationProfile(profile);
    }

    // Enable FP16 mode if specified.
    if (isFP16)
    {
        config->setFlag(BuilderFlag::kFP16);  // Use mixed precision for faster inference
    }

    // Create a parser to convert the ONNX model to a TensorRT network.
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    assert(parser != nullptr);  // Ensure the parser is created successfully

    // Parse the ONNX model from the specified file.
    bool parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(gLogger.getReportableSeverity()));

    // Ensure the CUDA stream used for profiling is valid.
    assert(mCudaStream != nullptr);

    // Serialize the built network into a binary plan for execution.
    IHostMemory* plan{ builder->buildSerializedNetwork(*network, *config) };
    assert(plan != nullptr);  // Ensure the network was serialized successfully

    // Create a runtime object for deserializing the engine.
    mRuntime = createInferRuntime(gLogger);
    assert(mRuntime != nullptr);  // Ensure the runtime is created successfully

    // Deserialize the serialized plan to create an execution engine (TensorRT 10 API).
    mEngine = mRuntime->deserializeCudaEngine(plan->data(), plan->size());
    assert(mEngine != nullptr);  // Ensure the engine was deserialized successfully

    // Create an execution context for running inference.
    mContext = mEngine->createExecutionContext();
    assert(mContext != nullptr);  // Ensure the context is created successfully

    // Clean up resources.
    delete network;
    delete config;
    delete parser;
    delete plan;

    // Initialize the engine with the input and output names.
    initialize(inputNames, outputNames);
}

void EngineTRT::saveEngine(const std::string& engineFilePath) {
    if (mEngine) {
        // Serialize the engine to a binary format.
        IHostMemory* serializedEngine = mEngine->serialize();
        std::ofstream engineFile(engineFilePath, std::ios::binary);
        if (engineFile) {
            // Write the serialized engine data to the specified file.
            engineFile.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
            std::cout << "Serialized engine saved to " << engineFilePath << std::endl;
        }
        // In TensorRT 10, objects are managed via C++ destructors.
        delete serializedEngine;  // Release the serialized engine memory
    }
}

void EngineTRT::deserializeEngine(string engine_name, vector<string> inputNames, vector<string> outputNames)
{
    // Open the engine file in binary mode.
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);  // Trigger an assertion failure if the file cannot be opened
    }

    // Determine the size of the file and read the serialized engine data.
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serializedEngine = new char[size];
    assert(serializedEngine);  // Ensure memory allocation was successful
    file.read(serializedEngine, size);
    file.close();

    // Create a runtime object and deserialize the engine.
    mRuntime = createInferRuntime(gLogger);
    assert(mRuntime);  // Ensure the runtime is created successfully
    mEngine = mRuntime->deserializeCudaEngine(serializedEngine, size);
    mContext = mEngine->createExecutionContext();
    delete[] serializedEngine;  // Free the serialized engine memory

    // Ensure the number of IO tensors matches the expected number of inputs and outputs.
    assert(mEngine->getNbIOTensors() == inputNames.size() + outputNames.size());

    // Initialize the engine with the input and output names.
    initialize(inputNames, outputNames);
}

void EngineTRT::initialize(vector<string> inputNames, vector<string> outputNames)
{
    // Resize the GPU and CPU buffer vectors to accommodate all the engine IO tensors
    const int numIOTensors = mEngine->getNbIOTensors();
    mGpuBuffers.resize(numIOTensors);
    mCpuBuffers.resize(numIOTensors);

    // Loop through all IO tensors to allocate memory and store dimension information
    for (int i = 0; i < numIOTensors; ++i)
    {
        const char* tensorName = mEngine->getIOTensorName(i);
        // Calculate the size required for the tensor based on its dimensions (Dims64)
        Dims64 shape64 = mEngine->getTensorShape(tensorName);
        size_t tensor_size = getSizeByDim64(shape64);
        mBufferBindingSizes.push_back(tensor_size);  // Store the size of the tensor
        mBufferBindingBytes.push_back(tensor_size * sizeof(float));  // Calculate the size in bytes

        // Allocate host memory for the CPU buffer
        mCpuBuffers[i] = new float[tensor_size];

        // Allocate device memory for the GPU buffer
        cudaMalloc(&mGpuBuffers[i], mBufferBindingBytes[i]);

        // Store input and output dimensions separately based on whether the tensor is an input or output
        if (mEngine->getTensorIOMode(tensorName) == TensorIOMode::kINPUT)
        {
            mInputDims.push_back(toDims32(shape64));
        }
        else
        {
            mOutputDims.push_back(toDims32(shape64));
        }
    }

    // Create a CUDA stream for asynchronous operations
    CUDA_CHECK(cudaStreamCreate(&mCudaStream));
}

bool EngineTRT::infer()
{
    // Copy data from host (CPU) input buffers to device (GPU) input buffers asynchronously
    copyInputToDeviceAsync(mCudaStream);

    // Perform inference using TensorRT, passing the GPU buffers
    bool status = mContext->executeV2(mGpuBuffers.data());

    if (!status)
    {
        // If inference fails, print an error message and return false
        cout << "inference error!" << endl;
        return false;
    }

    // Copy the results from device (GPU) output buffers to host (CPU) output buffers asynchronously
    copyOutputToHostAsync(mCudaStream);

    // Return true if inference was successful
    return true;
}

void EngineTRT::copyInputToDeviceAsync(const cudaStream_t& stream)
{
    // Perform asynchronous memory copy from CPU to GPU for the input buffers
    memcpyBuffers(true, false, true, stream);
}

void EngineTRT::copyOutputToHostAsync(const cudaStream_t& stream)
{
    // Calls memcpyBuffers to handle the copying of data from GPU to CPU memory.
    // Arguments: false (do not copy input buffers), true (copy data from device to host),
    // true (perform the copy asynchronously), and the given CUDA stream.
    memcpyBuffers(false, true, true, stream);
}

void EngineTRT::memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t& stream)
{
    // Loop through all IO tensors (inputs and outputs) in the TensorRT engine.
    for (int i = 0; i < mEngine->getNbIOTensors(); i++)
    {
        const char* tensorName = mEngine->getIOTensorName(i);
        const bool isInput = (mEngine->getTensorIOMode(tensorName) == TensorIOMode::kINPUT);
        // Determine the destination and source pointers based on the copy direction.
        void* dstPtr = deviceToHost ? mCpuBuffers[i] : mGpuBuffers[i];
        const void* srcPtr = deviceToHost ? mGpuBuffers[i] : mCpuBuffers[i];
        // Get the size of the buffer in bytes.
        const size_t byteSize = mBufferBindingBytes[i];
        // Set the type of memory copy operation based on the direction.
        const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;

        // Check if the current tensor is an input or output and copy accordingly.
        if ((copyInput && isInput) || (!copyInput && !isInput))
        {
            if (async)
            {
                // Perform asynchronous memory copy using the CUDA stream.
                CUDA_CHECK(cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream));
            }
            else
            {
                // Perform synchronous memory copy.
                CUDA_CHECK(cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType));
            }
        }
    }
}

size_t EngineTRT::getSizeByDim(const Dims& dims)
{
    size_t size = 1;

    // Loop through each dimension and multiply to calculate the total size.
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        // If the dimension is -1 (dynamic), use a predefined maximum size.
        if (dims.d[i] == -1)
            size *= MAX_NUM_PROMPTS;
        else
            size *= dims.d[i];
    }

    return size;
}

void EngineTRT::setInput(Mat& image)
{
    // Extract input dimensions (height and width) from the model's input shape
    const int inputH = mInputDims[0].d[2];
    const int inputW = mInputDims[0].d[3];

    int i = 0;  // Index counter for buffer placement

    // Iterate over each pixel in the input image
    for (int row = 0; row < image.rows; ++row)
    {
        // Pointer to the start of the row in the image data
        uchar* uc_pixel = image.data + row * image.step;

        for (int col = 0; col < image.cols; ++col)
        {
            // Normalizing the pixel values for the RGB channels
            // Convert the BGR image to normalized RGB and store in mCpuBuffers
            mCpuBuffers[0][i] = ((float)uc_pixel[2] / 255.0f - 0.485f) / 0.229f; // Red channel
            mCpuBuffers[0][i + image.rows * image.cols] = ((float)uc_pixel[1] / 255.0f - 0.456f) / 0.224f; // Green channel
            mCpuBuffers[0][i + 2 * image.rows * image.cols] = ((float)uc_pixel[0] / 255.0f - 0.406f) / 0.225f; // Blue channel

            uc_pixel += 3;  // Move to the next pixel
            ++i;  // Increment index
        }
    }
}

void EngineTRT::setInput(float* features, float* imagePointCoords, float* imagePointLabels, float* maskInput, float* hasMaskInput, int numPoints)
{
    // Clean up old buffers and allocate new buffers for the input data
    delete[] mCpuBuffers[1];
    delete[] mCpuBuffers[2];
    mCpuBuffers[1] = new float[numPoints * 2];  // Buffer for point coordinates
    mCpuBuffers[2] = new float[numPoints];      // Buffer for point labels

    // Allocate memory on the GPU for the input data
    cudaMalloc(&mGpuBuffers[1], sizeof(float) * numPoints * 2); // Coordinates
    cudaMalloc(&mGpuBuffers[2], sizeof(float) * numPoints);     // Labels

    // Set the size of the data binding in bytes for TensorRT
    mBufferBindingBytes[1] = sizeof(float) * numPoints * 2;
    mBufferBindingBytes[2] = sizeof(float) * numPoints;

    // Copy input data into CPU buffers
    memcpy(mCpuBuffers[0], features, mBufferBindingBytes[0]);
    memcpy(mCpuBuffers[1], imagePointCoords, sizeof(float) * numPoints * 2);
    memcpy(mCpuBuffers[2], imagePointLabels, sizeof(float) * numPoints);
    memcpy(mCpuBuffers[3], maskInput, mBufferBindingBytes[3]);
    memcpy(mCpuBuffers[4], hasMaskInput, mBufferBindingBytes[4]);

    // Configure TensorRT to use a dynamic input shape (TensorRT 10 IO tensor API)
    mContext->setOptimizationProfileAsync(0, mCudaStream); // Set the optimization profile
    const char* coordsTensorName = mEngine->getIOTensorName(1);
    const char* labelsTensorName = mEngine->getIOTensorName(2);
    mContext->setInputShape(coordsTensorName, Dims3{ 1, numPoints, 2 }); // Set input dimensions for coordinates
    mContext->setInputShape(labelsTensorName, Dims2{ 1, numPoints });    // Set input dimensions for labels
}

void EngineTRT::getOutput(float* features)
{
    // Find the output tensor index and copy its data
    int outIndex = -1;
    const int numIOTensors = mEngine->getNbIOTensors();
    for (int i = 0; i < numIOTensors; ++i)
    {
        const char* tensorName = mEngine->getIOTensorName(i);
        if (mEngine->getTensorIOMode(tensorName) == TensorIOMode::kOUTPUT)
        {
            outIndex = i;
            break;
        }
    }

    if (outIndex < 0)
    {
        std::cerr << "No output tensor found for features." << std::endl;
        return;
    }

    memcpy(features, mCpuBuffers[outIndex], mBufferBindingBytes[outIndex]);
}

void EngineTRT::getOutput(float* iouPrediction, float* lowResolutionMasks)
{
    // Identify the two output tensors and map by size: small -> IoU, large -> low-res masks
    int outIdxA = -1, outIdxB = -1;
    const int numIOTensors = mEngine->getNbIOTensors();
    for (int i = 0; i < numIOTensors; ++i)
    {
        const char* tensorName = mEngine->getIOTensorName(i);
        if (mEngine->getTensorIOMode(tensorName) == TensorIOMode::kOUTPUT)
        {
            if (outIdxA == -1) outIdxA = i;
            else { outIdxB = i; break; }
        }
    }

    if (outIdxA < 0 || outIdxB < 0)
    {
        std::cerr << "Expected two output tensors (IoU and low-res masks)." << std::endl;
        return;
    }

    const size_t bytesA = mBufferBindingBytes[outIdxA];
    const size_t bytesB = mBufferBindingBytes[outIdxB];

    int iouIdx, masksIdx;
    if (bytesA <= bytesB)
    {
        iouIdx = outIdxA;
        masksIdx = outIdxB;
    }
    else
    {
        iouIdx = outIdxB;
        masksIdx = outIdxA;
    }

    memcpy(iouPrediction, mCpuBuffers[iouIdx], mBufferBindingBytes[iouIdx]);
    memcpy(lowResolutionMasks, mCpuBuffers[masksIdx], mBufferBindingBytes[masksIdx]);
}

