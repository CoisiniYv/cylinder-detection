#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <cstddef>
#include <string>
#include <vector>

// TensorRT engine wrapper used by SpeedSam.
//
// Owns TensorRT objects, host/device buffers and one CUDA transfer stream.
// Public inputs/outputs are mapped by the configured tensor names rather than
// assuming a particular engine enumeration order.
class EngineTRT {
public:
    EngineTRT(
        const std::string& model_path,
        const std::vector<std::string>& input_names,
        const std::vector<std::string>& output_names,
        bool dynamic_shape,
        bool fp16);
    ~EngineTRT();

    EngineTRT(const EngineTRT&) = delete;
    EngineTRT& operator=(const EngineTRT&) = delete;

    bool infer();

    void setInput(const cv::Mat& image);
    void setInput(
        const float* features,
        const float* image_point_coords,
        const float* image_point_labels,
        const float* mask_input,
        const float* has_mask_input,
        int num_points);

    void getOutput(float* iou_prediction, float* low_resolution_masks) const;
    void getOutput(float* features) const;
    void saveEngine(const std::string& engine_file_path) const;

private:
    void build(
        const std::string& onnx_path,
        const std::vector<std::string>& input_names,
        const std::vector<std::string>& output_names,
        bool dynamic_shape,
        bool fp16);
    void deserializeEngine(
        const std::string& engine_name,
        const std::vector<std::string>& input_names,
        const std::vector<std::string>& output_names);
    void initialize(
        const std::vector<std::string>& input_names,
        const std::vector<std::string>& output_names);

    void releaseBuffers() noexcept;
    void releaseResources() noexcept;

    int tensorIndex(const std::string& tensor_name) const;
    int inputIndex(std::size_t logical_index) const;
    int outputIndex(std::size_t logical_index) const;

    void memcpyBuffers(bool copy_input, bool device_to_host, bool async, cudaStream_t stream);
    void copyInputToDeviceAsync(cudaStream_t stream);
    void copyOutputToHostAsync(cudaStream_t stream);

private:
    // Dims and indices are ordered according to the names passed by the caller,
    // not according to TensorRT's internal tensor enumeration order.
    std::vector<std::string> mInputNames;
    std::vector<std::string> mOutputNames;
    std::vector<int> mInputIndices;
    std::vector<int> mOutputIndices;
    std::vector<nvinfer1::Dims> mInputDims;
    std::vector<nvinfer1::Dims> mOutputDims;

    // Buffer vectors remain indexed by TensorRT engine tensor index because
    // executeV2 consumes the binding array in engine order.
    std::vector<void*> mGpuBuffers;
    std::vector<float*> mCpuBuffers;
    std::vector<std::size_t> mBufferBindingBytes;
    std::vector<std::size_t> mBufferBindingSizes;

    cudaStream_t mCudaStream = nullptr;
    nvinfer1::IRuntime* mRuntime = nullptr;
    nvinfer1::ICudaEngine* mEngine = nullptr;
    nvinfer1::IExecutionContext* mContext = nullptr;
};
