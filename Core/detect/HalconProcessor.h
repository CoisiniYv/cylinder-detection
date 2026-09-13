#pragma once

#include <HalconCpp.h>
#include <opencv2/core/cuda.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

class HalconProcessor {
public:
    enum class SaveType {
        PATH_SAVE
    };

    using SaveCompleteCallback = std::function<void(
        const std::string& file_path,
        bool success,
        SaveType type,
        const std::string& message)>;

    struct AsyncStatus {
        int pendingOperations = 0;
        int completedOperations = 0;
        int failedOperations = 0;
    };

    HalconProcessor();
    ~HalconProcessor();

    HalconProcessor(const HalconProcessor&) = delete;
    HalconProcessor& operator=(const HalconProcessor&) = delete;

    bool processImage(
        const cv::cuda::GpuMat& input_image,
        std::string& output_image_path,
        std::vector<std::pair<double, double>>& center_points,
        bool enable_save_to_path = false,
        const std::string& save_path = {},
        std::uint64_t group_id = 0,
        int index_in_group = -1);

    void waitForAsyncOperations();
    AsyncStatus getAsyncStatus() const;

    void setSaveCompleteCallback(SaveCompleteCallback callback);
    void clearSaveCompleteCallback();

    static bool gpuMatToHalconHObject(
        const cv::cuda::GpuMat& gpu_mat,
        HalconCpp::HObject& image);

private:
    struct AsyncSaveTask {
        HalconCpp::HObject image;
        HalconCpp::HObject overlay_region;
        std::string target_path;
        std::promise<bool> promise;
    };

    void asyncWorkerFunction();
    void invokeSaveCallback(
        const std::string& file_path,
        bool success,
        SaveType type,
        const std::string& message);

    std::future<bool> saveOverlayImageAsync(
        const HalconCpp::HObject& image,
        const HalconCpp::HObject& overlay_region,
        const std::string& output_path);

    bool performSaveOperation(
        const HalconCpp::HObject& image,
        const HalconCpp::HObject& overlay_region,
        const std::string& output_path);

    std::string generateSavePathFilename(
        const std::string& save_path,
        std::uint64_t group_id,
        int index_in_group) const;

private:
    std::atomic<int> mPendingAsyncOperations{0};
    std::atomic<int> mCompletedAsyncOperations{0};
    std::atomic<int> mFailedAsyncOperations{0};
    std::atomic<bool> mStopAsyncThread{false};

    std::thread mAsyncWorkerThread;
    std::queue<std::shared_ptr<AsyncSaveTask>> mAsyncSaveQueue;
    mutable std::mutex mQueueMutex;
    std::condition_variable mQueueCondition;

    SaveCompleteCallback mSaveCompleteCallback;
    mutable std::mutex mCallbackMutex;
};
