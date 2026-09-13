#pragma once

#include "detect/HalconProcessor.h"
#include "detect/sam.h"
#include "detect/trtyolo_slice.hpp"
#include "detect_params.hpp"
#include "image_types.hpp"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace XL {

// Dedicated serialized worker for the single-image HTTP API.
// Model instances are cached between requests and recreated only when their
// model path or CUDA device changes.
class SingleDetectThread {
public:
    SingleDetectThread();
    ~SingleDetectThread();

    SingleDetectThread(const SingleDetectThread&) = delete;
    SingleDetectThread& operator=(const SingleDetectThread&) = delete;

    void start();
    void stop();
    void join();
    bool isRunning() const noexcept { return mRunning.load(std::memory_order_relaxed); }

    bool submitAndWait(
        const DetectParams& params,
        SingleImageResult& out_result,
        std::string& error_message);

private:
    struct Task {
        DetectParams params;
        SingleImageResult result;
        bool ok = false;
        bool done = false;
        std::string error;
        std::mutex mutex;
        std::condition_variable condition;
    };

    void worker();
    void failPendingTasks(const std::string& message);
    static void completeTask(
        const std::shared_ptr<Task>& task,
        bool ok,
        SingleImageResult result,
        std::string error);

private:
    std::thread mThread;
    std::atomic<bool> mRunning{false};
    std::mutex mQueueMutex;
    std::condition_variable mQueueCondition;
    std::deque<std::shared_ptr<Task>> mTasks;

    std::unique_ptr<trtyolo::SliceDetector> mSliceDetector;
    std::string mLastTrtEngineFile;

    std::unique_ptr<SamSegmenter> mSam;
    bool mSamReady = false;
    std::string mLastEncoderPath;
    std::string mLastDecoderPath;

    int mCurrentGpuDevice = -1;
    HalconProcessor mHalcon;
};

} // namespace XL
