#pragma once

#include "detect_params.hpp"
#include "detection_context.hpp"
#include "image_types.hpp"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace XL {

class DetectionPipeline;

// Serialized worker for the single-image HTTP API.
//
// This class owns task queuing and synchronization only. Model caching, CUDA
// placement and image-processing steps belong to DetectionPipeline.
class SingleDetectThread {
public:
    explicit SingleDetectThread(DetectionContext context);
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
    DetectionContext mContext;
    std::unique_ptr<DetectionPipeline> mPipeline;

    std::thread mThread;
    std::atomic<bool> mRunning{false};
    std::mutex mQueueMutex;
    std::condition_variable mQueueCondition;
    std::deque<std::shared_ptr<Task>> mTasks;
};

} // namespace XL
