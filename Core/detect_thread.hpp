#pragma once

#include "detect_params.hpp"
#include "detection_context.hpp"

#include <atomic>
#include <memory>
#include <thread>

namespace XL {

class DetectionPipeline;

// One worker in the multi-face pipeline.
//
// Responsibilities are intentionally narrow: own one worker thread, consume
// ImageFrame objects from QueueManager, delegate image processing to
// DetectionPipeline, and publish SingleImageResult objects.
class DetectThread {
public:
    DetectThread(
        int thread_index,
        const DetectParams& params,
        DetectionContext context);
    ~DetectThread();

    DetectThread(const DetectThread&) = delete;
    DetectThread& operator=(const DetectThread&) = delete;

    void start();
    void stop();
    void join();
    bool isRunning() const noexcept { return mRunning.load(std::memory_order_relaxed); }
    int index() const noexcept { return mIndex; }

private:
    void run();

private:
    int mIndex = 0;
    std::thread mThread;
    std::atomic<bool> mRunning{false};
    DetectParams mParams;
    DetectionContext mContext;
    std::unique_ptr<DetectionPipeline> mPipeline;
};

} // namespace XL
