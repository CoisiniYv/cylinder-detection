#include "detect_thread.hpp"

#include "Utils/Log.hpp"
#include "detection_pipeline.hpp"
#include "queue_manager.hpp"

#include <string>
#include <utility>

namespace XL {
namespace {

void publishResult(SingleImageResult&& result) {
    if (!g_queue_manager.pushResult(std::move(result))) {
        LOGE("failed to publish detection result because result queue is stopped");
    }
}

} // namespace

DetectThread::DetectThread(
    int thread_index,
    const DetectParams& params,
    DetectionContext context)
    : mIndex(thread_index),
      mParams(params),
      mContext(std::move(context)),
      mPipeline(std::make_unique<DetectionPipeline>(thread_index)) {}

DetectThread::~DetectThread() {
    stop();
    join();
}

void DetectThread::start() {
    if (mRunning.exchange(true, std::memory_order_relaxed)) return;
    mThread = std::thread(&DetectThread::run, this);
}

void DetectThread::stop() {
    mRunning.store(false, std::memory_order_relaxed);
}

void DetectThread::join() {
    if (mThread.joinable()) mThread.join();
}

void DetectThread::run() {
    if (!mContext.valid()) {
        LOGE("DetectThread[%d] output context is invalid", mIndex);
        mRunning.store(false, std::memory_order_relaxed);
        return;
    }

    std::string configure_error;
    if (!mPipeline || !mPipeline->configure(mParams, configure_error)) {
        LOGE("DetectThread[%d] pipeline initialization failed: %s",
             mIndex, configure_error.c_str());
        mRunning.store(false, std::memory_order_relaxed);
        return;
    }

    LOGI("DetectThread[%d] ready on GPU %d", mIndex, mPipeline->gpuDevice());

    while (mRunning.load(std::memory_order_relaxed)) {
        ImageFrame frame;
        if (!g_queue_manager.popRawBlocking(frame, 100)) {
            if (g_queue_manager.stopped()) break;
            continue;
        }

        SingleImageResult result = mPipeline->processFrame(frame, mContext);
        if (!result.processing_ok) {
            LOGE("DetectThread[%d] frame processing failed: group=%llu face=%d error=%s",
                 mIndex,
                 static_cast<unsigned long long>(result.meta.group_id),
                 result.meta.index_in_group,
                 result.error_message.c_str());
        }
        publishResult(std::move(result));
    }

    mRunning.store(false, std::memory_order_relaxed);
    LOGI("DetectThread[%d] stopped", mIndex);
}

} // namespace XL
