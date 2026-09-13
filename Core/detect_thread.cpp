#include "detect_thread.hpp"

#include "Utils/Log.hpp"
#include "detection_pipeline.hpp"
#include "queue_manager.hpp"

#include <exception>
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

    {
        std::lock_guard<std::mutex> lock(mStartupMutex);
        mStartupState = StartupState::Starting;
        mStartupError.clear();
    }

    try {
        mThread = std::thread(&DetectThread::run, this);
    }
    catch (const std::exception& e) {
        mRunning.store(false, std::memory_order_relaxed);
        markStartupFailed(std::string("failed to create worker thread: ") + e.what());
    }
    catch (...) {
        mRunning.store(false, std::memory_order_relaxed);
        markStartupFailed("failed to create worker thread: unknown exception");
    }
}

bool DetectThread::waitUntilReady(std::string& error_message) {
    std::unique_lock<std::mutex> lock(mStartupMutex);
    mStartupCondition.wait(lock, [this] {
        return mStartupState == StartupState::Ready ||
               mStartupState == StartupState::Failed;
    });

    if (mStartupState == StartupState::Ready) {
        error_message.clear();
        return true;
    }

    error_message = mStartupError.empty()
        ? "worker initialization failed"
        : mStartupError;
    return false;
}

void DetectThread::stop() {
    mRunning.store(false, std::memory_order_relaxed);
}

void DetectThread::join() {
    if (mThread.joinable()) mThread.join();
}

void DetectThread::markStartupReady() {
    {
        std::lock_guard<std::mutex> lock(mStartupMutex);
        mStartupState = StartupState::Ready;
        mStartupError.clear();
    }
    mStartupCondition.notify_all();
}

void DetectThread::markStartupFailed(std::string message) {
    {
        std::lock_guard<std::mutex> lock(mStartupMutex);
        mStartupState = StartupState::Failed;
        mStartupError = std::move(message);
    }
    mStartupCondition.notify_all();
}

void DetectThread::run() {
    try {
        if (!mContext.valid()) {
            throw std::runtime_error("output context is invalid");
        }

        std::string configure_error;
        if (!mPipeline || !mPipeline->configure(mParams, configure_error)) {
            if (configure_error.empty()) configure_error = "pipeline initialization failed";
            throw std::runtime_error(configure_error);
        }

        markStartupReady();
        LOGI("DetectThread[%d] ready on GPU %d", mIndex, mPipeline->gpuDevice());
    }
    catch (const std::exception& e) {
        LOGE("DetectThread[%d] startup failed: %s", mIndex, e.what());
        markStartupFailed(e.what());
        mRunning.store(false, std::memory_order_relaxed);
        return;
    }
    catch (...) {
        LOGE("DetectThread[%d] startup failed with unknown exception", mIndex);
        markStartupFailed("unknown startup exception");
        mRunning.store(false, std::memory_order_relaxed);
        return;
    }

    while (mRunning.load(std::memory_order_relaxed)) {
        ImageFrame frame;
        if (!g_queue_manager.popRawBlocking(frame, 100)) {
            if (g_queue_manager.stopped()) break;
            continue;
        }

        SingleImageResult result;
        try {
            result = mPipeline->processFrame(frame, mContext);
        }
        catch (const std::exception& e) {
            result.meta = frame.meta;
            result.processing_ok = false;
            result.error_message = e.what();
        }
        catch (...) {
            result.meta = frame.meta;
            result.processing_ok = false;
            result.error_message = "unknown pipeline exception";
        }

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
