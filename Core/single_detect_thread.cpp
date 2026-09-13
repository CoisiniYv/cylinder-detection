#include "single_detect_thread.hpp"

#include "Utils/Log.hpp"
#include "detection_pipeline.hpp"

#include <utility>

namespace XL {

SingleDetectThread::SingleDetectThread(DetectionContext context)
    : mContext(std::move(context)),
      mPipeline(std::make_unique<DetectionPipeline>(-1)) {}

SingleDetectThread::~SingleDetectThread() {
    stop();
    join();
}

void SingleDetectThread::start() {
    if (mRunning.exchange(true, std::memory_order_relaxed)) return;
    mThread = std::thread(&SingleDetectThread::worker, this);
}

void SingleDetectThread::stop() {
    mRunning.store(false, std::memory_order_relaxed);
    failPendingTasks("single-image worker is stopping");
    mQueueCondition.notify_all();
}

void SingleDetectThread::join() {
    if (mThread.joinable()) mThread.join();
}

void SingleDetectThread::completeTask(
    const std::shared_ptr<Task>& task,
    bool ok,
    SingleImageResult result,
    std::string error) {
    if (!task) return;
    {
        std::lock_guard<std::mutex> lock(task->mutex);
        if (task->done) return;
        task->ok = ok;
        task->result = std::move(result);
        task->error = std::move(error);
        task->done = true;
    }
    task->condition.notify_one();
}

void SingleDetectThread::failPendingTasks(const std::string& message) {
    std::deque<std::shared_ptr<Task>> pending;
    {
        std::lock_guard<std::mutex> lock(mQueueMutex);
        pending.swap(mTasks);
    }
    for (auto& task : pending) {
        completeTask(task, false, {}, message);
    }
}

bool SingleDetectThread::submitAndWait(
    const DetectParams& params,
    SingleImageResult& out_result,
    std::string& error_message) {
    if (!mContext.valid()) {
        error_message = "single-image output context is invalid";
        return false;
    }
    if (!mRunning.load(std::memory_order_relaxed)) start();

    auto task = std::make_shared<Task>();
    task->params = params;
    {
        std::lock_guard<std::mutex> lock(mQueueMutex);
        if (!mRunning.load(std::memory_order_relaxed)) {
            error_message = "single-image worker is not running";
            return false;
        }
        mTasks.push_back(task);
    }
    mQueueCondition.notify_one();

    std::unique_lock<std::mutex> lock(task->mutex);
    task->condition.wait(lock, [&task] { return task->done; });
    if (!task->ok) {
        error_message = task->error;
        return false;
    }

    out_result = std::move(task->result);
    return true;
}

void SingleDetectThread::worker() {
    while (true) {
        std::shared_ptr<Task> task;
        {
            std::unique_lock<std::mutex> lock(mQueueMutex);
            mQueueCondition.wait(lock, [this] {
                return !mRunning.load(std::memory_order_relaxed) || !mTasks.empty();
            });
            if (!mRunning.load(std::memory_order_relaxed) && mTasks.empty()) break;
            if (mTasks.empty()) continue;
            task = mTasks.front();
            mTasks.pop_front();
        }

        std::string configure_error;
        if (!mPipeline || !mPipeline->configure(task->params, configure_error)) {
            LOGE("single-image pipeline configuration failed: %s", configure_error.c_str());
            completeTask(task, false, {}, configure_error);
            continue;
        }

        SingleImageResult result = mPipeline->processFile(
            task->params.input_image_path,
            mContext);
        if (!result.processing_ok) {
            LOGE("single-image task failed: %s", result.error_message.c_str());
            completeTask(task, false, std::move(result), result.error_message);
            continue;
        }

        completeTask(task, true, std::move(result), {});
    }

    failPendingTasks("single-image worker stopped before processing task");
    mRunning.store(false, std::memory_order_relaxed);
}

} // namespace XL
