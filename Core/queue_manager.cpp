#include "queue_manager.hpp"

namespace XL {

    QueueManager g_queue_manager; // 全局实例定义

    QueueManager::QueueManager() {}
    QueueManager::~QueueManager() { stop(); }

    void QueueManager::start() {
        mStopped.store(false, std::memory_order_relaxed);
    }

    void QueueManager::stop() {
        mStopped.store(true, std::memory_order_relaxed);
        // 唤醒所有阻塞等待者
        mRawCv.notify_all();
        mResultCv.notify_all();
    }

    void QueueManager::clearRaw() {
        RawType tmp;
        while (mRawQueue.try_dequeue(tmp)) {
            // 释放资源（cv::Mat 内部引用计数自动处理）
            tmp.clear();
        }
    }

    void QueueManager::clearResult() {
        ResultType tmp;
        while (mResultQueue.try_dequeue(tmp)) {
            // 结果中包含的 Mat 也会自动释放
        }
    }

    bool QueueManager::pushRaw(const RawType& item) {
        if (mStopped.load(std::memory_order_relaxed)) return false;
        mRawQueue.enqueue(item);
        mRawCv.notify_one();
        return true;
    }

    bool QueueManager::pushRaw(RawType&& item) {
        if (mStopped.load(std::memory_order_relaxed)) return false;
        mRawQueue.enqueue(std::move(item));
        mRawCv.notify_one();
        return true;
    }

    bool QueueManager::tryPopRaw(RawType& out) {
        if (mRawQueue.try_dequeue(out)) {
            return true;
        }
        return false;
    }

    bool QueueManager::popRawBlocking(RawType& out, int timeout_ms) {
        if (tryPopRaw(out)) return true;
        std::unique_lock<std::mutex> lock(mRawMtx);
        if (timeout_ms < 0) {
            // 无限等待，直到有数据或停止
            while (!mStopped.load(std::memory_order_relaxed)) {
                mRawCv.wait(lock);
                if (tryPopRaw(out)) return true;
            }
            return false;
        }
        else {
            auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
            while (!mStopped.load(std::memory_order_relaxed)) {
                if (mRawCv.wait_until(lock, deadline) == std::cv_status::timeout) {
                    return tryPopRaw(out); // 最后再尝试一次
                }
                if (tryPopRaw(out)) return true;
            }
            return false;
        }
    }

    bool QueueManager::pushResult(const ResultType& item) {
        if (mStopped.load(std::memory_order_relaxed)) return false;
        mResultQueue.enqueue(item);
        mResultCv.notify_one();
        return true;
    }

    bool QueueManager::pushResult(ResultType&& item) {
        if (mStopped.load(std::memory_order_relaxed)) return false;
        mResultQueue.enqueue(std::move(item));
        mResultCv.notify_one();
        return true;
    }

    bool QueueManager::tryPopResult(ResultType& out) {
        if (mResultQueue.try_dequeue(out)) {
            return true;
        }
        return false;
    }

    bool QueueManager::popResultBlocking(ResultType& out, int timeout_ms) {
        if (tryPopResult(out)) return true;
        std::unique_lock<std::mutex> lock(mResultMtx);
        if (timeout_ms < 0) {
            // 无限等待，直到有数据或停止
            while (!mStopped.load(std::memory_order_relaxed)) {
                mResultCv.wait(lock);
                if (tryPopResult(out)) return true;
            }
            return false;
        }
        else {
            auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
            while (!mStopped.load(std::memory_order_relaxed)) {
                if (mResultCv.wait_until(lock, deadline) == std::cv_status::timeout) {
                    return tryPopResult(out); // 最后再尝试一次
                }
                if (tryPopResult(out)) return true;
            }
            return false;
        }
    }

    std::size_t QueueManager::rawSizeApprox() const {
        return mRawQueue.size_approx();
    }

    std::size_t QueueManager::resultSizeApprox() const {
        return mResultQueue.size_approx();
    }

} // namespace XL