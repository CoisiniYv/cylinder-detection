#include "queue_manager.hpp"

#include <utility>

namespace XL {

QueueManager g_queue_manager;

QueueManager::~QueueManager() {
    stop();
}

void QueueManager::start() {
    mStopped.store(false, std::memory_order_release);
}

void QueueManager::stop() {
    mStopped.store(true, std::memory_order_release);
    mRawCv.notify_all();
    mResultCv.notify_all();
}

void QueueManager::clearRaw() {
    RawType item;
    while (mRawQueue.try_dequeue(item)) {
        item.clear();
    }
    mRawCount.store(0, std::memory_order_release);
}

void QueueManager::clearResult() {
    ResultType item;
    while (mResultQueue.try_dequeue(item)) {
    }
    mResultCount.store(0, std::memory_order_release);
}

template <typename Item, typename Queue>
bool QueueManager::tryPop(Queue& queue, std::atomic<std::size_t>& count, Item& out) {
    if (!queue.try_dequeue(out)) return false;

    // All queue mutations go through QueueManager. The guard avoids wrapping
    // an unsigned counter if clear*() races a final consumer during teardown.
    std::size_t current = count.load(std::memory_order_relaxed);
    while (current > 0 &&
           !count.compare_exchange_weak(
               current,
               current - 1,
               std::memory_order_release,
               std::memory_order_relaxed)) {
    }
    return true;
}

bool QueueManager::pushRaw(const RawType& item) {
    if (stopped()) return false;
    mRawQueue.enqueue(item);
    mRawCount.fetch_add(1, std::memory_order_release);
    mRawCv.notify_one();
    return true;
}

bool QueueManager::pushRaw(RawType&& item) {
    if (stopped()) return false;
    mRawQueue.enqueue(std::move(item));
    mRawCount.fetch_add(1, std::memory_order_release);
    mRawCv.notify_one();
    return true;
}

bool QueueManager::tryPopRaw(RawType& out) {
    return tryPop(mRawQueue, mRawCount, out);
}

bool QueueManager::popRawBlocking(RawType& out, int timeout_ms) {
    if (tryPopRaw(out)) return true;

    std::unique_lock<std::mutex> lock(mRawMutex);
    const auto ready = [this] {
        return stopped() || mRawCount.load(std::memory_order_acquire) > 0;
    };

    if (timeout_ms < 0) {
        while (!stopped()) {
            mRawCv.wait(lock, ready);
            if (tryPopRaw(out)) return true;
        }
        return tryPopRaw(out);
    }

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!stopped()) {
        if (!mRawCv.wait_until(lock, deadline, ready)) return tryPopRaw(out);
        if (tryPopRaw(out)) return true;
        if (std::chrono::steady_clock::now() >= deadline) return false;
    }
    return tryPopRaw(out);
}

bool QueueManager::pushResult(const ResultType& item) {
    if (stopped()) return false;
    mResultQueue.enqueue(item);
    mResultCount.fetch_add(1, std::memory_order_release);
    mResultCv.notify_one();
    return true;
}

bool QueueManager::pushResult(ResultType&& item) {
    if (stopped()) return false;
    mResultQueue.enqueue(std::move(item));
    mResultCount.fetch_add(1, std::memory_order_release);
    mResultCv.notify_one();
    return true;
}

bool QueueManager::tryPopResult(ResultType& out) {
    return tryPop(mResultQueue, mResultCount, out);
}

bool QueueManager::popResultBlocking(ResultType& out, int timeout_ms) {
    if (tryPopResult(out)) return true;

    std::unique_lock<std::mutex> lock(mResultMutex);
    const auto ready = [this] {
        return stopped() || mResultCount.load(std::memory_order_acquire) > 0;
    };

    if (timeout_ms < 0) {
        while (!stopped()) {
            mResultCv.wait(lock, ready);
            if (tryPopResult(out)) return true;
        }
        return tryPopResult(out);
    }

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!stopped()) {
        if (!mResultCv.wait_until(lock, deadline, ready)) return tryPopResult(out);
        if (tryPopResult(out)) return true;
        if (std::chrono::steady_clock::now() >= deadline) return false;
    }
    return tryPopResult(out);
}

std::size_t QueueManager::rawSizeApprox() const noexcept {
    return mRawCount.load(std::memory_order_acquire);
}

std::size_t QueueManager::resultSizeApprox() const noexcept {
    return mResultCount.load(std::memory_order_acquire);
}

} // namespace XL
