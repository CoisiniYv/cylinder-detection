#pragma once

#include "image_types.hpp"

#include <concurrentqueue.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>

namespace XL {

// Thread-safe handoff between acquisition, inference workers and aggregation.
//
// QueueManager owns queue state and blocking/wakeup semantics only. Producers
// and consumers retain ownership of their own threads and lifecycle policy.
// Item counters are maintained alongside ConcurrentQueue so condition-variable
// predicates cannot lose a notification between an initial dequeue attempt and
// entering wait().
class QueueManager {
public:
    using RawType = ImageFrame;
    using ResultType = SingleImageResult;

    QueueManager() = default;
    ~QueueManager();

    QueueManager(const QueueManager&) = delete;
    QueueManager& operator=(const QueueManager&) = delete;

    void start();
    void stop();
    bool stopped() const noexcept { return mStopped.load(std::memory_order_acquire); }

    void clearRaw();
    void clearResult();

    bool pushRaw(const RawType& item);
    bool pushRaw(RawType&& item);
    bool tryPopRaw(RawType& out);
    bool popRawBlocking(RawType& out, int timeout_ms = -1);

    bool pushResult(const ResultType& item);
    bool pushResult(ResultType&& item);
    bool tryPopResult(ResultType& out);
    bool popResultBlocking(ResultType& out, int timeout_ms = -1);

    std::size_t rawSizeApprox() const noexcept;
    std::size_t resultSizeApprox() const noexcept;

private:
    template <typename Item, typename Queue>
    static bool tryPop(Queue& queue, std::atomic<std::size_t>& count, Item& out);

private:
    moodycamel::ConcurrentQueue<RawType> mRawQueue;
    moodycamel::ConcurrentQueue<ResultType> mResultQueue;

    std::atomic<bool> mStopped{false};
    std::atomic<std::size_t> mRawCount{0};
    std::atomic<std::size_t> mResultCount{0};

    std::condition_variable mRawCv;
    mutable std::mutex mRawMutex;

    std::condition_variable mResultCv;
    mutable std::mutex mResultMutex;
};

extern QueueManager g_queue_manager;

} // namespace XL
