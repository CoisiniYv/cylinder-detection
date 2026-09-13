#pragma once

#include "image_types.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace XL {

class CameraThread;

// Aggregates per-face worker results into one four-face inspection result.
// It owns grouping/timeout policy, but not persistence or camera hardware.
class GroupManager {
public:
    GroupManager();
    ~GroupManager();

    GroupManager(const GroupManager&) = delete;
    GroupManager& operator=(const GroupManager&) = delete;

    void start();
    void stop();
    void join();
    bool isRunning() const noexcept { return mRunning.load(std::memory_order_relaxed); }

    void setCameraThread(CameraThread* camera) { mCamera = camera; }
    void setGroupTimeoutMs(int timeout_ms) { mGroupTimeoutMs = timeout_ms; }
    void setOnGroupComplete(std::function<void(const QuadFrameResult&)> callback) {
        mOnComplete = std::move(callback);
    }

private:
    struct GroupState {
        std::array<bool, kQuadImageCount> received{};
        std::array<SingleImageResult, kQuadImageCount> results{};
        std::chrono::steady_clock::time_point started_at = std::chrono::steady_clock::now();
        std::uint64_t group_id = 0;
    };

    void run();
    bool handleResult(const SingleImageResult& result);
    bool finalizeGroup(std::uint64_t group_id, bool timed_out);
    void checkTimeouts();
    static bool isFaceNG(const SingleImageResult& result);

private:
    std::unordered_map<std::uint64_t, GroupState> mGroups;
    std::mutex mMutex;
    std::atomic<bool> mRunning{false};
    std::thread mThread;
    int mGroupTimeoutMs = 0;
    CameraThread* mCamera = nullptr; // non-owning; Scheduler owns the shared camera reference
    std::function<void(const QuadFrameResult&)> mOnComplete;
};

} // namespace XL
