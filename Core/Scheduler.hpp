#pragma once

#include "detect_params.hpp"

#include <atomic>
#include <memory>
#include <vector>

namespace XL {

class CameraThread;
class DetectThread;
class GroupManager;

// Owns the multi-face pipeline lifecycle.
//
// Scheduler coordinates workers and their shutdown order; it does not parse
// HTTP requests and it does not implement image-processing algorithms.
class Scheduler {
public:
    Scheduler();
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    bool start(const DetectParams& params, int num_detect_threads);
    bool startFromRuntimeState();

    void stop();
    void join();
    bool isRunning() const noexcept { return mRunning.load(std::memory_order_relaxed); }

    void setGroupTimeoutMs(int timeout_ms) { mGroupTimeoutMs = timeout_ms; }

private:
    void cleanup();

private:
    std::atomic<bool> mRunning{false};
    DetectParams mParams{};
    std::shared_ptr<CameraThread> mCamera;
    std::vector<std::unique_ptr<DetectThread>> mWorkers;
    std::unique_ptr<GroupManager> mGroupMgr;
    int mGroupTimeoutMs = 55000; // 55 seconds
};

} // namespace XL
