#include "Scheduler.hpp"

#include "Config.hpp"
#include "Utils/Log.hpp"
#include "camera_thread.hpp"
#include "detect_thread.hpp"
#include "detection_context.hpp"
#include "group_manager.hpp"
#include "inspection_repository.hpp"
#include "queue_manager.hpp"
#include "runtime_state.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace XL {

Scheduler::Scheduler() = default;

Scheduler::~Scheduler() {
    stop();
}

bool Scheduler::start(const DetectParams& params, int num_detect_threads) {
    if (mRunning.load(std::memory_order_relaxed)) {
        LOGE("Scheduler is already running");
        return false;
    }
    if (num_detect_threads <= 0) {
        LOGE("detect thread count must be positive");
        return false;
    }
    if (!g_runtime_state.config || g_runtime_state.run_id.empty()) {
        LOGE("runtime state is not initialized");
        return false;
    }

    const Config& config = *g_runtime_state.config;
    const DetectionContext detection_context{config.outputdir, g_runtime_state.run_id};
    if (!detection_context.valid()) {
        LOGE("detection output context is invalid");
        return false;
    }

    mParams = params;
    g_queue_manager.start();

    if (!g_camera_thread) g_camera_thread = std::make_shared<CameraThread>();
    mCamera = g_camera_thread;

    try {
        // Start all workers first so model initialization can run concurrently,
        // then wait until every worker reports READY before touching hardware.
        mWorkers.reserve(static_cast<std::size_t>(num_detect_threads));
        for (int i = 0; i < num_detect_threads; ++i) {
            auto worker = std::make_unique<DetectThread>(
                i,
                mParams,
                detection_context);
            worker->start();
            mWorkers.emplace_back(std::move(worker));
        }

        for (const auto& worker : mWorkers) {
            std::string startup_error;
            if (!worker || !worker->waitUntilReady(startup_error)) {
                const int worker_index = worker ? worker->index() : -1;
                throw std::runtime_error(
                    "DetectThread[" + std::to_string(worker_index) +
                    "] failed to initialize: " + startup_error);
            }
        }

        const std::string database_file =
            config.dbPath.empty() ? "my_inspection.db" : config.dbPath;
        auto repository = std::make_shared<InspectionRepository>(
            database_file,
            config.outputdir,
            g_runtime_state.run_id,
            mParams.device_id);

        mGroupMgr = std::make_unique<GroupManager>();
        mGroupMgr->setCameraThread(mCamera.get());
        mGroupMgr->setGroupTimeoutMs(mGroupTimeoutMs);
        mGroupMgr->setOnGroupComplete(
            [repository](const QuadFrameResult& result) {
                if (!repository->save(result)) {
                    LOGE("group persistence callback failed");
                }
            });
        mGroupMgr->start();

        // Hardware acquisition starts last. If CUDA/TensorRT initialization
        // fails, no cylinder is moved and no camera frames are captured.
        if (!mCamera->isRunning() && !mCamera->start(
                mParams.delay_ms,
                mParams.device_id,
                config.slidePort,
                config.slideAxisId,
                config.slideTimeoutMs,
                true)) {
            throw std::runtime_error("CameraThread startup failed");
        }
    }
    catch (const std::exception& e) {
        LOGE("Scheduler startup exception: %s", e.what());
        stop();
        return false;
    }

    mRunning.store(true, std::memory_order_relaxed);
    LOGI("Scheduler started: camera=1 workers=%d group_manager=1", num_detect_threads);
    return true;
}

bool Scheduler::startFromRuntimeState() {
    if (!g_runtime_state.has_task.load(std::memory_order_relaxed)) {
        LOGE("no task is available in RuntimeState");
        return false;
    }
    if (!g_runtime_state.config) {
        LOGE("RuntimeState has no Config");
        return false;
    }

    setGroupTimeoutMs(g_runtime_state.config->groupTimeoutMs);
    return start(g_runtime_state.last_params, g_runtime_state.config->detectThreads);
}

void Scheduler::stop() {
    if (!mRunning.load(std::memory_order_relaxed) && mWorkers.empty() && !mGroupMgr && !mCamera) {
        return;
    }

    // Stop producers/consumers before marking queues stopped, then wake any
    // blocking queue waits so every owned thread can exit and be joined.
    if (mCamera) mCamera->stop();
    if (mGroupMgr) mGroupMgr->stop();
    for (auto& worker : mWorkers) {
        if (worker) worker->stop();
    }
    g_queue_manager.stop();

    join();
    cleanup();
    mRunning.store(false, std::memory_order_relaxed);
    LOGI("Scheduler stopped");
}

void Scheduler::join() {
    if (mCamera) mCamera->join();
    for (auto& worker : mWorkers) {
        if (worker) worker->join();
    }
    if (mGroupMgr) mGroupMgr->join();
}

void Scheduler::cleanup() {
    g_queue_manager.clearRaw();
    g_queue_manager.clearResult();
    mWorkers.clear();
    mGroupMgr.reset();
    mCamera.reset();
}

} // namespace XL
