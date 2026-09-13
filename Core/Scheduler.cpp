#include "Scheduler.hpp"

#include "Config.hpp"
#include "Utils/Log.hpp"
#include "camera_thread.hpp"
#include "db_utils.hpp"
#include "group_manager.hpp"
#include "queue_manager.hpp"
#include "runtime_state.hpp"
#include "sqlite_helper.hpp"
#include "detect_thread.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace XL {
namespace {

void upsertGroup(SQLiteHelper& db,
                 const std::string& run_id,
                 long long group_id,
                 const std::string& device_id,
                 const std::string& status) {
    db.prepare(
        "INSERT INTO inspection_groups(run_id,group_id,device_id,status) VALUES(?,?,?,?) "
        "ON CONFLICT(run_id,group_id) DO UPDATE SET "
        "device_id=excluded.device_id,status=excluded.status;");
    db.bind(1, run_id);
    db.bind(2, group_id);
    db.bind(3, device_id);
    db.bind(4, status);
    db.step();
}

long long upsertFace(SQLiteHelper& db,
                     const std::string& run_id,
                     long long group_id,
                     int face_index,
                     const SingleImageResult& result) {
    db.prepare(
        "INSERT INTO inspection_faces(run_id,group_id,face_index,saved_path,skeleton_path,original_path) "
        "VALUES(?,?,?,?,?,?) "
        "ON CONFLICT(run_id,group_id,face_index) DO UPDATE SET "
        "saved_path=excluded.saved_path,skeleton_path=excluded.skeleton_path,original_path=excluded.original_path;");
    db.bind(1, run_id);
    db.bind(2, group_id);
    db.bind(3, face_index);
    db.bind(4, result.saved_path);
    db.bind(5, result.skeleton_path);
    db.bind(6, result.original_path);
    db.step();

    // run_id is generated locally and group_id/face_index are numeric, so this
    // lookup is not exposed to request-provided SQL. A parameterized query API
    // is a future SQLiteHelper improvement.
    const std::string sql =
        "SELECT face_id FROM inspection_faces WHERE run_id='" + run_id +
        "' AND group_id=" + std::to_string(group_id) +
        " AND face_index=" + std::to_string(face_index) + ";";
    const auto rows = db.query(sql);
    return rows.empty() ? 0 : std::stoll(rows.front().front());
}

void replaceDefects(SQLiteHelper& db, long long face_id, const std::vector<Detection>& detections) {
    db.prepare("DELETE FROM defect_detections WHERE face_id=?;");
    db.bind(1, face_id);
    db.step();

    for (const auto& detection : detections) {
        db.prepare(
            "INSERT INTO defect_detections(face_id,label_id,confidence,bbox_x,bbox_y,bbox_w,bbox_h,length,area) "
            "VALUES(?,?,?,?,?,?,?,?,?);");
        db.bind(1, face_id);
        db.bind(2, static_cast<int>(detection.label_id));
        db.bind(3, static_cast<double>(detection.confidence));
        db.bind(4, static_cast<double>(detection.box.x));
        db.bind(5, static_cast<double>(detection.box.y));
        db.bind(6, static_cast<double>(detection.box.w));
        db.bind(7, static_cast<double>(detection.box.h));

        const double length = detection.length > 0.0
            ? detection.length
            : std::max(static_cast<double>(detection.box.w), static_cast<double>(detection.box.h));
        const double area = detection.area > 0.0
            ? detection.area
            : static_cast<double>(detection.box.w) * static_cast<double>(detection.box.h);
        db.bind(8, length);
        db.bind(9, area);
        db.step();
    }
}

bool saveGroupResult(const QuadFrameResult& group_result,
                     const std::string& run_id,
                     const std::string& device_id,
                     const std::string& db_file,
                     const std::string& output_root) {
    try {
        SQLiteHelper db(db_file);

        const auto group_it = std::find_if(
            group_result.results.begin(),
            group_result.results.end(),
            [](const SingleImageResult& result) { return result.meta.group_id != 0; });
        if (group_it == group_result.results.end()) {
            LOGE("cannot persist group without group_id");
            return false;
        }

        const std::uint64_t group_id_u64 = group_it->meta.group_id;
        const long long group_id = static_cast<long long>(group_id_u64);
        const bool is_ng = std::any_of(
            group_result.results.begin(),
            group_result.results.end(),
            [](const SingleImageResult& result) { return !result.detections.empty(); });
        const std::string status = is_ng ? "NG" : "GOOD";

        db.begin();
        upsertGroup(db, run_id, group_id, device_id, status);

        const std::filesystem::path group_dir =
            std::filesystem::path(output_root) / run_id / std::to_string(group_id);

        for (std::size_t i = 0; i < kQuadImageCount; ++i) {
            SingleImageResult result = group_result.results[i];
            if (result.original_path.empty()) {
                result.original_path =
                    (group_dir /
                     ("original_g" + std::to_string(group_id) + "_i" + std::to_string(i) + ".png"))
                        .string();
            }

            const long long face_id = upsertFace(db, run_id, group_id, static_cast<int>(i), result);
            if (face_id <= 0) {
                db.rollback();
                LOGE("failed to persist face: run=%s group=%llu face=%zu",
                     run_id.c_str(),
                     static_cast<unsigned long long>(group_id_u64),
                     i);
                return false;
            }
            replaceDefects(db, face_id, result.detections);
        }

        db.commit();
        LOGI("group persisted: run=%s group=%llu device=%s status=%s",
             run_id.c_str(),
             static_cast<unsigned long long>(group_id_u64),
             device_id.c_str(),
             status.c_str());
        return true;
    }
    catch (const std::exception& e) {
        LOGE("persist group failed: %s", e.what());
        return false;
    }
}

} // namespace

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

    mParams = params;
    g_queue_manager.start();

    if (!g_camera_thread) g_camera_thread = std::make_shared<CameraThread>();
    if (!g_camera_thread->isRunning()) {
        const Config& config = *g_runtime_state.config;
        if (!g_camera_thread->start(
                mParams.delay_ms,
                mParams.device_id,
                config.slidePort,
                config.slideAxisId,
                config.slideTimeoutMs,
                true)) {
            LOGE("CameraThread startup failed");
            g_queue_manager.stop();
            return false;
        }
    }
    mCamera = g_camera_thread;

    try {
        mWorkers.reserve(static_cast<std::size_t>(num_detect_threads));
        for (int i = 0; i < num_detect_threads; ++i) {
            auto worker = std::make_unique<DetectThread>(i, mParams);
            worker->start();
            mWorkers.emplace_back(std::move(worker));
        }

        const Config& config = *g_runtime_state.config;
        const std::string db_file = config.dbPath.empty() ? "my_inspection.db" : config.dbPath;
        const std::string run_id = g_runtime_state.run_id;
        const std::string output_root = config.outputdir;

        mGroupMgr = std::make_unique<GroupManager>();
        mGroupMgr->setCameraThread(mCamera.get());
        mGroupMgr->setGroupTimeoutMs(mGroupTimeoutMs);
        mGroupMgr->setOnGroupComplete(
            [this, db_file, run_id, output_root](const QuadFrameResult& result) {
                if (!saveGroupResult(result, run_id, mParams.device_id, db_file, output_root)) {
                    LOGE("group persistence callback failed");
                }
            });
        mGroupMgr->start();
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

    // Producers stop before queues so no new frames arrive during teardown.
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
