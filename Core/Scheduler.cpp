#include "Scheduler.hpp"

#include <algorithm>
#include <filesystem>
#include <json/json.h>

#include "Utils/Common.hpp"
#include "db_utils.hpp"

namespace XL {

static Json::Value detection_to_json(const Detection& d) {
    Json::Value j;
    j["label_id"] = d.label_id;
    j["confidence"] = d.confidence;
    Json::Value box;
    box["x"] = d.box.x;
    box["y"] = d.box.y;
    box["w"] = d.box.w;
    box["h"] = d.box.h;
    j["bbox"] = box;
    return j;
}

static void upsert_group(SQLiteHelper& db, const std::string& run_id, long long gid,
                         const std::string& device_id, const std::string& status) {
    db.prepare("SELECT 1 FROM inspection_groups WHERE run_id=? AND group_id=?;");
    db.bind(1, run_id);
    db.bind(2, gid);
    // SQLiteHelper currently exposes row-returning queries only through query(), so use
    // INSERT ... ON CONFLICT to avoid constructing SQL with runtime strings.
    db.step();

    db.prepare(
        "INSERT INTO inspection_groups(run_id,group_id,device_id,status) VALUES(?,?,?,?) "
        "ON CONFLICT(run_id,group_id) DO UPDATE SET device_id=excluded.device_id,status=excluded.status;");
    db.bind(1, run_id);
    db.bind(2, gid);
    db.bind(3, device_id);
    db.bind(4, status);
    db.step();
}

static long long upsert_face(SQLiteHelper& db, const std::string& run_id, long long gid,
                             int face_index, const SingleImageResult& r) {
    db.prepare(
        "INSERT INTO inspection_faces(run_id,group_id,face_index,saved_path,skeleton_path,original_path) "
        "VALUES(?,?,?,?,?,?) "
        "ON CONFLICT(run_id,group_id,face_index) DO UPDATE SET "
        "saved_path=excluded.saved_path,skeleton_path=excluded.skeleton_path,original_path=excluded.original_path;");
    db.bind(1, run_id);
    db.bind(2, gid);
    db.bind(3, face_index);
    db.bind(4, r.saved_path);
    db.bind(5, r.skeleton_path);
    db.bind(6, r.original_path);
    db.step();

    const std::string sql =
        "SELECT face_id FROM inspection_faces WHERE run_id='" + run_id +
        "' AND group_id=" + std::to_string(gid) +
        " AND face_index=" + std::to_string(face_index) + ";";
    auto rows = db.query(sql);
    return rows.empty() ? 0 : std::stoll(rows[0][0]);
}

static void replace_defects(SQLiteHelper& db, long long face_id, const std::vector<Detection>& dets) {
    db.prepare("DELETE FROM defect_detections WHERE face_id=?;");
    db.bind(1, face_id);
    db.step();

    for (const auto& d : dets) {
        db.prepare(
            "INSERT INTO defect_detections(face_id,label_id,confidence,bbox_x,bbox_y,bbox_w,bbox_h,length,area) "
            "VALUES(?,?,?,?,?,?,?,?,?);");
        db.bind(1, face_id);
        db.bind(2, d.label_id);
        db.bind(3, static_cast<double>(d.confidence));
        db.bind(4, static_cast<double>(d.box.x));
        db.bind(5, static_cast<double>(d.box.y));
        db.bind(6, static_cast<double>(d.box.w));
        db.bind(7, static_cast<double>(d.box.h));
        const double length = d.length > 0.0 ? d.length : std::max<double>(d.box.w, d.box.h);
        const double area = d.area > 0.0 ? d.area : static_cast<double>(d.box.w) * d.box.h;
        db.bind(8, length);
        db.bind(9, area);
        db.step();
    }
}

static bool save_group_db(const QuadFrameResult& qres, const std::string& run_id,
                          const std::string& device_id, const std::string& dbfile) {
    try {
        SQLiteHelper db(dbfile);
        bool ng = false;
        for (const auto& result : qres.results) {
            if (!result.detections.empty()) {
                ng = true;
                break;
            }
        }

        const std::uint64_t gid_u64 = qres.source.group_id ? qres.source.group_id : qres.results[0].meta.group_id;
        const long long gid = static_cast<long long>(gid_u64);
        const std::string status = ng ? "NG" : "GOOD";

        db.begin();
        upsert_group(db, run_id, gid, device_id, status);

        for (int i = 0; i < static_cast<int>(kQuadImageCount); ++i) {
            SingleImageResult res = qres.results[i];
            if (res.original_path.empty()) {
                const auto* cfg = g_server_state.config;
                const std::filesystem::path base = cfg ? cfg->outputdir : std::string();
                const std::filesystem::path dir = base / run_id / std::to_string(gid);
                res.original_path = (dir / ("original_g" + std::to_string(gid) + "_i" + std::to_string(i) + ".png")).string();
            }

            const long long face_id = upsert_face(db, run_id, gid, i, res);
            if (face_id <= 0) {
                db.rollback();
                LOGE("写入面记录失败: run_id=%s, group=%llu, face_index=%d",
                     run_id.c_str(), static_cast<unsigned long long>(gid_u64), i);
                return false;
            }
            replace_defects(db, face_id, res.detections);
        }

        db.commit();
        LOGI("组结果已写入数据库: run_id=%s, group_id=%llu, device=%s, status=%s",
             run_id.c_str(), static_cast<unsigned long long>(gid_u64), device_id.c_str(), status.c_str());
        return true;
    }
    catch (const std::exception& e) {
        LOGE("写入数据库异常: %s", e.what());
        return false;
    }
}

Scheduler::Scheduler() = default;
Scheduler::~Scheduler() {
    stop();
    join();
}

bool Scheduler::start(const DetectParams& params, int num_detect_threads) {
    if (mRunning.load(std::memory_order_relaxed)) {
        LOGE("Scheduler 已在运行，忽略重复启动");
        return false;
    }

    mParams = params;
    g_queue_manager.start();

    if (!g_camera_thread) g_camera_thread = std::make_shared<CameraThread>();
    if (!g_camera_thread->isRunning()) {
        const auto* cfg = g_server_state.config;
        const std::string slide_port = cfg ? cfg->slidePort : std::string();
        const int slide_axis = cfg ? cfg->slideAxisId : 0;
        const int slide_timeout_ms = cfg ? cfg->slideTimeoutMs : 20000;
        if (!g_camera_thread->start(mParams.delay_ms, mParams.device_id,
                                    slide_port, slide_axis, slide_timeout_ms, true)) {
            LOGE("CameraThread 启动失败，停止调度器");
            g_queue_manager.stop();
            return false;
        }
    }
    mCamera = g_camera_thread;

    if (num_detect_threads <= 0) num_detect_threads = 1;
    mWorkers.reserve(num_detect_threads);
    for (int i = 0; i < num_detect_threads; ++i) {
        auto worker = std::make_unique<DetectThread>(i, mParams);
        worker->start();
        mWorkers.emplace_back(std::move(worker));
    }

    const auto* cfg = g_server_state.config;
    const std::string dbfile = (cfg && !cfg->dbPath.empty()) ? cfg->dbPath : "my_inspection.db";
    const std::string run_id = g_server_state.run_id;
    if (run_id.empty()) {
        LOGE("run_id 未初始化，拒绝启动 Scheduler");
        stop();
        return false;
    }

    mGroupMgr = std::make_unique<GroupManager>();
    mGroupMgr->setCameraThread(mCamera.get());
    mGroupMgr->setGroupTimeoutMs(mGroupTimeoutMs);
    mGroupMgr->setOnGroupComplete([this, dbfile, run_id](const QuadFrameResult& qres) {
        if (!save_group_db(qres, run_id, mParams.device_id, dbfile)) {
            LOGE("保存组结果到数据库失败");
        }
    });
    mGroupMgr->start();

    mRunning.store(true, std::memory_order_relaxed);
    LOGI("Scheduler启动完成: camera=1, workers=%d, group_manager=1", num_detect_threads);
    return true;
}

bool Scheduler::startFromServerState(int num_detect_threads) {
    if (!g_server_state.has_task.load(std::memory_order_relaxed)) {
        LOGE("ServerState未检测到任务，无法启动 Scheduler");
        return false;
    }
    if (g_server_state.config) {
        setGroupTimeoutMs(g_server_state.config->groupTimeoutMs);
        if (num_detect_threads <= 0) num_detect_threads = g_server_state.config->detectThreads;
    }
    return start(g_server_state.last_params, num_detect_threads);
}

void Scheduler::stop() {
    if (!mRunning.load(std::memory_order_relaxed) && mWorkers.empty() && !mGroupMgr && !mCamera) return;

    if (mCamera) mCamera->stop();
    g_queue_manager.stop();
    if (mGroupMgr) mGroupMgr->stop();
    for (auto& worker : mWorkers) if (worker) worker->stop();

    join();
    cleanup();
    mRunning.store(false, std::memory_order_relaxed);
    LOGI("Scheduler 已停止");
}

void Scheduler::join() {
    if (mCamera) mCamera->join();
    for (auto& worker : mWorkers) if (worker) worker->join();
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
