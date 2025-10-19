#include "Scheduler.hpp"
#include <filesystem>
#include <fstream>
#include <json/json.h>

namespace XL {

static Json::Value detection_to_json(const Detection& d) {
    Json::Value j;
    j["label_id"] = d.label_id;
    j["label"] = d.label;
    j["confidence"] = d.confidence;
    Json::Value box;
    box["x"] = d.box.x; box["y"] = d.box.y; box["w"] = d.box.w; box["h"] = d.box.h;
    j["bbox"] = box;
    return j;
}

static Json::Value single_result_to_json(const SingleImageResult& r) {
    Json::Value j;
    Json::Value meta;
    meta["timestamp_ms"] = static_cast<Json::Int64>(r.meta.timestamp_ms);
    meta["device_id"] = r.meta.device_id;
    meta["group_id"] = static_cast<Json::UInt64>(r.meta.group_id);
    meta["index_in_group"] = r.meta.index_in_group;
    meta["sequence_id"] = static_cast<Json::UInt64>(r.meta.sequence_id);
    j["meta"] = meta;

    j["saved_path"] = r.saved_path;
    j["thumbnail_path"] = r.thumbnail_path;

    Json::Value dets(Json::arrayValue);
    for (const auto& d : r.detections) dets.append(detection_to_json(d));
    j["detections"] = dets;
    return j;
}

static bool save_group_json(const QuadFrameResult& qres, const std::string& device_id, const std::string& outdir) {
    // 组状态：任一面有检测则视为 NG
    bool ng = false;
    for (int i = 0; i < static_cast<int>(kQuadImageCount); ++i) {
        if (!qres.results[i].detections.empty()) { ng = true; break; }
    }

    Json::Value root;
    root["device_id"] = device_id;
    // 取第0面或 source.group_id 作为组ID
    std::uint64_t gid = qres.source.group_id ? qres.source.group_id : qres.results[0].meta.group_id;
    root["group_id"] = static_cast<Json::UInt64>(gid);
    root["status"] = ng ? "NG" : "GOOD";

    Json::Value faces(Json::arrayValue);
    for (int i = 0; i < static_cast<int>(kQuadImageCount); ++i) {
        faces.append(single_result_to_json(qres.results[i]));
    }
    root["faces"] = faces;

    // 文件名：device_group_timestamp.json
    const auto ts = getCurTimestamp();
    std::filesystem::path dir(outdir.empty() ? std::filesystem::path("./output") : std::filesystem::path(outdir));
    std::error_code ec;
    if (!std::filesystem::exists(dir)) {
        std::filesystem::create_directories(dir, ec);
        if (ec) {
            LOGE("创建输出目录失败: %s", ec.message().c_str());
            return false;
        }
    }
    char fname[256];
    snprintf(fname, sizeof(fname), "%s_%llu_%lld.json", device_id.empty() ? "device" : device_id.c_str(), (unsigned long long)gid, (long long)ts);
    std::filesystem::path outpath = dir / fname;

    Json::StreamWriterBuilder w;
    w["indentation"] = "";
    std::ofstream ofs(outpath.string(), std::ios::binary);
    if (!ofs.is_open()) {
        LOGE("写入 JSON 失败：%s", outpath.string().c_str());
        return false;
    }
    ofs << Json::writeString(w, root);
    ofs.close();

    LOGI("组结果已保存: %s", outpath.string().c_str());
    return true;
}

Scheduler::Scheduler() {}
Scheduler::~Scheduler() { stop(); join(); }

bool Scheduler::start(const DetectParams& params, int num_detect_threads) {
    if (mRunning.load(std::memory_order_relaxed)) {
        LOGE("Scheduler 已在运行，忽略重复启动");
        return false;
    }

    // 保存参数（传递给工作线程）
    mParams = params;

    // 启动队列管理（唤醒阻塞）
    g_queue_manager.start();

    // 创建并启动相机线程
    mCamera = std::make_unique<CameraThread>();
    if (!mCamera->start(mParams.delay_ms, /*device_id*/mParams.device_id)) {
        LOGE("CameraThread 启动失败，停止调度器");
        g_queue_manager.stop();
        mCamera.reset();
        return false;
    }

    // 创建并启动检测线程
    if (num_detect_threads <= 0) num_detect_threads = 1 ;
    mWorkers.reserve(num_detect_threads);
    for (int i = 0; i < num_detect_threads; ++i) {
        auto worker = std::make_unique<DetectThread>(i, mParams);
        worker->start();
        mWorkers.emplace_back(std::move(worker));
    }

    // 创建并启动组管理线程
    mGroupMgr = std::make_unique<GroupManager>();
    mGroupMgr->setCameraThread(mCamera.get());
    mGroupMgr->setGroupTimeoutMs(mGroupTimeoutMs);
    mGroupMgr->setOnGroupComplete([this](const QuadFrameResult& qres) {
        // 保存到 JSON 文件（使用配置的 uploadDir 作为输出目录）
        extern ServerState g_server_state;
        std::string outdir;
        if (g_server_state.config && !g_server_state.config->outputdir.empty()) {
            outdir = g_server_state.config->outputdir;
        } else {
            outdir = "./output";
        }
        save_group_json(qres, mParams.device_id, outdir);
        // TODO: 也可以在此处通过 HTTP 上报，若配置提供 reportUrl
    });
    mGroupMgr->start();

    mRunning.store(true, std::memory_order_relaxed);
    LOGI("Scheduler 启动完成：camera=1, workers=%d, group_manager=1", num_detect_threads);
    return true;
}

bool Scheduler::startFromServerState(int num_detect_threads) {
    // 若 Server 提供了最近一次任务参数，则从中启动
    extern ServerState g_server_state;
    if (!g_server_state.has_task.load(std::memory_order_relaxed)) {
        LOGE("ServerState 未检测到任务，无法启动 Scheduler");
        return false;
    }
    return start(g_server_state.last_params, num_detect_threads);
}

void Scheduler::stop() {
    if (!mRunning.load(std::memory_order_relaxed)) return;

    LOGI("Scheduler 停止中...");

    // 先停止相机，避免继续入队原始帧
    if (mCamera) mCamera->stop();

    // 停止队列管理器，唤醒所有阻塞的 pop
    g_queue_manager.stop();

    // 停止组管理器与检测线程
    if (mGroupMgr) mGroupMgr->stop();
    for (auto& w : mWorkers) { if (w) w->stop(); }

    // 等待线程退出
    join();

    // 清理资源
    cleanup();

    mRunning.store(false, std::memory_order_relaxed);
    LOGI("Scheduler 已停止");
}

void Scheduler::join() {
    if (mCamera) mCamera->join();
    for (auto& w : mWorkers) { if (w) w->join(); }
    if (mGroupMgr) mGroupMgr->join();
}

void Scheduler::cleanup() {
    // 可选：清空队列残留
    g_queue_manager.clearRaw();
    g_queue_manager.clearResult();

    mWorkers.clear();
    mGroupMgr.reset();
    mCamera.reset();
}

} // namespace XL