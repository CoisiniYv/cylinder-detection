#include "Scheduler.hpp"

#include <filesystem>
#include <fstream>
#include <algorithm>
#include <json/json.h>
#include "Utils/Common.hpp"
#include "db_utils.hpp"


namespace XL {

	static Json::Value detection_to_json(const Detection& d) {
		Json::Value j;
		j["label_id"] = d.label_id;
		j["confidence"] = d.confidence;
		Json::Value box;
		box["x"] = d.box.x; box["y"] = d.box.y; box["w"] = d.box.w; box["h"] = d.box.h;
		j["bbox"] = box;
		return j;
	}

	static Json::Value single_result_to_json(const SingleImageResult& r) {
		Json::Value j;
		Json::Value meta;
		meta["device_id"] = r.meta.device_id;
		meta["group_id"] = static_cast<Json::UInt64>(r.meta.group_id);
		meta["index_in_group"] = r.meta.index_in_group;
		j["meta"] = meta;

		j["saved_path"] = r.saved_path;
		j["skeleton_path"] = r.skeleton_path;

		Json::Value dets(Json::arrayValue);
		for (const auto& d : r.detections) dets.append(detection_to_json(d));
		j["detections"] = dets;
		return j;
	}

	static void upsert_group(SQLiteHelper& db, const std::string& run_id, long long gid, const std::string& device_id, const std::string& status) {
		// 查询是否存在当前 run 的该组记录
		char buf[256];
		snprintf(buf, sizeof(buf), "SELECT 1 FROM inspection_groups WHERE run_id='%s' AND group_id=%llu;", run_id.c_str(), (unsigned long long)gid);
		auto rows = db.query(buf);
		if (!rows.empty()) {
			db.prepare("UPDATE inspection_groups SET device_id=?, status=? WHERE run_id=? AND group_id=?;");
			db.bind(1, device_id);
			db.bind(2, status);
			db.bind(3, run_id);
			db.bind(4, gid);
			db.step();
		}
		else {
			db.prepare("INSERT INTO inspection_groups(run_id,group_id,device_id,status) VALUES(?,?,?,?);");
			db.bind(1, run_id);
			db.bind(2, gid);
			db.bind(3, device_id);
			db.bind(4, status);
			db.step();
		}
	}

	static long long upsert_face(SQLiteHelper& db, const std::string& run_id, long long gid, int face_index, const SingleImageResult& r) {
		const std::string saved_path = r.saved_path;
    	const std::string skeleton_path = r.skeleton_path;
    	const std::string original_path = r.original_path;

		// 查找当前 run + group + face_index 的 face_id
		char qface[256];
		snprintf(qface, sizeof(qface), "SELECT face_id FROM inspection_faces WHERE run_id='%s' AND group_id=%llu AND face_index=%d;",
			run_id.c_str(), (unsigned long long)gid, face_index);
		auto frows = db.query(qface);
		long long face_id = 0;
		if (!frows.empty()) {
			face_id = std::stoll(frows[0][0]);
			db.prepare("UPDATE inspection_faces SET saved_path=?, skeleton_path=?,original_path=? WHERE face_id=?;");
			db.bind(1, saved_path);
			db.bind(2, skeleton_path);
			db.bind(3, original_path);
			db.bind(4, face_id);
			db.step();
		}
		else {
			db.prepare("INSERT INTO inspection_faces(run_id,group_id,face_index,saved_path,skeleton_path,original_path) VALUES(?,?,?,?,?,?);");
			db.bind(1, run_id);
			db.bind(2, gid);
			db.bind(3, face_index);
			db.bind(4, saved_path);
			db.bind(5, skeleton_path);
			db.bind(6, original_path);
			db.step();
			// 查回 face_id
			auto f2 = db.query(qface);
			if (!f2.empty()) face_id = std::stoll(f2[0][0]);
		}
		return face_id;
	}

	static void replace_defects(SQLiteHelper& db, long long face_id, const std::vector<Detection>& dets) {
		db.prepare("DELETE FROM defect_detections WHERE face_id=?;");
		db.bind(1, face_id);
		db.step();

		for (const auto& d : dets) {
			db.prepare("INSERT INTO defect_detections(face_id,label_id,confidence,bbox_x,bbox_y,bbox_w,bbox_h,length,area) VALUES(?,?,?,?,?,?,?,?,?);");
			db.bind(1, face_id);
			db.bind(2, d.label_id);
			db.bind(3, static_cast<double>(d.confidence));
			db.bind(4, static_cast<double>(d.box.x));
			db.bind(5, static_cast<double>(d.box.y));
			db.bind(6, static_cast<double>(d.box.w));
			db.bind(7, static_cast<double>(d.box.h));
			// 优先写入检测自带的真实长度/面积；如缺失则回退到 bbox 估计
			double length = d.length > 0.0 ? d.length : std::max(static_cast<double>(d.box.w), static_cast<double>(d.box.h));
			double area = d.area > 0.0 ? d.area : static_cast<double>(d.box.w) * static_cast<double>(d.box.h);
			db.bind(8, length);
			db.bind(9, area);
			db.step();
		}
	}

	static bool save_group_db(const QuadFrameResult& qres, const std::string& run_id, const std::string& device_id, const std::string& dbfile) {
		try {
			SQLiteHelper db(dbfile);

			// 计算组状态：任一面有检测则视为 NG
			bool ng = false;
			for (int i = 0; i < static_cast<int>(kQuadImageCount); ++i) {
				if (!qres.results[i].detections.empty()) { ng = true; break; }
			}
			const std::string status = ng ? "NG" : "GOOD";
			std::uint64_t gid_u64 = qres.source.group_id ? qres.source.group_id : qres.results[0].meta.group_id;
			long long gid = static_cast<long long>(gid_u64);

			db.begin();
			upsert_group(db, run_id, gid, device_id, status);

			// 写入四个面的记录与缺陷
			for (int i = 0; i < static_cast<int>(kQuadImageCount); ++i) {
				SingleImageResult res = qres.results[i];
				if(res.original_path.empty()) {
					const std::string savePath = (g_server_state.config ? g_server_state.config->outputdir : std::string()) + "\\" + g_server_state.run_id + "\\" + std::to_string(gid);
					std::string original_name = "original_g" + std::to_string(gid) + "_i" + std::to_string(i);
					std::string original_path = savePath + "\\" + original_name + ".png";
					res.original_path = original_path;
				}
				long long face_id = upsert_face(db, run_id, gid, i, res);
				if (face_id <= 0) {
					LOGE("写入面记录失败：run_id=%s, group=%llu, face_index=%d", run_id.c_str(), (unsigned long long)gid_u64, i);
					db.rollback();
					return false;
				}
				replace_defects(db, face_id, res.detections);
			}

			db.commit();
			LOGI("组结果已写入数据库: run_id=%s, group_id=%llu, device=%s, status=%s", run_id.c_str(), (unsigned long long)gid_u64, device_id.c_str(), status.c_str());
			return true;
		}
		catch (const std::exception& e) {
			LOGE("写入数据库异常: %s", e.what());
			return false;
		}
	}

	Scheduler::Scheduler() {}
	Scheduler::~Scheduler() { stop(); join(); }

	bool Scheduler::start(const DetectParams& params, int num_detect_threads) {
		if (mRunning.load(std::memory_order_relaxed)) {
			LOGE("Scheduler 已在运行，忽略重复启动");
			return false;
		}

		mParams = params;

		// 启动队列管理（唤醒阻塞）
		g_queue_manager.start();

		// 复用或创建共享相机线程
		if (!XL::g_camera_thread) {
			XL::g_camera_thread = std::make_shared<CameraThread>();
		}
		if (!XL::g_camera_thread->isRunning()) {
			extern ServerState g_server_state;
			const std::string slide_port = (g_server_state.config) ? g_server_state.config->slidePort : std::string();
			const int slide_axis = (g_server_state.config) ? g_server_state.config->slideAxisId : 0;
			const int slide_timeout_ms = (g_server_state.config) ? g_server_state.config->slideTimeoutMs : 20000;
			if (!XL::g_camera_thread->start(mParams.delay_ms, /*device_id*/mParams.device_id,
				slide_port, slide_axis, slide_timeout_ms, true)) {
				LOGE("CameraThread 启动失败，停止调度器");
				g_queue_manager.stop();
				return false;
			}
		}
		mCamera = XL::g_camera_thread;

		// 创建并启动检测线程
		if (num_detect_threads <= 0) num_detect_threads = 1;
		mWorkers.reserve(num_detect_threads);
		for (int i = 0; i < num_detect_threads; ++i) {
			auto worker = std::make_unique<DetectThread>(i, mParams);
			worker->start();
			mWorkers.emplace_back(std::move(worker));
		}

		// 数据库路径来自配置，缺省为 my_inspection.db（初始化已在 main 完成）
		const std::string dbfile = (g_server_state.config && !g_server_state.config->dbPath.empty())
			? g_server_state.config->dbPath
			: std::string("my_inspection.db");
		// 每次启动生成新的批次ID（run_id），并注册到数据库
		const std::string run_id = std::string("run_") + XL::getCurFormatTimeStr("%Y%m%d_%H%M%S");
		g_server_state.run_id = run_id;
		XL::register_run(dbfile, run_id);

		// 创建并启动组管理线程
		mGroupMgr = std::make_unique<GroupManager>();
		mGroupMgr->setCameraThread(mCamera.get());
		mGroupMgr->setGroupTimeoutMs(mGroupTimeoutMs);
		mGroupMgr->setOnGroupComplete([this, dbfile, run_id](const QuadFrameResult& qres) {
			// 仅进行写入，初始化已在 main() 完成
			if (!save_group_db(qres, run_id, mParams.device_id, dbfile)) {
				LOGE("保存组结果到数据库失败");
			}
			});
		mGroupMgr->start();

		mRunning.store(true, std::memory_order_relaxed);
		LOGI("Scheduler启动完成：camera=1, workers=%d, group_manager=1", num_detect_threads);
		return true;
	}

	bool Scheduler::startFromServerState(int num_detect_threads) {
		// 若 Server 提供了最近一次任务参数，则从中启动
		extern ServerState g_server_state;
		if (!g_server_state.has_task.load(std::memory_order_relaxed)) {
			LOGE("ServerState未检测到任务，无法启动 Scheduler");
			return false;
		}
		return start(g_server_state.last_params, num_detect_threads);
	}

	void Scheduler::stop() {
		if (!mRunning.load(std::memory_order_relaxed)) return;

		LOGI("Scheduler停止中...");

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
