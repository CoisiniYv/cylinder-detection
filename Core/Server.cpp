#include "Server.hpp"
#ifdef WIN32
#pragma comment(lib, "ws2_32.lib")
#endif

#include "Utils/Log.hpp"
#include "Utils/Common.hpp"
#include "queue_manager.hpp"
#include "Scheduler.hpp"
#include "single_detect_thread.hpp"
#include "db_utils.hpp"
#include "camera_thread.hpp"

#include <cstring>
#include <cctype>
#include <filesystem>
#include <memory>
#include <mutex>
#include <future>


using namespace XL;

#define RECV_BUF_MAX_SIZE 1024*8

// 全局共享状态定义
ServerState g_server_state;
static std::unique_ptr<Scheduler> g_scheduler; // 调度器实例（在接口中启动/停止）
static std::unique_ptr<SingleDetectThread> g_single_thread; // 单图检测线程实例
std::shared_ptr<CameraThread> XL::g_camera_thread; // 共享相机线程实例
static std::mutex g_camera_mtx; // 相机实例创建/释放保护

// ========== 工具函数：发送 JSON 响应 ==========
static void send_json(struct evhttp_request* req, int code, const std::string& msg, const Json::Value& data = Json::Value()) {
	struct evbuffer* evb = evbuffer_new();
	if (!evb) return;
	Json::Value root;
	root["code"] = code;
	root["msg"] = msg;
	if (!data.isNull()) root["data"] = data;

	Json::StreamWriterBuilder wbuilder;
	wbuilder["indentation"] = ""; // 紧凑输出
	std::string payload = Json::writeString(wbuilder, root);

	evbuffer_add(evb, payload.data(), payload.size());
	struct evkeyvalq* headers = evhttp_request_get_output_headers(req);
	evhttp_add_header(headers, "Content-Type", "application/json");
	evhttp_send_reply(req, 200, "OK", evb);
	evbuffer_free(evb);
}

// ========== JSON 序列化辅助 ==========
static Json::Value to_json_frame_meta(const FrameMeta& m) {
	Json::Value j;
	j["device_id"] = m.device_id;
	j["group_id"] = static_cast<Json::UInt64>(m.group_id);
	j["index_in_group"] = m.index_in_group;
	return j;
}

static Json::Value to_json_detection(const Detection& d) {
	Json::Value j;
	j["label_id"] = d.label_id;
	j["confidence"] = d.confidence;
	Json::Value box;
	box["x"] = d.box.x; box["y"] = d.box.y; box["w"] = d.box.w; box["h"] = d.box.h;
	j["box"] = box;
	j["length"] = d.length;
	j["area"] = d.area;
	return j;
}

static Json::Value to_json_single_result(const SingleImageResult& r) {
	Json::Value j;
	j["meta"] = to_json_frame_meta(r.meta);
	Json::Value arr(Json::arrayValue);
	for (const auto& d : r.detections) arr.append(to_json_detection(d));
	j["detections"] = arr;
	j["saved_path"] = r.saved_path;
	j["skeleton_path"] = r.skeleton_path;
	return j;
}

// ========== GET/POST 解析函数==========
void parse_get(struct evhttp_request* req, struct evkeyvalq* params) {
	if (req == nullptr) {
		return;
	}
	const char* url = evhttp_request_get_uri(req);
	if (url == nullptr) {
		return;
	}
	struct evhttp_uri* decoded = evhttp_uri_parse(url);
	if (!decoded) {
		return;
	}
	const char* path = evhttp_uri_get_path(decoded);
	if (path == nullptr) {
		path = "/";
	}
	char* query = (char*)evhttp_uri_get_query(decoded);
	if (query == nullptr) {
		return;
	}
	evhttp_parse_query_str(query, params);
}
void parse_post(struct evhttp_request* req, char* buf) {
	size_t post_size = 0;

	post_size = evbuffer_get_length(req->input_buffer);
	if (post_size <= 0) {
		return;
	}
	else {
		size_t copy_len = post_size > RECV_BUF_MAX_SIZE ? RECV_BUF_MAX_SIZE : post_size;
		memcpy(buf, evbuffer_pullup(req->input_buffer, -1), copy_len);
		buf[copy_len] = '\0';
	}
}

// ========== 参数解析辅助 ==========
static bool parse_bool(const std::string& s, bool defv) {
	if (s.empty()) return defv;
	if (s == "1" || s == "true" || s == "True" || s == "TRUE") return true;
	if (s == "0" || s == "false" || s == "False" || s == "FALSE") return false;
	return defv;
}

static int parse_int(const std::string& s, int defv) {
	if (s.empty()) return defv;
	try { return std::stoi(s); }
	catch (...) { return defv; }
}

static double parse_double(const std::string& s, double defv) {
	if (s.empty()) return defv;
	try { return std::stod(s); }
	catch (...) { return defv; }
}

static float parse_float(const std::string& s, float defv) {
	if (s.empty()) return defv;
	try { return std::stof(s); }
	catch (...) { return defv; }
}

// 组装路径（优先绝对路径；否则与配置中的 modelDir 拼接）
static bool is_absolute_path(const std::string& p) {
	if (p.size() >= 2 && std::isalpha(static_cast<unsigned char>(p[0])) && p[1] == ':') return true; // C:\...
	if (!p.empty() && (p[0] == '\\' || p[0] == '/')) return true; // \\server 或 /root
	return false;
}
static std::string join_path(const std::string& base, const std::string& name) {
	if (name.empty()) return std::string();
	if (is_absolute_path(name)) return name;
	if (base.empty()) return name;
	const char sep = '\\';
	if (!base.empty() && (base.back() == '\\' || base.back() == '/')) return base + name;
	return base + sep + name;
}

// 从 GET 查询参数解析 DetectParams
static DetectParams parse_params_from_get(struct evhttp_request* req, ServerState* state) {
	DetectParams p;
	struct evkeyvalq params;
	evhttp_parse_query_str("", &params); // init
	parse_get(req, &params);

	auto get = [&](const char* key) -> std::string {
		const char* v = evhttp_find_header(&params, key);
		return v ? std::string(v) : std::string();
		};

	// 基本与模型
	const std::string modelDir = (state && state->config) ? state->config->modelDir : std::string();
	p.model_name = get("model_name");
	// 解析检测模型路径
	if (!p.model_name.empty()) {
		p.engine_file = join_path(modelDir, p.model_name);
	}
	else {
		p.engine_file = modelDir;
	}

	// SAM模型完整路径
	p.sam_encoder_engine_file = join_path(modelDir, p.sam_encoder_name);
	p.sam_decoder_engine_file = join_path(modelDir, p.sam_decoder_name);
	p.sam_encoder_onnx_file = join_path(modelDir, p.sam_encoder_onnx_name);
	p.sam_decoder_onnx_file = join_path(modelDir, p.sam_decoder_onnx_name);

	p.input_image_path = get("input_image_path");

	// ROI & Circle
	p.enable_four_side_crop = parse_bool(get("enable_four_side_crop"), false);
	p.crop_x = parse_int(get("crop_x"), 0);
	p.crop_y = parse_int(get("crop_y"), 0);
	p.crop_width = parse_int(get("crop_width"), 0);
	p.crop_height = parse_int(get("crop_height"), 0);
	p.is_qw = parse_bool(get("is_qw"), false);
	p.circle_x1 = parse_int(get("circle_x1"), 0);
	p.circle_y1 = parse_int(get("circle_y1"), 0);
	p.circle_x2 = parse_int(get("circle_x2"), 0);
	p.circle_y2 = parse_int(get("circle_y2"), 0);
	p.radius = parse_int(get("radius"), 0);

	// 条纹去除/去噪
	p.enable_fourier_transform = parse_bool(get("enable_fourier_transform"), false);
	p.filter_width = parse_int(get("filter_width"), 10);
	p.attenuation_factor = parse_double(get("attenuation_factor"), 0.0001);
	p.target_angle = parse_int(get("target_angle"), 90);
	p.angle_tolerance = parse_int(get("angle_tolerance"), 10);
	p.enable_denoising = parse_bool(get("enable_denoising"), false);
	p.denoise_h = parse_float(get("denoise_h"), 5.0f);
	p.denoise_hColor = parse_float(get("denoise_hColor"), 8.0f);
	p.denoise_search_window = parse_int(get("denoise_search_window"), 17);
	p.denoise_template_window = parse_int(get("denoise_template_window"), 11);

	// 切片与阈值
	p.slice_width = parse_int(get("slice_width"), 640);
	p.slice_height = parse_int(get("slice_height"), 640);
	p.slice_distance = parse_int(get("slice_distance"), 40);
	p.nms_threshold = parse_float(get("nms_threshold"), 0.45f);
	p.conf_threshold = parse_float(get("conf_threshold"), 0.20f);

	// SAM过滤参数（mm、mm^2）
	p.pix_to_mm = parse_double(get("pix_to_mm"), 0.021);
	p.min_area_mm2 = parse_double(get("min_area_mm2"), 0.0);
	p.min_diameter_mm = parse_double(get("min_diameter_mm"), 0.0);

	// 相机延迟与设备ID
	p.delay_ms = parse_int(get("delay_ms"), 0);
	p.qw_index = parse_int(get("qw_index"), 0);

	return p;
}

// 从 POST JSON 解析 DetectParams
static DetectParams parse_params_from_post(struct evhttp_request* req, ServerState* state) {
	DetectParams p;
	char buf[RECV_BUF_MAX_SIZE + 1] = { 0 };
	parse_post(req, buf);
	if (buf[0] == '\0') return p;

	Json::CharReaderBuilder builder;
	JSONCPP_STRING errs;
	Json::Value root;
	std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
	bool ok = reader->parse(buf, buf + strlen(buf), &root, &errs);
	if (!ok) {
		LOGE("POST JSON parse error: %s", errs.c_str());
		return p;
	}

	auto getS = [&](const char* k, const std::string& def) {
		return root.isMember(k) ? root[k].asString() : def;
		};
	auto getB = [&](const char* k, bool def) {
		return root.isMember(k) ? root[k].asBool() : def;
		};
	auto getI = [&](const char* k, int def) {
		return root.isMember(k) ? root[k].asInt() : def;
		};
	auto getF = [&](const char* k, float def) {
		return root.isMember(k) ? root[k].asFloat() : def;
		};
	auto getD = [&](const char* k, double def) {
		return root.isMember(k) ? root[k].asDouble() : def;
		};

	p.input_image_path = getS("input_image_path", "");

	const std::string modelDir = (state && state->config) ? state->config->modelDir : std::string();
	p.model_name = getS("model_name", "");

	if (!p.model_name.empty()) {
		p.engine_file = join_path(modelDir, p.model_name);
	}
	else {
		p.engine_file = modelDir;
	}

	p.sam_encoder_engine_file = join_path(modelDir, p.sam_encoder_name);
	p.sam_decoder_engine_file = join_path(modelDir, p.sam_decoder_name);
	p.sam_encoder_onnx_file = join_path(modelDir, p.sam_encoder_onnx_name);
	p.sam_decoder_onnx_file = join_path(modelDir, p.sam_decoder_onnx_name);

	p.enable_four_side_crop = getB("enable_four_side_crop", false);
	p.crop_x = getI("crop_x", 0);
	p.crop_y = getI("crop_y", 0);
	p.crop_width = getI("crop_width", 0);
	p.crop_height = getI("crop_height", 0);
	p.is_qw = getB("is_qw", false);
	p.circle_x1 = getI("circle_x1", 0);
	p.circle_y1 = getI("circle_y1", 0);
	p.circle_x2 = getI("circle_x2", 0);
	p.circle_y2 = getI("circle_y2", 0);
	p.radius = getI("radius", 0);

	p.enable_fourier_transform = getB("enable_fourier_transform", false);
	p.filter_width = getI("filter_width", 10);
	p.attenuation_factor = getD("attenuation_factor", 0.0001);
	p.target_angle = getI("target_angle", 90);
	p.angle_tolerance = getI("angle_tolerance", 10);
	p.enable_denoising = getB("enable_denoising", false);
	p.denoise_h = getF("denoise_h", 5.0f);
	p.denoise_hColor = getF("denoise_hColor", 8.0f);
	p.denoise_search_window = getI("denoise_search_window", 17);
	p.denoise_template_window = getI("denoise_template_window", 11);

	p.slice_width = getI("slice_width", 640);
	p.slice_height = getI("slice_height", 640);
	p.slice_distance = getI("slice_distance", 40);
	p.nms_threshold = getF("nms_threshold", 0.45f);
	p.conf_threshold = getF("conf_threshold", 0.20f);

	// SAM 过滤参数（mm、mm^2）
	p.pix_to_mm = getD("pix_to_mm", 0.021);
	p.min_area_mm2 = getD("min_area_mm2", 0.0);
	p.min_diameter_mm = getD("min_diameter_mm", 0.0);

	p.delay_ms = getI("delay_ms", 0);
	p.qw_index = getI("qw_index", 0);

	return p;
}

// ========== API 处理 ==========
void api_health(struct evhttp_request* req, void* arg) {
	ServerState* state = static_cast<ServerState*>(arg);
	Json::Value data;
	data["status"] = "ok";
	data["raw_queue_size"] = static_cast<Json::UInt64>(g_queue_manager.rawSizeApprox());
	data["result_queue_size"] = static_cast<Json::UInt64>(g_queue_manager.resultSizeApprox());
	if (state && state->config) {
		data["host"] = state->config->ip;
		data["port"] = state->config->analyzerPort;
		data["modelDir"] = state->config->modelDir;
		data["uploadDir"] = state->config->outputdir;
	}
	send_json(req, 0, "OK", data);
}

void api_control_add(struct evhttp_request* req, void* arg) {
	ServerState* state = static_cast<ServerState*>(arg);
	if (req == nullptr || state == nullptr) {
		send_json(req, -1, "请求失败");
		return;
	}

	DetectParams p;
	// 优先 POST JSON，其次 GET 查询
	if (evbuffer_get_length(req->input_buffer) > 0) {
		p = parse_params_from_post(req, state);
	}
	else {
		p = parse_params_from_get(req, state);
	}

	if (p.nms_threshold <= 0 || p.conf_threshold <= 0) {
		// 放宽为 >0，如果需要更严格可调整到 (0,1]
		send_json(req, -3, "nms_threshold/conf_threshold 必须大于 0");
		return;
	}

	// 保存到共享状态供后续scheduler使用
	state->last_params = p;
	state->has_task.store(true, std::memory_order_relaxed);
	LOGI("接收到相机检测任务：model=%s, sam_encoder=%s, sam_decoder=%s, delay_ms=%d, qw_index=%d, device_id=%s",
		p.engine_file.c_str(), p.sam_encoder_engine_file.c_str(), p.sam_decoder_engine_file.c_str(), p.delay_ms, p.qw_index, p.device_id.c_str());

	// 启动/重启调度器
	if (!g_scheduler) {
		g_scheduler = std::make_unique<Scheduler>();
	}
	else if (g_scheduler->isRunning()) {
		g_scheduler->stop();
	}
	bool started = g_scheduler->startFromServerState(4);

	send_json(req, 0, "相机检测任务已接收");
}

void api_control_cancel(struct evhttp_request* req, void* arg) {
	ServerState* state = static_cast<ServerState*>(arg);
	if (!state) {
		send_json(req, -1, "请求失败");
		return;
	}
	state->has_task.store(false, std::memory_order_relaxed);

	// 停止调度器
	if (g_scheduler) {
		g_scheduler->stop();
	}

	LOGI("收到取消任务指令，调度器已停止");
	send_json(req, 0, "任务已取消");
}

// ========== 单图检测 API ==========
void api_single_detect_add(struct evhttp_request* req, void* arg) {
	ServerState* state = static_cast<ServerState*>(arg);
	if (req == nullptr || state == nullptr) {
		send_json(req, -1, "请求失败");
		return;
	}

	DetectParams p;
	if (evbuffer_get_length(req->input_buffer) > 0) {
		p = parse_params_from_post(req, state);
	}
	else {
		p = parse_params_from_get(req, state);
	}

	// 基本校验
	if (p.input_image_path.empty()) {
		send_json(req, -10, "input_image_path 不能为空");
		return;
	}
	std::error_code fec;
	if (!std::filesystem::exists(p.input_image_path, fec)) {
		send_json(req, -11, "input_image_path 文件不存在");
		return;
	}
	if (p.nms_threshold <= 0 || p.conf_threshold <= 0) {
		send_json(req, -3, "nms_threshold/conf_threshold 必须大于 0");
		return;
	}

	// 启动/复用单图检测线程
	if (!g_single_thread) {
		g_single_thread = std::make_unique<SingleDetectThread>();
	}
	if (!g_single_thread->isRunning()) {
		g_single_thread->start();
	}

	// 提交并阻塞等待结果
	SingleImageResult res;
	std::string err;
	bool ok = g_single_thread->submitAndWait(p, res, err);
	if (!ok) {
		send_json(req, -12, std::string("单图检测失败: ") + err);
		return;
	}

	send_json(req, 0, "OK", to_json_single_result(res));
}

void api_single_detect_cancel(struct evhttp_request* req, void* /*arg*/) {
	if (g_single_thread) {
		g_single_thread->stop();
		g_single_thread->join();
		g_single_thread.reset();
		LOGI("单图检测线程已释放");
		send_json(req, 0, "单图检测线程已释放");
	}
	else {
		LOGI("单图检测线程不存在");
		send_json(req, 0, "单图检测线程不存在");
	}
}

// ========== 相机单图采集 API ==========
void api_camera_single_capture(struct evhttp_request* req, void* arg) {
	ServerState* state = static_cast<ServerState*>(arg);
	if (req == nullptr || state == nullptr) {
		send_json(req, -1, "请求失败");
		return;
	}

	// 复用或创建共享相机线程
	bool was_running = false;
	{
		std::lock_guard<std::mutex> lk(g_camera_mtx);
		if (!XL::g_camera_thread) {
			XL::g_camera_thread = std::make_shared<CameraThread>();
		}
		was_running = XL::g_camera_thread->isRunning();
		if (!was_running) {
			const int delay_ms = state->last_params.delay_ms;
			const std::string device_id = state->last_params.device_id;
			const std::string slide_port = (state->config) ? state->config->slidePort : std::string();
			const int slide_axis = (state->config) ? state->config->slideAxisId : 0;
			const int slide_timeout_ms = (state->config) ? state->config->slideTimeoutMs : 20000;
			if (!XL::g_camera_thread->start(delay_ms, device_id,
				slide_port, slide_axis, slide_timeout_ms, false)) {
				send_json(req, -20, "相机启动失败或不可用");
				return;
			}
		}
	}

	// 发起单张采集
	ImageFrame frame;
	try {
		auto fut = XL::g_camera_thread->captureSingleImage(10000);
		frame = fut.get();
	}
	catch (const std::exception& e) {
		send_json(req, -21, std::string("单张采集失败: ") + e.what());
		return;
	}

	// 保存到 single_detect_thread 中使用的目录
	std::string baseDir = (state->config ? state->config->outputdir : std::string());
	std::string saveDir = baseDir + "\\" + state->run_id + "\\single";
	std::error_code ec;
	std::filesystem::create_directories(saveDir, ec);
	if (ec) {
		send_json(req, -22, "创建保存目录失败");
		return;
	}
	std::string savePath = saveDir + "\\camera_image.png";

	// 保存 PNG（无损）
	std::vector<int> params;
	params.push_back(cv::IMWRITE_PNG_COMPRESSION);
	params.push_back(0);
	bool ok = cv::imwrite(savePath, frame.ps_image, params);
	if (!ok) {
		send_json(req, -23, "相机采集图片保存失败");
		return;
	}

	Json::Value data;
	data["saved_path"] = savePath;
	data["camera_running"] = XL::g_camera_thread && XL::g_camera_thread->isRunning();
	send_json(req, 0, "OK", data);

	// 若本次为临时启动且未运行调度器，则在采集后释放相机线程，避免组采集占用资源
	if (!was_running && (!g_scheduler || !g_scheduler->isRunning())) {
		std::lock_guard<std::mutex> lk(g_camera_mtx);
		if (XL::g_camera_thread) {
			XL::g_camera_thread->stop();
			XL::g_camera_thread->join();
		}
	}
}

// 相机状态查询
void api_camera_status(struct evhttp_request* req, void* arg) {
	ServerState* state = static_cast<ServerState*>(arg);
	(void)state;
	Json::Value data;
	bool running = XL::g_camera_thread && XL::g_camera_thread->isRunning();
	data["camera_running"] = running;
	send_json(req, 0, "OK", data);
}

// 相机释放
void api_camera_stop(struct evhttp_request* req, void* arg) {
	ServerState* state = static_cast<ServerState*>(arg);
	(void)state;
	{
		std::lock_guard<std::mutex> lk(g_camera_mtx);
		if (XL::g_camera_thread) {
			XL::g_camera_thread->stop();
			XL::g_camera_thread->join();
			XL::g_camera_thread.reset();
			send_json(req, 0, "相机线程已释放");
			return;
		}
	}
	send_json(req, -1, "相机线程不存在");
}

// 未匹配到的路径返回 404
static void api_not_found(struct evhttp_request* req, void* /*arg*/) {
	struct evbuffer* evb = evbuffer_new();
	evbuffer_add_printf(evb, "404 Not Found");
	evhttp_send_reply(req, 404, "Not Found", evb);
	evbuffer_free(evb);
}

// ========== Server 类实现 ==========
Server::Server() {}
Server::~Server() {}

void Server::start(void* arg) {
	Config* cfg = static_cast<Config*>(arg);
	if (!cfg || !cfg->mState) {
		LOGE("配置无效，无法启动服务器");
		return;
	}

	// Windows 套接字初始化
	WSADATA wsaData;
	int wsaerr = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (wsaerr != 0) {
		LOGE("WSAStartup 失败: %d", wsaerr);
		return;
	}

	// 创建共享状态
	// 使用全局共享状态（便于后续 scheduler 读取）
	auto* state = &g_server_state;
	state->config = cfg;

	// 创建 event base 与 http server
	event_base* base = event_base_new();
	if (!base) {
		LOGE("event_base_new 失败");
		WSACleanup();
		return;
	}
	evhttp* http = evhttp_new(base);
	if (!http) {
		LOGE("evhttp_new 失败");
		event_base_free(base);
		WSACleanup();
		return;
	}

	// 绑定地址与端口
	if (evhttp_bind_socket(http, cfg->ip.c_str(), cfg->analyzerPort) != 0) {
		LOGE("绑定 %s:%d 失败", cfg->ip.c_str(), cfg->analyzerPort);
		evhttp_free(http);
		event_base_free(base);
		WSACleanup();
		return;
	}

	// 设置回调
	evhttp_set_cb(http, "/api/health", api_health, state);
	
	evhttp_set_cb(http, "/api/control/add", api_control_add, state);
	evhttp_set_cb(http, "/api/control/cancel", api_control_cancel, state);
	evhttp_set_cb(http, "/api/single_detect/add", api_single_detect_add, state);
	evhttp_set_cb(http, "/api/single_detect/cancel", api_single_detect_cancel, state);
	evhttp_set_cb(http, "/api/camera/single_capture", api_camera_single_capture, state);
	evhttp_set_cb(http, "/api/camera/status", api_camera_status, state);
	evhttp_set_cb(http, "/api/camera/stop", api_camera_stop, state);
	evhttp_set_gencb(http, api_not_found, nullptr);

	LOGI("HTTP 服务器已启动: http://%s:%d", cfg->ip.c_str(), cfg->analyzerPort);

	// 运行事件循环（阻塞）
	event_base_dispatch(base);

	// 清理资源
	evhttp_free(http);
	event_base_free(base);
	WSACleanup();
}
