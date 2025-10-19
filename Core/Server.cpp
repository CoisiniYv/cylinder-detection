/*
@server实现HTTP接口，用于控制检测流程和配置参数。外部可通过接口下发“开始检测”、“停止检测”命令，以及指定ROI遮罩或其他参数。
本模块仅负责：接收参数 + 校验 + 保存到共享状态（ServerState）。后续由 scheduler 读取并启动具体线程。
使用库：libevent、jsoncpp。
*/

#include "Server.hpp"
#ifdef WIN32
#pragma comment(lib, "ws2_32.lib")
#endif



#include "Utils/Log.hpp"
#include "Utils/Common.hpp"
#include "queue_manager.hpp"
#include "Scheduler.hpp"
#include <memory>
#include <cstring>

using namespace XL;

#define RECV_BUF_MAX_SIZE 1024*8

// 全局共享状态定义
ServerState g_server_state;
static std::unique_ptr<Scheduler> g_scheduler; // 调度器实例（在接口中启动/停止）

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
    try { return std::stoi(s); } catch (...) { return defv; }
}

static double parse_double(const std::string& s, double defv) {
    if (s.empty()) return defv;
    try { return std::stod(s); } catch (...) { return defv; }
}

static float parse_float(const std::string& s, float defv) {
    if (s.empty()) return defv;
    try { return std::stof(s); } catch (...) { return defv; }
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

    // 基本
    p.trt_engine_file = get("trt_engine_file");
    if (p.trt_engine_file.empty() && state && state->config) {
        // 默认使用配置的 modelDir（如果前端未提供完整路径，后续 scheduler 可拼接）
        p.trt_engine_file = state->config->modelDir;
    }
    p.enable_swap_rb = parse_bool(get("enable_swap_rb"), true);
    p.input_image_path = get("input_image_path");

    // ROI & line
    p.crop_x = parse_int(get("crop_x"), 0);
    p.crop_y = parse_int(get("crop_y"), 0);
    p.crop_width = parse_int(get("crop_width"), 0);
    p.crop_height = parse_int(get("crop_height"), 0);
    p.line_x1 = parse_int(get("line_x1"), 0);
    p.line_y1 = parse_int(get("line_y1"), 0);
    p.line_x2 = parse_int(get("line_x2"), 0);
    p.line_y2 = parse_int(get("line_y2"), 0);
    p.stripe_radius = parse_int(get("stripe_radius"), 0);

    // 条纹去除/去噪
    p.filter_width = parse_int(get("filter_width"), 10);
    p.attenuation_factor = parse_double(get("attenuation_factor"), 0.0001);
    p.target_angle = parse_int(get("target_angle"), 90);
    p.angle_tolerance = parse_int(get("angle_tolerance"), 10);
    p.enable_denoising = parse_bool(get("enable_denoising"), true);
    p.denoise_h = parse_float(get("denoise_h"), 5.0f);
    p.denoise_hColor = parse_float(get("denoise_hColor"), 8.0f);
    p.denoise_search_window = parse_int(get("denoise_search_window"), 17);
    p.denoise_template_window = parse_int(get("denoise_template_window"), 11);

    // 切片与阈值
    p.save_slices = parse_bool(get("save_slices"), false);
    p.slices_save_dir = get("slices_save_dir");
    if (p.slices_save_dir.empty()) p.slices_save_dir = "./slices";
    p.slice_width = parse_int(get("slice_width"), 640);
    p.slice_height = parse_int(get("slice_height"), 640);
    p.slice_distance = parse_int(get("slice_distance"), 40);
    p.nms_threshold = parse_float(get("nms_threshold"), 0.45f);
    p.conf_threshold = parse_float(get("conf_threshold"), 0.20f);

    // 输出与标签
    p.result_image_path = get("result_image_path");
    if (p.result_image_path.empty()) p.result_image_path = "./output/detection_result.png";
    {
        std::string labels = get("labels");
        if (!labels.empty()) {
            auto parts = XL::split(labels, ",");
            p.labels = parts;
        }
    }

    // 相机延迟与设备ID
    p.delay_ms = parse_int(get("delay_ms"), 0);
    p.device_id = get("device_id");

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

    p.trt_engine_file = getS("trt_engine_file", state && state->config ? state->config->modelDir : "");
    p.enable_swap_rb = getB("enable_swap_rb", true);
    p.input_image_path = getS("input_image_path", "");

    p.crop_x = getI("crop_x", 0);
    p.crop_y = getI("crop_y", 0);
    p.crop_width = getI("crop_width", 0);
    p.crop_height = getI("crop_height", 0);
    p.line_x1 = getI("line_x1", 0);
    p.line_y1 = getI("line_y1", 0);
    p.line_x2 = getI("line_x2", 0);
    p.line_y2 = getI("line_y2", 0);
    p.stripe_radius = getI("stripe_radius", 0);

    p.filter_width = getI("filter_width", 10);
    p.attenuation_factor = getD("attenuation_factor", 0.0001);
    p.target_angle = getI("target_angle", 90);
    p.angle_tolerance = getI("angle_tolerance", 10);
    p.enable_denoising = getB("enable_denoising", true);
    p.denoise_h = getF("denoise_h", 5.0f);
    p.denoise_hColor = getF("denoise_hColor", 8.0f);
    p.denoise_search_window = getI("denoise_search_window", 17);
    p.denoise_template_window = getI("denoise_template_window", 11);

    p.save_slices = getB("save_slices", false);
    p.slices_save_dir = getS("slices_save_dir", "./slices");
    p.slice_width = getI("slice_width", 640);
    p.slice_height = getI("slice_height", 640);
    p.slice_distance = getI("slice_distance", 40);
    p.nms_threshold = getF("nms_threshold", 0.45f);
    p.conf_threshold = getF("conf_threshold", 0.20f);
    p.result_image_path = getS("result_image_path", "./output/detection_result.png");

    if (root.isMember("labels") && root["labels"].isArray()) {
        for (const auto& v : root["labels"]) {
            p.labels.emplace_back(v.asString());
        }
    }

    p.delay_ms = getI("delay_ms", 0);
    p.device_id = getS("device_id", "");

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
        send_json(req, -1, "bad request");
        return;
    }

    DetectParams p;
    // 优先 POST JSON，其次 GET 查询
    if (evbuffer_get_length(req->input_buffer) > 0) {
        p = parse_params_from_post(req, state);
    } else {
        p = parse_params_from_get(req, state);
    }

    // 基本校验：ROI 与阈值
    if (p.crop_width <= 0 || p.crop_height <= 0) {
        send_json(req, -2, "crop_width/crop_height 必须为正数");
        return;
    }
    if (p.nms_threshold <= 0 || p.conf_threshold <= 0) {
        // 放宽为 >0，如果需要更严格可调整到 (0,1]
        send_json(req, -3, "nms_threshold/conf_threshold 必须大于 0");
        return;
    }

    // 保存到共享状态供后续 scheduler 使用
    state->last_params = p;
    state->has_task.store(true, std::memory_order_relaxed);
    LOGI("接收到检测任务：engine=%s, delay_ms=%d, device_id=%s", p.trt_engine_file.c_str(), p.delay_ms, p.device_id.c_str());

    // 启动/重启调度器
    if (!g_scheduler) {
        g_scheduler = std::make_unique<Scheduler>();
    } else if (g_scheduler->isRunning()) {
        g_scheduler->stop();
    }
    bool started = g_scheduler->startFromServerState(4);

    // 返回回显
    Json::Value data;
    data["trt_engine_file"] = p.trt_engine_file;
    data["enable_swap_rb"] = p.enable_swap_rb;
    data["input_image_path"] = p.input_image_path;
    data["crop_x"] = p.crop_x;
    data["crop_y"] = p.crop_y;
    data["crop_width"] = p.crop_width;
    data["crop_height"] = p.crop_height;
    data["line_x1"] = p.line_x1;
    data["line_y1"] = p.line_y1;
    data["line_x2"] = p.line_x2;
    data["line_y2"] = p.line_y2;
    data["stripe_radius"] = p.stripe_radius;
    data["filter_width"] = p.filter_width;
    data["attenuation_factor"] = p.attenuation_factor;
    data["target_angle"] = p.target_angle;
    data["angle_tolerance"] = p.angle_tolerance;
    data["enable_denoising"] = p.enable_denoising;
    data["denoise_h"] = p.denoise_h;
    data["denoise_hColor"] = p.denoise_hColor;
    data["denoise_search_window"] = p.denoise_search_window;
    data["denoise_template_window"] = p.denoise_template_window;
    data["save_slices"] = p.save_slices;
    data["slices_save_dir"] = p.slices_save_dir;
    data["slice_width"] = p.slice_width;
    data["slice_height"] = p.slice_height;
    data["slice_distance"] = p.slice_distance;
    data["nms_threshold"] = p.nms_threshold;
    data["conf_threshold"] = p.conf_threshold;
    data["result_image_path"] = p.result_image_path;
    data["delay_ms"] = p.delay_ms;
    data["device_id"] = p.device_id;
    {
        Json::Value arr(Json::arrayValue);
        for (const auto& s : p.labels) arr.append(s);
        data["labels"] = arr;
    }
    data["scheduler_started"] = started;

    send_json(req, 0, "任务已接收", data);
}

void api_control_cancel(struct evhttp_request* req, void* arg) {
    ServerState* state = static_cast<ServerState*>(arg);
    if (!state) {
        send_json(req, -1, "bad request");
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
    evhttp_set_gencb(http, api_not_found, nullptr);

    LOGI("HTTP 服务器已启动: http://%s:%d", cfg->ip.c_str(), cfg->analyzerPort);

    // 运行事件循环（阻塞）
    event_base_dispatch(base);

    // 清理资源
    evhttp_free(http);
    event_base_free(base);
    WSACleanup();
}