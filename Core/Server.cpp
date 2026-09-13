#include "Server.hpp"

#include "Config.hpp"
#include "Scheduler.hpp"
#include "Utils/Log.hpp"
#include "camera_thread.hpp"
#include "detection_context.hpp"
#include "image_types.hpp"
#include "queue_manager.hpp"
#include "runtime_state.hpp"
#include "single_detect_thread.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <WinSock2.h>
#include <WS2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#endif

#include <event2/buffer.h>
#include <event2/event.h>
#include <event2/http.h>
#include <event2/keyvalq_struct.h>
#include <json/json.h>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace XL {

RuntimeState g_runtime_state;
std::shared_ptr<CameraThread> g_camera_thread;

namespace {

constexpr std::size_t kMaxRequestBodyBytes = 64 * 1024;
constexpr int kSingleCaptureTimeoutMs = 10000;

std::unique_ptr<Scheduler> g_scheduler;
std::unique_ptr<SingleDetectThread> g_single_thread;
std::mutex g_camera_mutex;

class RequestParams {
public:
    RequestParams() {
        evhttp_parse_query_str("", &mQuery);
    }

    ~RequestParams() {
        evhttp_clear_headers(&mQuery);
    }

    RequestParams(const RequestParams&) = delete;
    RequestParams& operator=(const RequestParams&) = delete;

    bool load(evhttp_request* req, std::string& error) {
        if (!req) {
            error = "request is null";
            return false;
        }

        evbuffer* input = evhttp_request_get_input_buffer(req);
        const std::size_t body_size = input ? evbuffer_get_length(input) : 0;
        if (body_size > 0) {
            if (body_size > kMaxRequestBodyBytes) {
                error = "request body exceeds 64 KiB";
                return false;
            }

            const unsigned char* body = evbuffer_pullup(input, -1);
            if (!body) {
                error = "unable to read request body";
                return false;
            }

            Json::CharReaderBuilder builder;
            builder["collectComments"] = false;
            std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
            JSONCPP_STRING parse_error;
            if (!reader->parse(
                    reinterpret_cast<const char*>(body),
                    reinterpret_cast<const char*>(body) + body_size,
                    &mJson,
                    &parse_error)) {
                error = "invalid JSON: " + parse_error;
                return false;
            }
            mUseJson = true;
            return true;
        }

        const char* uri_text = evhttp_request_get_uri(req);
        if (!uri_text) return true;

        evhttp_uri* uri = evhttp_uri_parse(uri_text);
        if (!uri) {
            error = "invalid request URI";
            return false;
        }

        const char* query = evhttp_uri_get_query(uri);
        if (query && evhttp_parse_query_str(query, &mQuery) != 0) {
            evhttp_uri_free(uri);
            error = "invalid query string";
            return false;
        }
        evhttp_uri_free(uri);
        return true;
    }

    std::string stringValue(const char* key, const std::string& fallback = {}) const {
        if (mUseJson) {
            const auto& value = mJson[key];
            if (value.isNull()) return fallback;
            return value.asString();
        }
        const char* value = evhttp_find_header(&mQuery, key);
        return value ? std::string(value) : fallback;
    }

    int intValue(const char* key, int fallback) const {
        if (mUseJson) {
            const auto& value = mJson[key];
            if (value.isNull()) return fallback;
            if (value.isInt() || value.isUInt()) return value.asInt();
            return parseInt(value.asString(), fallback);
        }
        return parseInt(stringValue(key), fallback);
    }

    float floatValue(const char* key, float fallback) const {
        if (mUseJson) {
            const auto& value = mJson[key];
            if (value.isNull()) return fallback;
            if (value.isNumeric()) return value.asFloat();
            return parseFloat(value.asString(), fallback);
        }
        return parseFloat(stringValue(key), fallback);
    }

    double doubleValue(const char* key, double fallback) const {
        if (mUseJson) {
            const auto& value = mJson[key];
            if (value.isNull()) return fallback;
            if (value.isNumeric()) return value.asDouble();
            return parseDouble(value.asString(), fallback);
        }
        return parseDouble(stringValue(key), fallback);
    }

    bool boolValue(const char* key, bool fallback) const {
        if (mUseJson) {
            const auto& value = mJson[key];
            if (value.isNull()) return fallback;
            if (value.isBool()) return value.asBool();
            return parseBool(value.asString(), fallback);
        }
        return parseBool(stringValue(key), fallback);
    }

    std::vector<int> intListValue(const char* key, const std::vector<int>& fallback) const {
        if (mUseJson) {
            const auto& value = mJson[key];
            if (value.isNull()) return fallback;
            if (!value.isArray()) return fallback;
            std::vector<int> result;
            result.reserve(value.size());
            for (const auto& item : value) {
                if (item.isInt() || item.isUInt()) result.push_back(item.asInt());
            }
            return result.empty() ? fallback : result;
        }

        const std::string text = stringValue(key);
        if (text.empty()) return fallback;
        std::vector<int> result;
        std::stringstream ss(text);
        std::string item;
        while (std::getline(ss, item, ',')) {
            try {
                result.push_back(std::stoi(item));
            }
            catch (...) {
                return fallback;
            }
        }
        return result.empty() ? fallback : result;
    }

private:
    static int parseInt(const std::string& text, int fallback) {
        if (text.empty()) return fallback;
        try { return std::stoi(text); }
        catch (...) { return fallback; }
    }

    static float parseFloat(const std::string& text, float fallback) {
        if (text.empty()) return fallback;
        try { return std::stof(text); }
        catch (...) { return fallback; }
    }

    static double parseDouble(const std::string& text, double fallback) {
        if (text.empty()) return fallback;
        try { return std::stod(text); }
        catch (...) { return fallback; }
    }

    static bool parseBool(std::string text, bool fallback) {
        if (text.empty()) return fallback;
        std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (text == "1" || text == "true" || text == "yes" || text == "on") return true;
        if (text == "0" || text == "false" || text == "no" || text == "off") return false;
        return fallback;
    }

private:
    bool mUseJson = false;
    Json::Value mJson;
    evkeyvalq mQuery{};
};

std::string resolvePath(const std::string& base, const std::string& value) {
    if (value.empty()) return {};
    std::filesystem::path path(value);
    if (path.is_absolute()) return path.lexically_normal().string();
    if (base.empty()) return path.lexically_normal().string();
    return (std::filesystem::path(base) / path).lexically_normal().string();
}

void resolveModelPaths(DetectParams& params, const Config* config) {
    const std::string model_dir = config ? config->modelDir : std::string();
    params.engine_file = resolvePath(model_dir, params.engine_file.empty() ? params.model_name : params.engine_file);
    params.sam_encoder_engine_file = resolvePath(model_dir, params.sam_encoder_name);
    params.sam_decoder_engine_file = resolvePath(model_dir, params.sam_decoder_name);
    params.sam_encoder_onnx_file = resolvePath(model_dir, params.sam_encoder_onnx_name);
    params.sam_decoder_onnx_file = resolvePath(model_dir, params.sam_decoder_onnx_name);
}

bool parseDetectParams(evhttp_request* req, RuntimeState* state, DetectParams& params, std::string& error) {
    RequestParams values;
    if (!values.load(req, error)) return false;

    params.model_name = values.stringValue("model_name", values.stringValue("yolo_model_name", params.model_name));
    params.engine_file = values.stringValue("trt_engine_file", params.engine_file);

    params.sam_encoder_name = values.stringValue("sam_encoder_name", params.sam_encoder_name);
    params.sam_decoder_name = values.stringValue("sam_decoder_name", params.sam_decoder_name);
    params.sam_encoder_onnx_name = values.stringValue("sam_encoder_onnx_name", params.sam_encoder_onnx_name);
    params.sam_decoder_onnx_name = values.stringValue("sam_decoder_onnx_name", params.sam_decoder_onnx_name);
    params.input_image_path = values.stringValue("input_image_path", params.input_image_path);

    params.enable_four_side_crop = values.boolValue("enable_four_side_crop", params.enable_four_side_crop);
    params.crop_x = values.intValue("crop_x", params.crop_x);
    params.crop_y = values.intValue("crop_y", params.crop_y);
    params.crop_width = values.intValue("crop_width", params.crop_width);
    params.crop_height = values.intValue("crop_height", params.crop_height);
    params.is_qw = values.boolValue("is_qw", params.is_qw);
    params.circle_x1 = values.intValue("circle_x1", params.circle_x1);
    params.circle_y1 = values.intValue("circle_y1", params.circle_y1);
    params.circle_x2 = values.intValue("circle_x2", params.circle_x2);
    params.circle_y2 = values.intValue("circle_y2", params.circle_y2);
    params.radius = values.intValue("radius", params.radius);

    params.enable_fourier_transform = values.boolValue("enable_fourier_transform", params.enable_fourier_transform);
    params.filter_width = values.intValue("filter_width", params.filter_width);
    params.attenuation_factor = values.doubleValue("attenuation_factor", params.attenuation_factor);
    params.target_angle = values.intValue("target_angle", params.target_angle);
    params.angle_tolerance = values.intValue("angle_tolerance", params.angle_tolerance);
    params.enable_denoising = values.boolValue("enable_denoising", params.enable_denoising);
    params.denoise_h = values.floatValue("denoise_h", params.denoise_h);
    params.denoise_hColor = values.floatValue("denoise_hColor", params.denoise_hColor);
    params.denoise_search_window = values.intValue("denoise_search_window", params.denoise_search_window);
    params.denoise_template_window = values.intValue("denoise_template_window", params.denoise_template_window);

    params.slice_width = values.intValue("slice_width", params.slice_width);
    params.slice_height = values.intValue("slice_height", params.slice_height);
    params.slice_distance = values.intValue("slice_distance", params.slice_distance);
    params.nms_threshold = values.floatValue("nms_threshold", params.nms_threshold);
    params.conf_threshold = values.floatValue("conf_threshold", params.conf_threshold);
    params.allowed_class_indices = values.intListValue("allowed_classes", params.allowed_class_indices);

    params.pix_to_mm = values.doubleValue("pix_to_mm", params.pix_to_mm);
    params.min_area_mm2 = values.doubleValue("min_area_mm2", params.min_area_mm2);
    params.min_diameter_mm = values.doubleValue("min_diameter_mm", params.min_diameter_mm);

    params.delay_ms = values.intValue("delay_ms", params.delay_ms);
    params.qw_index = values.intValue("qw_index", params.qw_index);
    params.device_id = values.stringValue("device_id", params.device_id);
    params.gpu_device = values.intValue(
        "gpu_device",
        state && state->config ? state->config->gpuDevice : params.gpu_device);

    resolveModelPaths(params, state ? state->config : nullptr);

    if (params.engine_file.empty()) {
        error = "model_name/trt_engine_file is required";
        return false;
    }
    if (params.nms_threshold <= 0.0f || params.conf_threshold <= 0.0f) {
        error = "nms_threshold/conf_threshold must be greater than 0";
        return false;
    }
    if (params.slice_width <= 0 || params.slice_height <= 0 || params.slice_distance < 0) {
        error = "slice dimensions must be positive and slice_distance must be non-negative";
        return false;
    }
    if (params.qw_index < 0 || params.qw_index >= static_cast<int>(kQuadImageCount)) {
        error = "qw_index must be in [0, 3]";
        return false;
    }
    if (params.gpu_device < 0) {
        error = "gpu_device must be non-negative";
        return false;
    }
    if (params.device_id.empty()) params.device_id = "CAM-001";
    return true;
}

DetectionContext makeDetectionContext(const RuntimeState* state) {
    DetectionContext context;
    if (!state || !state->config) return context;
    context.output_root = state->config->outputdir;
    context.run_id = state->run_id;
    return context;
}

Json::Value toJson(const FrameMeta& meta) {
    Json::Value value;
    value["device_id"] = meta.device_id;
    value["group_id"] = static_cast<Json::UInt64>(meta.group_id);
    value["index_in_group"] = meta.index_in_group;
    return value;
}

Json::Value toJson(const Detection& detection) {
    Json::Value value;
    value["label_id"] = detection.label_id;
    value["confidence"] = detection.confidence;
    value["box"]["x"] = detection.box.x;
    value["box"]["y"] = detection.box.y;
    value["box"]["w"] = detection.box.w;
    value["box"]["h"] = detection.box.h;
    value["length"] = detection.length;
    value["area"] = detection.area;
    return value;
}

Json::Value toJson(const SingleImageResult& result) {
    Json::Value value;
    value["meta"] = toJson(result.meta);
    Json::Value detections(Json::arrayValue);
    for (const auto& detection : result.detections) detections.append(toJson(detection));
    value["detections"] = std::move(detections);
    value["saved_path"] = result.saved_path;
    value["skeleton_path"] = result.skeleton_path;
    value["original_path"] = result.original_path;
    return value;
}

void sendJson(evhttp_request* req, int code, const std::string& message, const Json::Value& data = {}) {
    if (!req) return;

    std::unique_ptr<evbuffer, decltype(&evbuffer_free)> buffer(evbuffer_new(), &evbuffer_free);
    if (!buffer) return;

    Json::Value root;
    root["code"] = code;
    root["msg"] = message;
    if (!data.isNull()) root["data"] = data;

    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    const std::string payload = Json::writeString(writer, root);
    evbuffer_add(buffer.get(), payload.data(), payload.size());
    evhttp_add_header(evhttp_request_get_output_headers(req), "Content-Type", "application/json; charset=utf-8");
    evhttp_send_reply(req, 200, "OK", buffer.get());
}

void apiHealth(evhttp_request* req, void* arg) {
    auto* state = static_cast<RuntimeState*>(arg);
    Json::Value data;
    data["status"] = "ok";
    data["raw_queue_size"] = static_cast<Json::UInt64>(g_queue_manager.rawSizeApprox());
    data["result_queue_size"] = static_cast<Json::UInt64>(g_queue_manager.resultSizeApprox());
    data["task_running"] = g_scheduler && g_scheduler->isRunning();
    if (state && state->config) {
        data["host"] = state->config->ip;
        data["port"] = state->config->analyzerPort;
        data["modelDir"] = state->config->modelDir;
        data["uploadDir"] = state->config->outputdir;
        data["gpuDevice"] = state->config->gpuDevice;
    }
    sendJson(req, 0, "OK", data);
}

void apiControlAdd(evhttp_request* req, void* arg) {
    auto* state = static_cast<RuntimeState*>(arg);
    if (!req || !state) {
        sendJson(req, -1, "invalid request state");
        return;
    }

    DetectParams params;
    std::string error;
    if (!parseDetectParams(req, state, params, error)) {
        sendJson(req, -3, error);
        return;
    }

    if (!g_scheduler) g_scheduler = std::make_unique<Scheduler>();
    if (g_scheduler->isRunning()) g_scheduler->stop();

    state->last_params = params;
    state->has_task.store(true, std::memory_order_relaxed);

    if (!g_scheduler->startFromRuntimeState()) {
        state->has_task.store(false, std::memory_order_relaxed);
        sendJson(req, -4, "failed to start detection pipeline");
        return;
    }

    LOGI("detection task started: model=%s, device_id=%s, gpu=%d",
         params.engine_file.c_str(), params.device_id.c_str(), params.gpu_device);
    sendJson(req, 0, "detection task started");
}

void apiControlCancel(evhttp_request* req, void* arg) {
    auto* state = static_cast<RuntimeState*>(arg);
    if (!state) {
        sendJson(req, -1, "invalid runtime state");
        return;
    }
    state->has_task.store(false, std::memory_order_relaxed);
    if (g_scheduler) g_scheduler->stop();
    sendJson(req, 0, "task cancelled");
}

void apiSingleDetectAdd(evhttp_request* req, void* arg) {
    auto* state = static_cast<RuntimeState*>(arg);
    if (!req || !state) {
        sendJson(req, -1, "invalid request state");
        return;
    }

    DetectParams params;
    std::string error;
    if (!parseDetectParams(req, state, params, error)) {
        sendJson(req, -3, error);
        return;
    }
    if (params.input_image_path.empty()) {
        sendJson(req, -10, "input_image_path is required");
        return;
    }

    std::error_code ec;
    if (!std::filesystem::is_regular_file(params.input_image_path, ec)) {
        sendJson(req, -11, "input_image_path does not exist or is not a file");
        return;
    }

    if (!g_single_thread) {
        DetectionContext context = makeDetectionContext(state);
        if (!context.valid()) {
            sendJson(req, -12, "runtime output context is unavailable");
            return;
        }
        g_single_thread = std::make_unique<SingleDetectThread>(std::move(context));
    }

    SingleImageResult result;
    if (!g_single_thread->submitAndWait(params, result, error)) {
        sendJson(req, -12, "single-image detection failed: " + error);
        return;
    }
    sendJson(req, 0, "OK", toJson(result));
}

void apiSingleDetectCancel(evhttp_request* req, void*) {
    if (g_single_thread) {
        g_single_thread->stop();
        g_single_thread->join();
        g_single_thread.reset();
    }
    sendJson(req, 0, "single-image worker stopped");
}

void apiCameraSingleCapture(evhttp_request* req, void* arg) {
    auto* state = static_cast<RuntimeState*>(arg);
    if (!req || !state) {
        sendJson(req, -1, "invalid request state");
        return;
    }

    bool was_running = false;
    {
        std::lock_guard<std::mutex> lock(g_camera_mutex);
        if (!g_camera_thread) g_camera_thread = std::make_shared<CameraThread>();
        was_running = g_camera_thread->isRunning();
        if (!was_running) {
            const Config* cfg = state->config;
            if (!g_camera_thread->start(
                    state->last_params.delay_ms,
                    state->last_params.device_id,
                    cfg ? cfg->slidePort : std::string(),
                    cfg ? cfg->slideAxisId : 0,
                    cfg ? cfg->slideTimeoutMs : 20000,
                    false)) {
                sendJson(req, -20, "camera startup failed");
                return;
            }
        }
    }

    ImageFrame frame;
    try {
        frame = g_camera_thread->captureSingleImage(kSingleCaptureTimeoutMs).get();
    }
    catch (const std::exception& e) {
        sendJson(req, -21, std::string("capture failed: ") + e.what());
        return;
    }

    const DetectionContext context = makeDetectionContext(state);
    if (!context.valid()) {
        sendJson(req, -22, "runtime output context is unavailable");
        return;
    }
    const std::filesystem::path output_dir = context.singleOutputDir();
    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    if (ec) {
        sendJson(req, -22, "failed to create capture output directory");
        return;
    }

    const std::filesystem::path output_file = output_dir / "camera_image.png";
    const std::vector<int> encode_params{cv::IMWRITE_PNG_COMPRESSION, 0};
    if (frame.ps_image.empty() || !cv::imwrite(output_file.string(), frame.ps_image, encode_params)) {
        sendJson(req, -23, "failed to save captured image");
        return;
    }

    Json::Value data;
    data["saved_path"] = output_file.string();
    data["camera_running"] = g_camera_thread && g_camera_thread->isRunning();
    sendJson(req, 0, "OK", data);

    if (!was_running && (!g_scheduler || !g_scheduler->isRunning())) {
        std::lock_guard<std::mutex> lock(g_camera_mutex);
        if (g_camera_thread) {
            g_camera_thread->stop();
            g_camera_thread->join();
            g_camera_thread.reset();
        }
    }
}

void apiCameraStatus(evhttp_request* req, void*) {
    Json::Value data;
    data["camera_running"] = g_camera_thread && g_camera_thread->isRunning();
    sendJson(req, 0, "OK", data);
}

void apiCameraStop(evhttp_request* req, void*) {
    std::lock_guard<std::mutex> lock(g_camera_mutex);
    if (!g_camera_thread) {
        sendJson(req, -1, "camera worker does not exist");
        return;
    }
    g_camera_thread->stop();
    g_camera_thread->join();
    g_camera_thread.reset();
    sendJson(req, 0, "camera worker stopped");
}

void apiNotFound(evhttp_request* req, void*) {
    if (!req) return;
    std::unique_ptr<evbuffer, decltype(&evbuffer_free)> buffer(evbuffer_new(), &evbuffer_free);
    if (!buffer) return;
    evbuffer_add_printf(buffer.get(), "404 Not Found");
    evhttp_send_reply(req, 404, "Not Found", buffer.get());
}

} // namespace

bool Server::start(Config* config) {
    if (!config || !config->mState) {
        LOGE("invalid configuration; server not started");
        return false;
    }

#ifdef _WIN32
    WSADATA wsa_data{};
    const int wsa_error = WSAStartup(MAKEWORD(2, 2), &wsa_data);
    if (wsa_error != 0) {
        LOGE("WSAStartup failed: %d", wsa_error);
        return false;
    }
#endif

    g_runtime_state.config = config;

    std::unique_ptr<event_base, decltype(&event_base_free)> base(event_base_new(), &event_base_free);
    if (!base) {
        LOGE("event_base_new failed");
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }

    std::unique_ptr<evhttp, decltype(&evhttp_free)> http(evhttp_new(base.get()), &evhttp_free);
    if (!http) {
        LOGE("evhttp_new failed");
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }

    if (evhttp_bind_socket(http.get(), config->ip.c_str(), config->analyzerPort) != 0) {
        LOGE("failed to bind %s:%d", config->ip.c_str(), config->analyzerPort);
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }

    RuntimeState* state = &g_runtime_state;
    evhttp_set_cb(http.get(), "/api/health", apiHealth, state);
    evhttp_set_cb(http.get(), "/api/control/add", apiControlAdd, state);
    evhttp_set_cb(http.get(), "/api/control/cancel", apiControlCancel, state);
    evhttp_set_cb(http.get(), "/api/single_detect/add", apiSingleDetectAdd, state);
    evhttp_set_cb(http.get(), "/api/single_detect/cancel", apiSingleDetectCancel, state);
    evhttp_set_cb(http.get(), "/api/camera/single_capture", apiCameraSingleCapture, state);
    evhttp_set_cb(http.get(), "/api/camera/status", apiCameraStatus, state);
    evhttp_set_cb(http.get(), "/api/camera/stop", apiCameraStop, state);
    evhttp_set_gencb(http.get(), apiNotFound, nullptr);

    LOGI("HTTP server listening on http://%s:%d", config->ip.c_str(), config->analyzerPort);
    const int dispatch_result = event_base_dispatch(base.get());

    if (g_scheduler) g_scheduler->stop();
    if (g_single_thread) {
        g_single_thread->stop();
        g_single_thread->join();
        g_single_thread.reset();
    }

#ifdef _WIN32
    WSACleanup();
#endif

    if (dispatch_result < 0) {
        LOGE("event loop exited with error: %d", dispatch_result);
        return false;
    }
    return true;
}

} // namespace XL
