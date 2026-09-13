#include "Server.hpp"

#include "Config.hpp"
#include "Scheduler.hpp"
#include "Utils/Log.hpp"
#include "camera_thread.hpp"
#include "detection_context.hpp"
#include "http/detect_request_mapper.hpp"
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
#include <json/json.h>
#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace XL {

RuntimeState g_runtime_state;
std::shared_ptr<CameraThread> g_camera_thread;

namespace {

constexpr int kSingleCaptureTimeoutMs = 10000;

std::unique_ptr<Scheduler> g_scheduler;
std::unique_ptr<SingleDetectThread> g_single_thread;
std::mutex g_camera_mutex;

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
    value["processing_ok"] = result.processing_ok;
    if (!result.error_message.empty()) value["error_message"] = result.error_message;

    Json::Value detections(Json::arrayValue);
    for (const auto& detection : result.detections) detections.append(toJson(detection));
    value["detections"] = std::move(detections);
    value["saved_path"] = result.saved_path;
    value["skeleton_path"] = result.skeleton_path;
    value["original_path"] = result.original_path;
    return value;
}

void sendJson(
    evhttp_request* request,
    int code,
    const std::string& message,
    const Json::Value& data = {}) {
    if (!request) return;

    std::unique_ptr<evbuffer, decltype(&evbuffer_free)> buffer(
        evbuffer_new(),
        &evbuffer_free);
    if (!buffer) return;

    Json::Value root;
    root["code"] = code;
    root["msg"] = message;
    if (!data.isNull()) root["data"] = data;

    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    const std::string payload = Json::writeString(writer, root);
    evbuffer_add(buffer.get(), payload.data(), payload.size());
    evhttp_add_header(
        evhttp_request_get_output_headers(request),
        "Content-Type",
        "application/json; charset=utf-8");
    evhttp_send_reply(request, 200, "OK", buffer.get());
}

void releaseTemporaryCamera(bool was_running_before_request) {
    if (was_running_before_request) return;

    std::lock_guard<std::mutex> lock(g_camera_mutex);
    if (g_scheduler && g_scheduler->isRunning()) return;
    if (!g_camera_thread) return;

    g_camera_thread->stop();
    g_camera_thread->join();
    g_camera_thread.reset();
}

void apiHealth(evhttp_request* request, void* argument) {
    auto* state = static_cast<RuntimeState*>(argument);

    Json::Value data;
    data["status"] = "ok";
    data["raw_queue_size"] = static_cast<Json::UInt64>(g_queue_manager.rawSizeApprox());
    data["result_queue_size"] = static_cast<Json::UInt64>(g_queue_manager.resultSizeApprox());
    data["task_running"] = g_scheduler && g_scheduler->isRunning();
    data["single_worker_running"] = g_single_thread && g_single_thread->isRunning();
    data["camera_running"] = g_camera_thread && g_camera_thread->isRunning();

    if (state && state->config) {
        data["host"] = state->config->ip;
        data["port"] = state->config->analyzerPort;
        data["modelDir"] = state->config->modelDir;
        data["uploadDir"] = state->config->outputdir;
        data["gpuDevice"] = state->config->gpuDevice;
        data["detectThreads"] = state->config->detectThreads;
    }

    sendJson(request, 0, "OK", data);
}

void apiControlAdd(evhttp_request* request, void* argument) {
    auto* state = static_cast<RuntimeState*>(argument);
    if (!request || !state) {
        sendJson(request, -1, "invalid request state");
        return;
    }

    DetectParams params;
    std::string error;
    if (!http::parseDetectParams(request, state->config, params, error)) {
        sendJson(request, -3, error);
        return;
    }

    if (!g_scheduler) g_scheduler = std::make_unique<Scheduler>();
    if (g_scheduler->isRunning()) g_scheduler->stop();

    state->last_params = params;
    state->has_task.store(true, std::memory_order_relaxed);

    if (!g_scheduler->startFromRuntimeState()) {
        state->has_task.store(false, std::memory_order_relaxed);
        sendJson(request, -4, "failed to start detection pipeline");
        return;
    }

    LOGI("detection task started: model=%s device_id=%s gpu=%d",
         params.engine_file.c_str(),
         params.device_id.c_str(),
         params.gpu_device);
    sendJson(request, 0, "detection task started");
}

void apiControlCancel(evhttp_request* request, void* argument) {
    auto* state = static_cast<RuntimeState*>(argument);
    if (!state) {
        sendJson(request, -1, "invalid runtime state");
        return;
    }

    state->has_task.store(false, std::memory_order_relaxed);
    if (g_scheduler) g_scheduler->stop();
    sendJson(request, 0, "task cancelled");
}

void apiSingleDetectAdd(evhttp_request* request, void* argument) {
    auto* state = static_cast<RuntimeState*>(argument);
    if (!request || !state) {
        sendJson(request, -1, "invalid request state");
        return;
    }

    DetectParams params;
    std::string error;
    if (!http::parseDetectParams(request, state->config, params, error)) {
        sendJson(request, -3, error);
        return;
    }
    if (params.input_image_path.empty()) {
        sendJson(request, -10, "input_image_path is required");
        return;
    }

    std::error_code file_error;
    if (!std::filesystem::is_regular_file(params.input_image_path, file_error)) {
        sendJson(request, -11, "input_image_path does not exist or is not a file");
        return;
    }

    if (!g_single_thread) {
        DetectionContext context = makeDetectionContext(state);
        if (!context.valid()) {
            sendJson(request, -12, "runtime output context is unavailable");
            return;
        }
        g_single_thread = std::make_unique<SingleDetectThread>(std::move(context));
    }

    SingleImageResult result;
    if (!g_single_thread->submitAndWait(params, result, error)) {
        sendJson(request, -12, "single-image detection failed: " + error);
        return;
    }
    sendJson(request, 0, "OK", toJson(result));
}

void apiSingleDetectCancel(evhttp_request* request, void*) {
    if (g_single_thread) {
        g_single_thread->stop();
        g_single_thread->join();
        g_single_thread.reset();
    }
    sendJson(request, 0, "single-image worker stopped");
}

void apiCameraSingleCapture(evhttp_request* request, void* argument) {
    auto* state = static_cast<RuntimeState*>(argument);
    if (!request || !state) {
        sendJson(request, -1, "invalid request state");
        return;
    }

    bool was_running = false;
    {
        std::lock_guard<std::mutex> lock(g_camera_mutex);
        if (!g_camera_thread) g_camera_thread = std::make_shared<CameraThread>();
        was_running = g_camera_thread->isRunning();

        if (!was_running) {
            const Config* config = state->config;
            if (!g_camera_thread->start(
                    state->last_params.delay_ms,
                    state->last_params.device_id,
                    config ? config->slidePort : std::string(),
                    config ? config->slideAxisId : 0,
                    config ? config->slideTimeoutMs : 20000,
                    false)) {
                sendJson(request, -20, "camera startup failed");
                g_camera_thread.reset();
                return;
            }
        }
    }

    ImageFrame frame;
    try {
        frame = g_camera_thread->captureSingleImage(kSingleCaptureTimeoutMs).get();
    }
    catch (const std::exception& e) {
        releaseTemporaryCamera(was_running);
        sendJson(request, -21, std::string("capture failed: ") + e.what());
        return;
    }

    const DetectionContext context = makeDetectionContext(state);
    if (!context.valid()) {
        releaseTemporaryCamera(was_running);
        sendJson(request, -22, "runtime output context is unavailable");
        return;
    }

    const std::filesystem::path output_dir = context.singleOutputDir();
    std::error_code directory_error;
    std::filesystem::create_directories(output_dir, directory_error);
    if (directory_error) {
        releaseTemporaryCamera(was_running);
        sendJson(request, -22, "failed to create capture output directory");
        return;
    }

    const std::filesystem::path output_file = output_dir / "camera_image.png";
    const std::vector<int> encode_params{cv::IMWRITE_PNG_COMPRESSION, 0};
    if (frame.ps_image.empty() ||
        !cv::imwrite(output_file.string(), frame.ps_image, encode_params)) {
        releaseTemporaryCamera(was_running);
        sendJson(request, -23, "failed to save captured image");
        return;
    }

    Json::Value data;
    data["saved_path"] = output_file.string();
    data["camera_running"] = g_camera_thread && g_camera_thread->isRunning();
    sendJson(request, 0, "OK", data);
    releaseTemporaryCamera(was_running);
}

void apiCameraStatus(evhttp_request* request, void*) {
    Json::Value data;
    data["camera_running"] = g_camera_thread && g_camera_thread->isRunning();
    data["owned_by_scheduler"] = g_scheduler && g_scheduler->isRunning();
    sendJson(request, 0, "OK", data);
}

void apiCameraStop(evhttp_request* request, void*) {
    if (g_scheduler && g_scheduler->isRunning()) {
        sendJson(request, -2, "camera is owned by an active detection task");
        return;
    }

    std::lock_guard<std::mutex> lock(g_camera_mutex);
    if (!g_camera_thread) {
        sendJson(request, -1, "camera worker does not exist");
        return;
    }

    g_camera_thread->stop();
    g_camera_thread->join();
    g_camera_thread.reset();
    sendJson(request, 0, "camera worker stopped");
}

void apiNotFound(evhttp_request* request, void*) {
    if (!request) return;

    std::unique_ptr<evbuffer, decltype(&evbuffer_free)> buffer(
        evbuffer_new(),
        &evbuffer_free);
    if (!buffer) return;

    evbuffer_add_printf(buffer.get(), "404 Not Found");
    evhttp_send_reply(request, 404, "Not Found", buffer.get());
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

    std::unique_ptr<event_base, decltype(&event_base_free)> base(
        event_base_new(),
        &event_base_free);
    if (!base) {
        LOGE("event_base_new failed");
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }

    std::unique_ptr<evhttp, decltype(&evhttp_free)> http_server(
        evhttp_new(base.get()),
        &evhttp_free);
    if (!http_server) {
        LOGE("evhttp_new failed");
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }

    if (evhttp_bind_socket(
            http_server.get(),
            config->ip.c_str(),
            config->analyzerPort) != 0) {
        LOGE("failed to bind %s:%d", config->ip.c_str(), config->analyzerPort);
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }

    RuntimeState* state = &g_runtime_state;
    evhttp_set_cb(http_server.get(), "/api/health", apiHealth, state);
    evhttp_set_cb(http_server.get(), "/api/control/add", apiControlAdd, state);
    evhttp_set_cb(http_server.get(), "/api/control/cancel", apiControlCancel, state);
    evhttp_set_cb(http_server.get(), "/api/single_detect/add", apiSingleDetectAdd, state);
    evhttp_set_cb(http_server.get(), "/api/single_detect/cancel", apiSingleDetectCancel, state);
    evhttp_set_cb(http_server.get(), "/api/camera/single_capture", apiCameraSingleCapture, state);
    evhttp_set_cb(http_server.get(), "/api/camera/status", apiCameraStatus, state);
    evhttp_set_cb(http_server.get(), "/api/camera/stop", apiCameraStop, state);
    evhttp_set_gencb(http_server.get(), apiNotFound, nullptr);

    LOGI("HTTP server listening on http://%s:%d", config->ip.c_str(), config->analyzerPort);
    const int dispatch_result = event_base_dispatch(base.get());

    if (g_scheduler) g_scheduler->stop();
    if (g_single_thread) {
        g_single_thread->stop();
        g_single_thread->join();
        g_single_thread.reset();
    }

    {
        std::lock_guard<std::mutex> lock(g_camera_mutex);
        if (g_camera_thread && (!g_scheduler || !g_scheduler->isRunning())) {
            g_camera_thread->stop();
            g_camera_thread->join();
            g_camera_thread.reset();
        }
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
