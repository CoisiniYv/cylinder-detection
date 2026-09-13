#include "Config.hpp"
#include "Utils/Log.hpp"

#include <filesystem>
#include <fstream>
#include <json/json.h>

namespace XL {
namespace {

std::filesystem::path resolve_path(const std::filesystem::path& config_dir, const std::string& raw) {
    if (raw.empty()) return {};
    std::filesystem::path p(raw);
    if (p.is_absolute()) return p.lexically_normal();
    return (config_dir / p).lexically_normal();
}

} // namespace

Config::Config(const char* file) {
    sourceFile = file ? file : "";
    if (sourceFile.empty()) {
        LOGE("configuration path is empty");
        return;
    }

    std::ifstream ifs(sourceFile, std::ios::binary);
    if (!ifs.is_open()) {
        LOGE("open %s error", sourceFile.c_str());
        return;
    }

    Json::CharReaderBuilder builder;
    builder["collectComments"] = false;
    JSONCPP_STRING errs;
    Json::Value root;
    if (!parseFromStream(builder, ifs, &root, &errs)) {
        LOGE("parse %s error: %s", sourceFile.c_str(), errs.c_str());
        return;
    }

    ip = root.isMember("host") ? root["host"].asString() : root.get("ip", ip).asString();
    analyzerPort = root.get("analyzerPort", analyzerPort).asInt();
    slidePort = root.get("slidePort", slidePort).asString();
    slideAxisId = root.get("slideAxisId", slideAxisId).asInt();
    slideTimeoutMs = root.get("slideTimeoutMs", slideTimeoutMs).asInt();
    gpuDevice = root.get("gpuDevice", gpuDevice).asInt();
    detectThreads = root.get("detectThreads", detectThreads).asInt();
    groupTimeoutMs = root.get("groupTimeoutMs", groupTimeoutMs).asInt();

    if (ip.empty()) ip = "127.0.0.1";
    if (analyzerPort <= 0 || analyzerPort > 65535) {
        LOGE("invalid analyzerPort: %d", analyzerPort);
        return;
    }
    if (slideTimeoutMs <= 0) slideTimeoutMs = 20000;
    if (gpuDevice < 0) gpuDevice = 0;
    if (detectThreads <= 0) detectThreads = 1;
    if (groupTimeoutMs <= 0) groupTimeoutMs = 55000;

    const std::filesystem::path cfg_abs = std::filesystem::absolute(std::filesystem::path(sourceFile));
    const std::filesystem::path config_dir = cfg_abs.parent_path();

    modelDir = resolve_path(config_dir, root.get("modelDir", modelDir).asString()).string();
    outputdir = resolve_path(config_dir, root.get("uploadDir", outputdir).asString()).string();
    dbPath = resolve_path(config_dir, root.get("dbPath", dbPath).asString()).string();

    std::error_code ec;
    if (!outputdir.empty()) {
        std::filesystem::create_directories(outputdir, ec);
        if (ec) {
            LOGE("create output directory failed: %s (%s)", outputdir.c_str(), ec.message().c_str());
            return;
        }
    }

    if (!dbPath.empty()) {
        const auto parent = std::filesystem::path(dbPath).parent_path();
        if (!parent.empty()) {
            ec.clear();
            std::filesystem::create_directories(parent, ec);
            if (ec) {
                LOGE("create database directory failed: %s (%s)", parent.string().c_str(), ec.message().c_str());
                return;
            }
        }
    }

    mState = true;
}

void Config::show() const {
    LOGC("config: %s", sourceFile.c_str());
    LOGC("ip: %s", ip.c_str());
    LOGC("analyzerPort: %d", analyzerPort);
    LOGC("uploadDir: %s", outputdir.c_str());
    LOGC("modelDir: %s", modelDir.c_str());
    LOGC("dbPath: %s", dbPath.c_str());
    LOGC("slidePort: %s", slidePort.empty() ? "(disabled)" : slidePort.c_str());
    LOGC("slideAxisId: %d", slideAxisId);
    LOGC("slideTimeoutMs: %d", slideTimeoutMs);
    LOGC("gpuDevice: %d", gpuDevice);
    LOGC("detectThreads: %d", detectThreads);
    LOGC("groupTimeoutMs: %d", groupTimeoutMs);
}

} // namespace XL
