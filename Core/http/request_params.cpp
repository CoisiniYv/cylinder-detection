#include "request_params.hpp"

#include <event2/buffer.h>
#include <event2/http.h>
#include <event2/keyvalq_struct.h>
#include <json/json.h>

#include <algorithm>
#include <cctype>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace XL::http {
namespace {

constexpr std::size_t kMaxRequestBodyBytes = 64 * 1024;

int parseInt(const std::string& text, int fallback) {
    if (text.empty()) return fallback;
    try { return std::stoi(text); }
    catch (...) { return fallback; }
}

float parseFloat(const std::string& text, float fallback) {
    if (text.empty()) return fallback;
    try { return std::stof(text); }
    catch (...) { return fallback; }
}

double parseDouble(const std::string& text, double fallback) {
    if (text.empty()) return fallback;
    try { return std::stod(text); }
    catch (...) { return fallback; }
}

bool parseBool(std::string text, bool fallback) {
    if (text.empty()) return fallback;
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (text == "1" || text == "true" || text == "yes" || text == "on") return true;
    if (text == "0" || text == "false" || text == "no" || text == "off") return false;
    return fallback;
}

} // namespace

struct RequestParams::Impl {
    bool use_json = false;
    Json::Value json;
    evkeyvalq query{};

    Impl() {
        evhttp_parse_query_str("", &query);
    }

    ~Impl() {
        evhttp_clear_headers(&query);
    }
};

RequestParams::RequestParams()
    : mImpl(std::make_unique<Impl>()) {}

RequestParams::~RequestParams() = default;

bool RequestParams::load(
    evhttp_request* request,
    std::string& error_message) {
    error_message.clear();
    if (!request) {
        error_message = "request is null";
        return false;
    }

    evbuffer* input = evhttp_request_get_input_buffer(request);
    const std::size_t body_size = input ? evbuffer_get_length(input) : 0;
    if (body_size > 0) {
        if (body_size > kMaxRequestBodyBytes) {
            error_message = "request body exceeds 64 KiB";
            return false;
        }

        const unsigned char* body = evbuffer_pullup(input, -1);
        if (!body) {
            error_message = "unable to read request body";
            return false;
        }

        Json::CharReaderBuilder builder;
        builder["collectComments"] = false;
        std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
        JSONCPP_STRING parse_error;
        if (!reader->parse(
                reinterpret_cast<const char*>(body),
                reinterpret_cast<const char*>(body) + body_size,
                &mImpl->json,
                &parse_error)) {
            error_message = "invalid JSON: " + parse_error;
            return false;
        }

        mImpl->use_json = true;
        return true;
    }

    const char* uri_text = evhttp_request_get_uri(request);
    if (!uri_text) return true;

    evhttp_uri* uri = evhttp_uri_parse(uri_text);
    if (!uri) {
        error_message = "invalid request URI";
        return false;
    }

    const char* query_text = evhttp_uri_get_query(uri);
    if (query_text && evhttp_parse_query_str(query_text, &mImpl->query) != 0) {
        evhttp_uri_free(uri);
        error_message = "invalid query string";
        return false;
    }

    evhttp_uri_free(uri);
    return true;
}

std::string RequestParams::stringValue(
    const char* key,
    const std::string& fallback) const {
    if (mImpl->use_json) {
        const auto& value = mImpl->json[key];
        if (value.isNull()) return fallback;
        return value.asString();
    }

    const char* value = evhttp_find_header(&mImpl->query, key);
    return value ? std::string(value) : fallback;
}

int RequestParams::intValue(const char* key, int fallback) const {
    if (mImpl->use_json) {
        const auto& value = mImpl->json[key];
        if (value.isNull()) return fallback;
        if (value.isInt() || value.isUInt()) return value.asInt();
        return parseInt(value.asString(), fallback);
    }
    return parseInt(stringValue(key), fallback);
}

float RequestParams::floatValue(const char* key, float fallback) const {
    if (mImpl->use_json) {
        const auto& value = mImpl->json[key];
        if (value.isNull()) return fallback;
        if (value.isNumeric()) return value.asFloat();
        return parseFloat(value.asString(), fallback);
    }
    return parseFloat(stringValue(key), fallback);
}

double RequestParams::doubleValue(const char* key, double fallback) const {
    if (mImpl->use_json) {
        const auto& value = mImpl->json[key];
        if (value.isNull()) return fallback;
        if (value.isNumeric()) return value.asDouble();
        return parseDouble(value.asString(), fallback);
    }
    return parseDouble(stringValue(key), fallback);
}

bool RequestParams::boolValue(const char* key, bool fallback) const {
    if (mImpl->use_json) {
        const auto& value = mImpl->json[key];
        if (value.isNull()) return fallback;
        if (value.isBool()) return value.asBool();
        return parseBool(value.asString(), fallback);
    }
    return parseBool(stringValue(key), fallback);
}

std::vector<int> RequestParams::intListValue(
    const char* key,
    const std::vector<int>& fallback) const {
    if (mImpl->use_json) {
        const auto& value = mImpl->json[key];
        if (value.isNull() || !value.isArray()) return fallback;

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
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        try {
            result.push_back(std::stoi(item));
        }
        catch (...) {
            return fallback;
        }
    }
    return result.empty() ? fallback : result;
}

} // namespace XL::http
