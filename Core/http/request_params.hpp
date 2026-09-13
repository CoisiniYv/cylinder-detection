#pragma once

#include <memory>
#include <string>
#include <vector>

struct evhttp_request;

namespace XL::http {

// Unified read-only view over either a JSON request body or URL query string.
// libevent/jsoncpp storage stays hidden behind PImpl so request mapping headers
// do not leak transport implementation details into the rest of the project.
class RequestParams {
public:
    RequestParams();
    ~RequestParams();

    RequestParams(const RequestParams&) = delete;
    RequestParams& operator=(const RequestParams&) = delete;

    bool load(evhttp_request* request, std::string& error_message);

    std::string stringValue(
        const char* key,
        const std::string& fallback = {}) const;
    int intValue(const char* key, int fallback) const;
    float floatValue(const char* key, float fallback) const;
    double doubleValue(const char* key, double fallback) const;
    bool boolValue(const char* key, bool fallback) const;
    std::vector<int> intListValue(
        const char* key,
        const std::vector<int>& fallback) const;

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace XL::http
