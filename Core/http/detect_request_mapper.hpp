#pragma once

#include <string>

struct evhttp_request;

namespace XL {

class Config;
struct DetectParams;

namespace http {

// Maps one libevent request into transport-independent DetectParams and applies
// request-level validation/path resolution. The rest of the application should
// consume DetectParams rather than HTTP/query/json primitives.
bool parseDetectParams(
    evhttp_request* request,
    const Config* config,
    DetectParams& params,
    std::string& error_message);

} // namespace http
} // namespace XL
