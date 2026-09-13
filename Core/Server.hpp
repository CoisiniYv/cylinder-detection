#pragma once

namespace XL {

class Config;

// HTTP control plane for the inspection service.
//
// Public responsibilities are intentionally small: bind the configured HTTP
// endpoint and run the libevent loop. Request parsing, API callbacks and
// runtime object ownership stay private to Server.cpp.
class Server {
public:
    Server() = default;
    ~Server() = default;

    // Blocking event loop. Returns false when startup/binding fails.
    bool start(Config* config);
};

} // namespace XL
