#pragma once

#include "detect_params.hpp"

#include <atomic>
#include <memory>
#include <string>

namespace XL {

class CameraThread;
class Config;

// Process-level runtime state shared by the HTTP control plane and scheduler.
// Ownership remains with main/Server; workers should consume task-local copies
// of DetectParams instead of reaching into this object for algorithm settings.
struct RuntimeState {
    Config* config = nullptr; // non-owning; Config lives for the process lifetime
    DetectParams last_params;
    std::atomic<bool> has_task{false};
    std::string run_id;
};

extern RuntimeState g_runtime_state;
extern std::shared_ptr<CameraThread> g_camera_thread;

} // namespace XL
