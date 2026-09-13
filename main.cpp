#include <cstdio>
#include <ctime>
#include <iostream>
#include <string>

#include "Core/Config.hpp"
#include "Core/Server.hpp"
#include "Core/db_utils.hpp"
#include "Core/runtime_state.hpp"

namespace {

void printUsage(const char* program) {
    std::printf("Usage: %s [-f <config.json>] [-h]\n", program ? program : "cylinder-detection");
    std::printf("  -f, --config   Path to runtime configuration (default: config.json)\n");
    std::printf("  -h, --help     Show this help and exit\n");
}

std::string makeRunId() {
    char time_buffer[32]{};
    const std::time_t now = std::time(nullptr);
    std::tm local_time{};
#if defined(_WIN32)
    localtime_s(&local_time, &now);
#else
    localtime_r(&now, &local_time);
#endif
    std::snprintf(
        time_buffer,
        sizeof(time_buffer),
        "%04d%02d%02d_%02d%02d%02d",
        local_time.tm_year + 1900,
        local_time.tm_mon + 1,
        local_time.tm_mday,
        local_time.tm_hour,
        local_time.tm_min,
        local_time.tm_sec);
    return std::string("run_") + time_buffer;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string config_file = "config.json";

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i] ? argv[i] : "";
            if (arg == "-h" || arg == "--help") {
                printUsage(argc > 0 ? argv[0] : nullptr);
                return 0;
            }
            if (arg == "-f" || arg == "--config") {
                if (i + 1 >= argc) {
                    std::fprintf(stderr, "Missing value after %s\n", arg.c_str());
                    printUsage(argc > 0 ? argv[0] : nullptr);
                    return 2;
                }
                config_file = argv[++i];
                continue;
            }

            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            printUsage(argc > 0 ? argv[0] : nullptr);
            return 2;
        }

        XL::Config config(config_file.c_str());
        if (!config.mState) {
            std::fprintf(stderr, "Failed to load configuration: %s\n", config_file.c_str());
            return 1;
        }
        config.show();

        const std::string db_file = config.dbPath.empty() ? "my_inspection.db" : config.dbPath;
        if (!XL::ensure_db_initialized(db_file)) {
            std::fprintf(stderr, "Failed to initialize database: %s\n", db_file.c_str());
            return 1;
        }

        XL::g_runtime_state.run_id = makeRunId();
        if (!XL::register_run(db_file, XL::g_runtime_state.run_id)) {
            std::fprintf(stderr, "Failed to register run_id: %s\n", XL::g_runtime_state.run_id.c_str());
            return 1;
        }

        XL::Server server;
        return server.start(&config) ? 0 : 1;
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Fatal error: unknown exception" << std::endl;
        return 1;
    }
}
