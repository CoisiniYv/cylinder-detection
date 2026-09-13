#include <cstdio>
#include <ctime>
#include <iostream>
#include <string>

#include "Core/Config.hpp"
#include "Core/Server.hpp"
#include "Core/db_utils.hpp"

using namespace XL;

namespace {

void print_usage(const char* program) {
    std::printf("Usage: %s [-f <config.json>] [-h]\n", program ? program : "cylinder-detection");
    std::printf("  -f, --config   Path to runtime configuration (default: config.json)\n");
    std::printf("  -h, --help     Show this help and exit\n");
}

std::string make_run_id() {
    char timebuf[32]{};
    const std::time_t now = std::time(nullptr);
    std::tm local_tm{};
    localtime_s(&local_tm, &now);
    std::snprintf(
        timebuf,
        sizeof(timebuf),
        "%04d%02d%02d_%02d%02d%02d",
        local_tm.tm_year + 1900,
        local_tm.tm_mon + 1,
        local_tm.tm_mday,
        local_tm.tm_hour,
        local_tm.tm_min,
        local_tm.tm_sec);
    return std::string("run_") + timebuf;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string config_file = "config.json";

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i] ? argv[i] : "";
            if (arg == "-h" || arg == "--help") {
                print_usage(argc > 0 ? argv[0] : nullptr);
                return 0;
            }
            if (arg == "-f" || arg == "--config") {
                if (i + 1 >= argc) {
                    std::fprintf(stderr, "Missing value after %s\n", arg.c_str());
                    print_usage(argc > 0 ? argv[0] : nullptr);
                    return 2;
                }
                config_file = argv[++i];
                continue;
            }

            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argc > 0 ? argv[0] : nullptr);
            return 2;
        }

        Config config(config_file.c_str());
        if (!config.mState) {
            std::fprintf(stderr, "Failed to load configuration: %s\n", config_file.c_str());
            return 1;
        }
        config.show();

        const std::string dbfile = config.dbPath.empty() ? std::string("my_inspection.db") : config.dbPath;
        if (!ensure_db_initialized(dbfile)) {
            std::fprintf(stderr, "Failed to initialize database: %s\n", dbfile.c_str());
            return 1;
        }

        const std::string run_id = make_run_id();
        if (!register_run(dbfile, run_id)) {
            std::fprintf(stderr, "Failed to register run_id: %s\n", run_id.c_str());
            return 1;
        }

        extern XL::ServerState g_server_state;
        g_server_state.run_id = run_id;

        Server server;
        server.start(&config);
        return 0;
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
