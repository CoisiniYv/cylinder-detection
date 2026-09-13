#pragma once

#include <cstdio>
#include <ctime>
#include <string>

namespace XL {

inline std::string logTime() {
    const std::time_t now = std::time(nullptr);
    std::tm local_time{};
#if defined(_WIN32)
    localtime_s(&local_time, &now);
#else
    localtime_r(&now, &local_time);
#endif

    char buffer[64]{};
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &local_time);
    return buffer;
}

} // namespace XL

#define LOGI(format, ...) \
    std::fprintf(stderr, "[INFO] %s [%s:%d] " format "\n", \
                 XL::logTime().c_str(), __func__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

#define LOGE(format, ...) \
    std::fprintf(stderr, "[ERROR] %s [%s:%d] " format "\n", \
                 XL::logTime().c_str(), __func__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

#define LOGC(format, ...) \
    std::fprintf(stderr, "[CONFIG] %s " format "\n", \
                 XL::logTime().c_str() __VA_OPT__(,) __VA_ARGS__)
