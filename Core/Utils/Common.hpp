#pragma once

#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace XL {

inline std::int64_t getCurTime() noexcept {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline std::string getCurFormatTimeStr(const char* format = "%Y-%m-%d %H:%M:%S") {
    if (!format) return {};

    const std::time_t now = std::time(nullptr);
    std::tm local_time{};
#if defined(_WIN32)
    localtime_s(&local_time, &now);
#else
    localtime_r(&now, &local_time);
#endif

    char buffer[128]{};
    const std::size_t length = std::strftime(buffer, sizeof(buffer), format, &local_time);
    return std::string(buffer, length);
}

inline std::vector<std::string> split(const std::string& value, const std::string& separator) {
    if (separator.empty()) return {value};

    std::vector<std::string> parts;
    std::size_t position = 0;
    while (position <= value.size()) {
        const std::size_t next = value.find(separator, position);
        if (next == std::string::npos) {
            const std::string tail = value.substr(position);
            if (!tail.empty()) parts.push_back(tail);
            break;
        }
        parts.push_back(value.substr(position, next - position));
        position = next + separator.size();
    }
    return parts;
}

inline bool removeFile(const std::string& filename) noexcept {
    std::error_code error;
    return std::filesystem::remove(filename, error) && !error;
}

inline int getRandomInt() {
    thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<int> distribution(10'000'000, 99'999'999);
    return distribution(generator);
}

} // namespace XL
