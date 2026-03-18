#pragma once

#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <thread>
#include <atomic>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace XL {

struct SerialConfig {
    std::string port;
    int baud = 115200;
    int timeout_ms = 100;
};

class SlideSerialClient {
public:
    SlideSerialClient();
    ~SlideSerialClient();

    bool Open(const SerialConfig& cfg);
    void Close();
    bool IsOpen() const noexcept;

    // send a line, auto appends '\n'
    bool WriteLine(const std::string& line);

    // non-blocking drain of received lines
    std::vector<std::string> DrainLines();

    // enumerate ports (best-effort)
    static std::vector<std::string> ListPorts();

private:
    void readerLoop();
    static std::string normalizePortName(const std::string& port);

private:
#ifdef _WIN32
    HANDLE mHandle = (HANDLE)(uintptr_t)-1;
#else
    void* mHandle = nullptr;
#endif
    std::atomic<bool> mRunning{ false };
    std::thread mThread;
    SerialConfig mCfg{};

    std::mutex mMtx;
    std::deque<std::string> mLines;
    std::string mBuf;
};

} // namespace XL
