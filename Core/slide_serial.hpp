#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace XL {

struct SerialConfig {
    std::string port;
    int baud = 115200;
    int timeout_ms = 100;
};

// Line-oriented serial transport for the slide controller.
// Protocol commands remain in CameraThread; this class only owns the serial
// handle, reader thread and received-line buffering.
class SlideSerialClient {
public:
    SlideSerialClient() = default;
    ~SlideSerialClient();

    SlideSerialClient(const SlideSerialClient&) = delete;
    SlideSerialClient& operator=(const SlideSerialClient&) = delete;

    bool Open(const SerialConfig& config);
    void Close();
    bool IsOpen() const noexcept;
    bool WriteLine(const std::string& line);
    std::vector<std::string> DrainLines();

    static std::vector<std::string> ListPorts();

private:
    void readerLoop();
    static std::string normalizePortName(const std::string& port);

private:
    void* mHandle = nullptr; // HANDLE on Windows; intentionally opaque in public header
    std::atomic<bool> mRunning{false};
    std::thread mThread;
    SerialConfig mConfig{};

    std::mutex mMutex;
    std::deque<std::string> mLines;
    std::string mBuffer;
};

} // namespace XL
