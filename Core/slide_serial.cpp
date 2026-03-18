#include "slide_serial.hpp"

#include <algorithm>
#include <chrono>

namespace XL {

SlideSerialClient::SlideSerialClient() = default;

SlideSerialClient::~SlideSerialClient() {
    Close();
}

bool SlideSerialClient::IsOpen() const noexcept {
#ifdef _WIN32
    return mHandle != INVALID_HANDLE_VALUE;
#else
    return mHandle != nullptr;
#endif
}

std::string SlideSerialClient::normalizePortName(const std::string& port) {
    if (port.empty()) return port;
#ifdef _WIN32
    // For COM10+, use \\.\COM10 format
    if (port.rfind("\\\\.\\", 0) == 0) return port;
    if (port.size() > 4 && (port.rfind("COM", 0) == 0 || port.rfind("com", 0) == 0)) {
        return std::string("\\\\.\\") + port;
    }
#endif
    return port;
}

bool SlideSerialClient::Open(const SerialConfig& cfg) {
    Close();
    mCfg = cfg;

#ifdef _WIN32
    std::string portName = normalizePortName(cfg.port);
    if (portName.empty()) return false;

    mHandle = CreateFileA(
        portName.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL);

    if (mHandle == INVALID_HANDLE_VALUE) {
        return false;
    }

    DCB dcb{};
    dcb.DCBlength = sizeof(dcb);
    if (!GetCommState(mHandle, &dcb)) {
        Close();
        return false;
    }

    dcb.BaudRate = cfg.baud;
    dcb.ByteSize = 8;
    dcb.StopBits = ONESTOPBIT;
    dcb.Parity = NOPARITY;
    dcb.fBinary = TRUE;
    dcb.fDtrControl = DTR_CONTROL_ENABLE;
    dcb.fRtsControl = RTS_CONTROL_ENABLE;

    if (!SetCommState(mHandle, &dcb)) {
        Close();
        return false;
    }

    COMMTIMEOUTS timeouts{};
    timeouts.ReadIntervalTimeout = 50;
    timeouts.ReadTotalTimeoutConstant = cfg.timeout_ms;
    timeouts.ReadTotalTimeoutMultiplier = 10;
    timeouts.WriteTotalTimeoutConstant = cfg.timeout_ms;
    timeouts.WriteTotalTimeoutMultiplier = 10;
    if (!SetCommTimeouts(mHandle, &timeouts)) {
        Close();
        return false;
    }

    mRunning.store(true, std::memory_order_relaxed);
    mThread = std::thread(&SlideSerialClient::readerLoop, this);
    return true;
#else
    (void)cfg;
    return false;
#endif
}

void SlideSerialClient::Close() {
#ifdef _WIN32
    mRunning.store(false, std::memory_order_relaxed);
    if (mThread.joinable()) {
        mThread.join();
    }
    if (mHandle != INVALID_HANDLE_VALUE) {
        CloseHandle(mHandle);
        mHandle = INVALID_HANDLE_VALUE;
    }
#else
    mRunning.store(false, std::memory_order_relaxed);
    if (mThread.joinable()) {
        mThread.join();
    }
    mHandle = nullptr;
#endif

    std::lock_guard<std::mutex> lk(mMtx);
    mLines.clear();
    mBuf.clear();
}

bool SlideSerialClient::WriteLine(const std::string& line) {
#ifdef _WIN32
    if (!IsOpen()) return false;
    std::string payload = line;
    if (payload.empty() || (payload.back() != '\n' && payload.back() != '\r')) {
        payload.push_back('\n');
    }
    DWORD written = 0;
    return WriteFile(mHandle, payload.c_str(), (DWORD)payload.size(), &written, NULL) != 0;
#else
    (void)line;
    return false;
#endif
}

std::vector<std::string> SlideSerialClient::DrainLines() {
    std::vector<std::string> out;
    std::lock_guard<std::mutex> lk(mMtx);
    while (!mLines.empty()) {
        out.emplace_back(std::move(mLines.front()));
        mLines.pop_front();
    }
    return out;
}

void SlideSerialClient::readerLoop() {
#ifdef _WIN32
    constexpr size_t kMaxLineLen = 63;
    char buf[128];
    while (mRunning.load(std::memory_order_relaxed)) {
        DWORD bytesRead = 0;
        BOOL ok = ReadFile(mHandle, buf, sizeof(buf), &bytesRead, NULL);
        if (!ok) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        if (bytesRead == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        for (DWORD i = 0; i < bytesRead; ++i) {
            char c = buf[i];
            if (c == '\r' || c == '\n') {
                if (!mBuf.empty()) {
                    std::lock_guard<std::mutex> lk(mMtx);
                    mLines.push_back(mBuf);
                    mBuf.clear();
                }
                continue;
            }
            if (mBuf.size() >= kMaxLineLen) {
                mBuf.clear();
                continue;
            }
            mBuf.push_back(c);
        }
    }
#endif
}

std::vector<std::string> SlideSerialClient::ListPorts() {
    std::vector<std::string> ports;
#ifdef _WIN32
    // Best-effort enumeration: try COM1..COM64
    for (int i = 1; i <= 64; ++i) {
        std::string name = "COM" + std::to_string(i);
        std::string portName = normalizePortName(name);
        HANDLE h = CreateFileA(portName.c_str(),
                               GENERIC_READ | GENERIC_WRITE,
                               0,
                               NULL,
                               OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL,
                               NULL);
        if (h != INVALID_HANDLE_VALUE) {
            ports.push_back(name);
            CloseHandle(h);
        }
    }
#endif
    return ports;
}

} // namespace XL
