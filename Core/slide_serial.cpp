#include "slide_serial.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <chrono>

namespace XL {
namespace {

#ifdef _WIN32
HANDLE asHandle(void* handle) {
    return static_cast<HANDLE>(handle);
}
#endif

} // namespace

SlideSerialClient::~SlideSerialClient() {
    Close();
}

bool SlideSerialClient::IsOpen() const noexcept {
    return mHandle != nullptr;
}

std::string SlideSerialClient::normalizePortName(const std::string& port) {
    if (port.empty()) return port;
#ifdef _WIN32
    if (port.rfind("\\\\.\\", 0) == 0) return port;
    if (port.size() > 4 && (port.rfind("COM", 0) == 0 || port.rfind("com", 0) == 0)) {
        return "\\\\.\\" + port;
    }
#endif
    return port;
}

bool SlideSerialClient::Open(const SerialConfig& config) {
    Close();
    mConfig = config;

#ifdef _WIN32
    const std::string port_name = normalizePortName(config.port);
    if (port_name.empty() || config.baud <= 0 || config.timeout_ms < 0) return false;

    HANDLE handle = CreateFileA(
        port_name.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (handle == INVALID_HANDLE_VALUE) return false;

    DCB dcb{};
    dcb.DCBlength = sizeof(dcb);
    if (!GetCommState(handle, &dcb)) {
        CloseHandle(handle);
        return false;
    }

    dcb.BaudRate = static_cast<DWORD>(config.baud);
    dcb.ByteSize = 8;
    dcb.StopBits = ONESTOPBIT;
    dcb.Parity = NOPARITY;
    dcb.fBinary = TRUE;
    dcb.fDtrControl = DTR_CONTROL_ENABLE;
    dcb.fRtsControl = RTS_CONTROL_ENABLE;
    if (!SetCommState(handle, &dcb)) {
        CloseHandle(handle);
        return false;
    }

    COMMTIMEOUTS timeouts{};
    timeouts.ReadIntervalTimeout = 50;
    timeouts.ReadTotalTimeoutConstant = static_cast<DWORD>(config.timeout_ms);
    timeouts.ReadTotalTimeoutMultiplier = 10;
    timeouts.WriteTotalTimeoutConstant = static_cast<DWORD>(config.timeout_ms);
    timeouts.WriteTotalTimeoutMultiplier = 10;
    if (!SetCommTimeouts(handle, &timeouts)) {
        CloseHandle(handle);
        return false;
    }

    mHandle = handle;
    mRunning.store(true, std::memory_order_relaxed);
    mThread = std::thread(&SlideSerialClient::readerLoop, this);
    return true;
#else
    (void)config;
    return false;
#endif
}

void SlideSerialClient::Close() {
    mRunning.store(false, std::memory_order_relaxed);
    if (mThread.joinable()) mThread.join();

#ifdef _WIN32
    if (mHandle) CloseHandle(asHandle(mHandle));
#endif
    mHandle = nullptr;

    std::lock_guard<std::mutex> lock(mMutex);
    mLines.clear();
    mBuffer.clear();
}

bool SlideSerialClient::WriteLine(const std::string& line) {
#ifdef _WIN32
    if (!IsOpen()) return false;

    std::string payload = line;
    if (payload.empty() || (payload.back() != '\n' && payload.back() != '\r')) {
        payload.push_back('\n');
    }

    DWORD written = 0;
    const BOOL ok = WriteFile(
        asHandle(mHandle),
        payload.data(),
        static_cast<DWORD>(payload.size()),
        &written,
        nullptr);
    return ok != FALSE && written == payload.size();
#else
    (void)line;
    return false;
#endif
}

std::vector<std::string> SlideSerialClient::DrainLines() {
    std::vector<std::string> lines;
    std::lock_guard<std::mutex> lock(mMutex);
    lines.reserve(mLines.size());
    while (!mLines.empty()) {
        lines.emplace_back(std::move(mLines.front()));
        mLines.pop_front();
    }
    return lines;
}

void SlideSerialClient::readerLoop() {
#ifdef _WIN32
    constexpr std::size_t kMaxLineLength = 63;
    char buffer[128];

    while (mRunning.load(std::memory_order_relaxed) && IsOpen()) {
        DWORD bytes_read = 0;
        const BOOL ok = ReadFile(
            asHandle(mHandle), buffer, sizeof(buffer), &bytes_read, nullptr);
        if (!ok) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        if (bytes_read == 0) continue;

        for (DWORD i = 0; i < bytes_read; ++i) {
            const char character = buffer[i];
            if (character == '\r' || character == '\n') {
                if (!mBuffer.empty()) {
                    std::lock_guard<std::mutex> lock(mMutex);
                    mLines.push_back(std::move(mBuffer));
                    mBuffer.clear();
                }
                continue;
            }

            if (mBuffer.size() >= kMaxLineLength) {
                mBuffer.clear();
                continue;
            }
            mBuffer.push_back(character);
        }
    }
#endif
}

std::vector<std::string> SlideSerialClient::ListPorts() {
    std::vector<std::string> ports;
#ifdef _WIN32
    for (int index = 1; index <= 64; ++index) {
        const std::string name = "COM" + std::to_string(index);
        const std::string normalized = normalizePortName(name);
        HANDLE handle = CreateFileA(
            normalized.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            0,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            nullptr);
        if (handle != INVALID_HANDLE_VALUE) {
            ports.push_back(name);
            CloseHandle(handle);
        }
    }
#endif
    return ports;
}

} // namespace XL
