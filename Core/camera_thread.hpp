#pragma once

#include "image_types.hpp"
#include "slide_serial.hpp"

#include <MPHdc_API.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace XL {

// Camera acquisition owner.
//
// Responsibilities: camera SDK lifecycle, frame acquisition and slide protocol
// sequencing. Queue lifecycle belongs to Scheduler/QueueManager, not this class.
class CameraThread {
public:
    CameraThread();
    ~CameraThread();

    CameraThread(const CameraThread&) = delete;
    CameraThread& operator=(const CameraThread&) = delete;

    bool start(
        int delay_ms = 0,
        const std::string& device_id = {},
        const std::string& slide_port = {},
        int slide_axis_id = 0,
        int slide_timeout_ms = 20000,
        bool enable_group_capture = true);

    void stop();
    void join();
    bool isRunning() const noexcept { return mRunning.load(std::memory_order_relaxed); }

    void allow_next_group();
    void setDelayMs(int delay_ms) { mDelayMs = delay_ms; }
    int getDelayMs() const noexcept { return mDelayMs; }

    std::future<ImageFrame> captureSingleImage(int timeout_ms = 10000);

private:
    bool initDevice();
    void cleanupDevice();
    void captureLoop();
    ImageFrame captureSingleImageInternal(int timeout_ms);

    static cv::Mat mergeBGR(
        unsigned char* blue,
        unsigned char* green,
        unsigned char* red,
        int height,
        int width);

    bool openSlideSerial();
    void closeSlideSerial();
    bool sendSlideCommand(int direction, int steps = 0, int delay = 0);
    bool waitSlideResponseAny(const std::vector<std::string>& tokens, int timeout_ms);
    bool waitSlideResponse(const std::string& token, int timeout_ms);
    void drainSlideLines();
    static bool lineHasToken(const std::string& line, const std::string& token);

private:
    std::atomic<bool> mRunning{false};
    std::atomic<bool> mStopRequested{false};
    std::thread mThread;

    int mDelayMs = 0;
    std::string mDeviceId;

    mphdc::IMPHdc* mDevice = nullptr; // SDK factory owns allocation/deallocation protocol
    mphdc::BasicSettingsStructType mBasicSettings{};
    std::uint64_t mGroupSequence = 1;

    std::condition_variable mAllowCondition;
    std::mutex mAllowMutex;
    bool mAllowNextGroup = true;

    std::mutex mSingleCaptureMutex;
    std::condition_variable mSingleCaptureCondition;
    bool mSingleCaptureRequested = false;
    int mSingleCaptureTimeoutMs = 10000;
    std::promise<ImageFrame> mSingleCapturePromise;

    SlideSerialClient mSlide;
    SerialConfig mSlideConfig{};
    int mSlideAxisId = 0;
    int mSlideTimeoutMs = 20000;
    bool mEnableGroupCapture = true;
};

} // namespace XL
