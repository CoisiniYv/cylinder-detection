#include "camera_thread.hpp"

#include "Utils/Log.hpp"
#include "queue_manager.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <chrono>
#include <clocale>
#include <stdexcept>

namespace XL {
namespace {

constexpr int kSlideLoad = 20;
constexpr int kSlideGotoCamera = 21;
constexpr int kSlideUnload = 22;
constexpr int kSlideCameraNext = 23;
constexpr int kCameraSnapTimeoutMs = 10000;

} // namespace

CameraThread::CameraThread() = default;

CameraThread::~CameraThread() {
    stop();
    join();
}

bool CameraThread::start(
    int delay_ms,
    const std::string& device_id,
    const std::string& slide_port,
    int slide_axis_id,
    int slide_timeout_ms,
    bool enable_group_capture) {
    if (isRunning()) return true;

    mDelayMs = std::max(0, delay_ms);
    mDeviceId = device_id;
    mSlideConfig = SerialConfig{};
    mSlideConfig.port = slide_port;
    mSlideAxisId = slide_axis_id;
    mSlideTimeoutMs = slide_timeout_ms > 0 ? slide_timeout_ms : 20000;
    mEnableGroupCapture = enable_group_capture;

    if (!initDevice()) {
        LOGE("camera initialization failed");
        return false;
    }

    if (!mSlideConfig.port.empty()) {
        if (!openSlideSerial()) {
            LOGE("failed to open slide serial port: %s", mSlideConfig.port.c_str());
        }
        else {
            LOGI("slide serial connected: %s", mSlideConfig.port.c_str());
        }
    }

    mStopRequested.store(false, std::memory_order_relaxed);
    mRunning.store(true, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(mAllowMutex);
        mAllowNextGroup = true;
    }
    mThread = std::thread(&CameraThread::captureLoop, this);
    LOGI("CameraThread started: delay_ms=%d group_capture=%d", mDelayMs, mEnableGroupCapture ? 1 : 0);
    return true;
}

void CameraThread::stop() {
    mStopRequested.store(true, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lock(mSingleCaptureMutex);
        if (mSingleCaptureRequested) {
            try {
                throw std::runtime_error("camera stopped before single capture completed");
            }
            catch (...) {
                mSingleCapturePromise.set_exception(std::current_exception());
            }
            mSingleCaptureRequested = false;
        }
    }

    mAllowCondition.notify_all();
    mSingleCaptureCondition.notify_all();
}

void CameraThread::join() {
    if (mThread.joinable()) mThread.join();
    mRunning.store(false, std::memory_order_relaxed);
    closeSlideSerial();
    cleanupDevice();
}

void CameraThread::allow_next_group() {
    {
        std::lock_guard<std::mutex> lock(mAllowMutex);
        mAllowNextGroup = true;
    }
    mAllowCondition.notify_one();
}

bool CameraThread::openSlideSerial() {
    if (mSlideConfig.port.empty()) return false;
    return mSlide.Open(mSlideConfig);
}

void CameraThread::closeSlideSerial() {
    mSlide.Close();
}

bool CameraThread::lineHasToken(const std::string& line, const std::string& token) {
    return !token.empty() && line.find(token) != std::string::npos;
}

void CameraThread::drainSlideLines() {
    if (mSlide.IsOpen()) (void)mSlide.DrainLines();
}

bool CameraThread::sendSlideCommand(int direction, int steps, int delay) {
    if (!mSlide.IsOpen()) return false;
    return mSlide.WriteLine(
        std::to_string(mSlideAxisId) + "," +
        std::to_string(direction) + "," +
        std::to_string(steps) + "," +
        std::to_string(delay));
}

bool CameraThread::waitSlideResponseAny(
    const std::vector<std::string>& tokens,
    int timeout_ms) {
    if (!mSlide.IsOpen() || timeout_ms <= 0) return false;

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!mStopRequested.load(std::memory_order_relaxed)) {
        for (const auto& line : mSlide.DrainLines()) {
            if (lineHasToken(line, "Parse Err") ||
                lineHasToken(line, "Bad Cmd") ||
                lineHasToken(line, "Flow Busy")) {
                LOGE("slide controller error: %s", line.c_str());
                return false;
            }
            for (const auto& token : tokens) {
                if (lineHasToken(line, token)) return true;
            }
        }

        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

bool CameraThread::waitSlideResponse(const std::string& token, int timeout_ms) {
    return waitSlideResponseAny({token}, timeout_ms);
}

std::future<ImageFrame> CameraThread::captureSingleImage(int timeout_ms) {
    std::lock_guard<std::mutex> lock(mSingleCaptureMutex);
    if (!isRunning() || mStopRequested.load(std::memory_order_relaxed)) {
        std::promise<ImageFrame> failed;
        auto future = failed.get_future();
        failed.set_exception(std::make_exception_ptr(
            std::runtime_error("camera worker is not running")));
        return future;
    }
    if (mSingleCaptureRequested) {
        std::promise<ImageFrame> failed;
        auto future = failed.get_future();
        failed.set_exception(std::make_exception_ptr(
            std::runtime_error("single capture is already in progress")));
        return future;
    }

    mSingleCaptureRequested = true;
    mSingleCaptureTimeoutMs = timeout_ms > 0 ? timeout_ms : kCameraSnapTimeoutMs;
    mSingleCapturePromise = std::promise<ImageFrame>();
    auto future = mSingleCapturePromise.get_future();
    mSingleCaptureCondition.notify_one();
    return future;
}

ImageFrame CameraThread::captureSingleImageInternal(int timeout_ms) {
    if (!mDevice) throw std::runtime_error("camera device is not initialized");

    mphdc::DataFormatType format;
    mphdc::DataFrameUndefinedStruct data;
    if (!mDevice->Snap(true, &format, &data, timeout_ms)) {
        throw std::runtime_error("camera Snap timed out or failed");
    }

    auto* data_2d = reinterpret_cast<mphdc::DataFrame2DStruct*>(&data);
    const int height = data_2d->Height();
    const int width = data_2d->Width();
    std::array<unsigned char*, 6> channels{};
    for (int index = 0; index < data_2d->Channel() && index < static_cast<int>(channels.size()); ++index) {
        unsigned char* pointer = nullptr;
        const int size = data_2d->GetRawChannelData(index, &pointer);
        if (pointer && size == height * width) channels[static_cast<std::size_t>(index)] = pointer;
    }

    for (const auto* channel : channels) {
        if (!channel) throw std::runtime_error("camera returned incomplete channel data");
    }

    ImageFrame frame;
    frame.ps_image = mergeBGR(channels[2], channels[1], channels[0], height, width);
    frame.rgb_image = mergeBGR(channels[5], channels[4], channels[3], height, width);
    if (frame.ps_image.empty() || frame.rgb_image.empty()) {
        throw std::runtime_error("failed to compose camera BGR images");
    }
    frame.meta.device_id = mDeviceId;
    return frame;
}

bool CameraThread::initDevice() {
    std::setlocale(LC_ALL, "");
    mDevice = mphdc::MPHdc_Factory::GetInstance(mphdc::LogMediaType::CallBack);
    if (!mDevice) {
        LOGE("MPHdc factory returned null");
        return false;
    }

    mDevice->UpdateDeviceList();
    const int device_count = mDevice->GetDeviceCount();
    if (device_count <= 0) {
        LOGE("no camera device found");
        mphdc::MPHdc_Factory::DestructInstance(mDevice);
        mDevice = nullptr;
        return false;
    }

    // Current vendor integration always opens the first enumerated device.
    // Mapping a logical device_id to a physical SDK device requires hardware
    // metadata and is intentionally left as a documented follow-up.
    if (!mDevice->Open(mDevice->GetDeviceInfo(0))) {
        LOGE("failed to open camera device 0");
        mphdc::MPHdc_Factory::DestructInstance(mDevice);
        mDevice = nullptr;
        return false;
    }

    mDevice->GetBasicSettings(&mBasicSettings);
    mBasicSettings.HoldState = 0;
    mBasicSettings.TriggerSource = mphdc::TriggerSourceType::SoftTriggerOnly;
    mDevice->SetBasicSettings(mBasicSettings);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    LOGI("camera device opened; enumerated_count=%d", device_count);
    return true;
}

void CameraThread::cleanupDevice() {
    if (!mDevice) return;
    mBasicSettings.HoldState = 1;
    mDevice->SetBasicSettings(mBasicSettings);
    mDevice->Close();
    mphdc::MPHdc_Factory::DestructInstance(mDevice);
    mDevice = nullptr;
}

cv::Mat CameraThread::mergeBGR(
    unsigned char* blue,
    unsigned char* green,
    unsigned char* red,
    int height,
    int width) {
    if (!blue || !green || !red || height <= 0 || width <= 0) return {};

    cv::Mat image(height, width, CV_8UC3);
    for (int row_index = 0; row_index < height; ++row_index) {
        cv::Vec3b* row = image.ptr<cv::Vec3b>(row_index);
        for (int column = 0; column < width; ++column) {
            const int index = row_index * width + column;
            row[column][0] = blue[index];
            row[column][1] = green[index];
            row[column][2] = red[index];
        }
    }
    return image;
}

void CameraThread::captureLoop() {
    LOGI("CameraThread capture loop started");

    while (!mStopRequested.load(std::memory_order_relaxed)) {
        {
            std::unique_lock<std::mutex> lock(mSingleCaptureMutex);
            if (mSingleCaptureCondition.wait_for(
                    lock,
                    std::chrono::milliseconds(100),
                    [this] {
                        return mSingleCaptureRequested ||
                               mStopRequested.load(std::memory_order_relaxed);
                    })) {
                if (mStopRequested.load(std::memory_order_relaxed)) break;
                if (mSingleCaptureRequested) {
                    try {
                        ImageFrame frame = captureSingleImageInternal(mSingleCaptureTimeoutMs);
                        mSingleCapturePromise.set_value(std::move(frame));
                    }
                    catch (...) {
                        mSingleCapturePromise.set_exception(std::current_exception());
                    }
                    mSingleCaptureRequested = false;
                    continue;
                }
            }
        }

        if (!mEnableGroupCapture) continue;

        {
            std::unique_lock<std::mutex> lock(mAllowMutex);
            mAllowCondition.wait(lock, [this] {
                return mAllowNextGroup || mStopRequested.load(std::memory_order_relaxed);
            });
            if (mStopRequested.load(std::memory_order_relaxed)) break;
            mAllowNextGroup = false;
        }

        const bool slide_enabled = mSlide.IsOpen();
        if (slide_enabled) {
            drainSlideLines();
            if (!sendSlideCommand(kSlideLoad) ||
                !waitSlideResponse("FLOW LOAD DONE", mSlideTimeoutMs)) {
                LOGE("slide LOAD flow failed");
                allow_next_group();
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }
            if (!sendSlideCommand(kSlideGotoCamera) ||
                !waitSlideResponse("FLOW GOTO_CAM DONE", mSlideTimeoutMs)) {
                LOGE("slide GOTO_CAM flow failed");
                allow_next_group();
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }
        }

        const std::uint64_t group_id = mGroupSequence++;
        LOGI("capture group started: group=%llu", static_cast<unsigned long long>(group_id));

        for (int face_index = 0; face_index < static_cast<int>(kQuadImageCount);) {
            if (mStopRequested.load(std::memory_order_relaxed)) break;

            mphdc::DataFormatType format;
            mphdc::DataFrameUndefinedStruct data;
            if (!mDevice || !mDevice->Snap(true, &format, &data, kCameraSnapTimeoutMs)) {
                // Preserve historical retry semantics until hardware-side retry
                // policy can be validated on the production device.
                LOGE("camera Snap failed; retrying current face");
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            auto* data_2d = reinterpret_cast<mphdc::DataFrame2DStruct*>(&data);
            const int height = data_2d->Height();
            const int width = data_2d->Width();
            std::array<unsigned char*, 6> channels{};
            for (int channel_index = 0;
                 channel_index < data_2d->Channel() && channel_index < static_cast<int>(channels.size());
                 ++channel_index) {
                unsigned char* pointer = nullptr;
                const int size = data_2d->GetRawChannelData(channel_index, &pointer);
                if (pointer && size == height * width) {
                    channels[static_cast<std::size_t>(channel_index)] = pointer;
                }
            }

            bool channels_complete = true;
            for (const auto* channel : channels) channels_complete = channels_complete && channel != nullptr;
            if (!channels_complete) {
                LOGE("camera channel data incomplete; retrying current face");
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            ImageFrame frame;
            frame.ps_image = mergeBGR(channels[2], channels[1], channels[0], height, width);
            frame.rgb_image = mergeBGR(channels[5], channels[4], channels[3], height, width);
            if (frame.ps_image.empty() || frame.rgb_image.empty()) {
                LOGE("failed to compose camera image; retrying current face");
                continue;
            }
            frame.meta.device_id = mDeviceId;
            frame.meta.group_id = group_id;
            frame.meta.index_in_group = face_index;

            if (!g_queue_manager.pushRaw(std::move(frame))) {
                LOGE("raw frame queue is stopped");
                break;
            }

            if (slide_enabled) {
                if (face_index < static_cast<int>(kQuadImageCount) - 1) {
                    if (!sendSlideCommand(kSlideCameraNext) ||
                        !waitSlideResponseAny({"Cam Next", "POS_DONE"}, mSlideTimeoutMs)) {
                        LOGE("slide CAM_NEXT failed: group=%llu face=%d",
                             static_cast<unsigned long long>(group_id), face_index);
                        break;
                    }
                }
                else if (!sendSlideCommand(kSlideUnload) ||
                         !waitSlideResponse("FLOW UNLOAD DONE", mSlideTimeoutMs)) {
                    LOGE("slide UNLOAD failed: group=%llu",
                         static_cast<unsigned long long>(group_id));
                }
            }

            ++face_index;
            if (mDelayMs > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(mDelayMs));
            }
        }
    }

    LOGI("CameraThread capture loop stopped");
}

} // namespace XL
