#include "camera_thread.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <clocale>


namespace XL {


    CameraThread::CameraThread() {}

    CameraThread::~CameraThread() {
        stop();
        join();
        cleanupDevice();
    }

    bool CameraThread::start(int delay_ms, const std::string& device_id) {
        if (isRunning()) return true;
        mDelayMs = delay_ms;
        mDeviceId = device_id;

        if (!initDevice()) {
            LOGE("init Fall");
            return false;
        }

        g_queue_manager.start();

        mStopRequested.store(false, std::memory_order_relaxed);
        mRunning.store(true, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(mAllowMtx);
            // 启动时默认允许首组
            mAllowNextGroupFlag = true;
        }
        mThread = std::thread([this]() { this->captureLoop(); });
        LOGI("CameraThread 已启动，delay_ms=%d", mDelayMs);
        return true;
    }

    void CameraThread::stop() {
        mStopRequested.store(true, std::memory_order_relaxed);
        mAllowCv.notify_all();
    }

    void CameraThread::join() {
        if (mThread.joinable()) {
            mThread.join();
        }
        mRunning.store(false, std::memory_order_relaxed);
        g_queue_manager.stop();
    }

    void CameraThread::allow_next_group() {
        {
            std::lock_guard<std::mutex> lk(mAllowMtx);
            mAllowNextGroupFlag = true;
        }
        mAllowCv.notify_one();
        LOGI("Allow next snap");
    }

    bool CameraThread::initDevice() {
        setlocale(LC_ALL, "");

        // 创建设备实例
        mDevice = mphdc::MPHdc_Factory::GetInstance(mphdc::LogMediaType::CallBack);
        if (!mDevice) {
            LOGE("GetInstance返回空");
            return false;
        }

        // 更新设备列表并打开第一个设备
        mDevice->UpdateDeviceList();
        int TotalDeviceCnt = mDevice->GetDeviceCount();
        if (TotalDeviceCnt <= 0) {
            LOGE("未发现可用设备");
            return false;
        }

        bool rtv = mDevice->Open(mDevice->GetDeviceInfo(0));
        if (!rtv) {
            LOGE("打开设备失败");
            mphdc::MPHdc_Factory::DestructInstance(mDevice);
            mDevice = nullptr;
            return false;
        }

        LOGI("设备打开成功，设备数=%d", TotalDeviceCnt);

        // 读取并设置基本参数：SoftTriggerOnly，关闭 Hold
        mDevice->GetBasicSettings(&mBasicSettings);
        LOGI("当前相机工作模式：%s", mphdc::EnumUtils::GetEnumName(mBasicSettings.WorkingMode.Mode));

        mBasicSettings.HoldState = 0; // 关闭 hold 状态
        mBasicSettings.TriggerSource = mphdc::TriggerSourceType::SoftTriggerOnly; // 软件触发
        mDevice->SetBasicSettings(mBasicSettings);

        // 设置写入后等待一小段时间
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        return true;
    }

    void CameraThread::cleanupDevice() {
        if (!mDevice) return;
        // 打开 hold 状态
        mBasicSettings.HoldState = 1;
        mDevice->SetBasicSettings(mBasicSettings);

        // 关闭并释放实例（注意不能直接 delete）
        mDevice->Close();
        mphdc::MPHdc_Factory::DestructInstance(mDevice);
        mDevice = nullptr;
    }

    // cv::Mat CameraThread::mergeBGR(unsigned char* b, unsigned char* g, unsigned char* r, int h, int w) {
    //     if (!b || !g || !r || h <= 0 || w <= 0) return {};
    //     // 将外部内存拷贝为 Mat
    //     cv::Mat mb(h, w, CV_8UC1, b);
    //     cv::Mat mg(h, w, CV_8UC1, g);
    //     cv::Mat mr(h, w, CV_8UC1, r);
    //     std::vector<cv::Mat> ch{ mb.clone(), mg.clone(), mr.clone() };
    //     cv::Mat out;
    //     cv::merge(ch, out); // B,G,R 合并为 CV_8UC3
    //     return out;
    // }
    cv::Mat CameraThread::mergeBGR(unsigned char* ch0,
        unsigned char* ch1,
        unsigned char* ch2,
        int h, int w)
    {
        cv::Mat rgb(h, w, CV_8UC3);
        for (int y = 0; y < h; ++y) {
            cv::Vec3b* row = rgb.ptr<cv::Vec3b>(y);
            for (int x = 0; x < w; ++x) {
                row[x][0] = ch0[y * w + x];   // B
                row[x][1] = ch1[y * w + x];   // G
                row[x][2] = ch2[y * w + x];   // R
            }
        }
        return rgb;
    }
    void CameraThread::captureLoop() {
        LOGI("CameraThread 采集循环开始");

        while (!mStopRequested.load(std::memory_order_relaxed)) {
            // 等待允许开启新的一组
            {
                std::unique_lock<std::mutex> lk(mAllowMtx);
                mAllowCv.wait(lk, [this]() {
                    return mAllowNextGroupFlag || mStopRequested.load(std::memory_order_relaxed);
                    });
                if (mStopRequested.load(std::memory_order_relaxed)) break;
                // 消耗许可，防止下一次循环直接开始下一组
                mAllowNextGroupFlag = false;
            }

            const std::uint64_t group_id = mGroupSeq++;
            LOGI("开始采集新组 group=%llu", group_id);

            for (int idx = 0; idx < static_cast<int>(kQuadImageCount); /* idx++ 在成功后 */) {
                if (mStopRequested.load(std::memory_order_relaxed)) break;

                mphdc::DataFormatType format;
                mphdc::DataFrameUndefinedStruct data;
                bool ok = mDevice && mDevice->Snap(true, &format, &data, 10000);
                if (!ok) {
                    LOGE("Snap 失败，重试中");
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue; // 本张重试
                }

                // 解码 6 通道为两张 BGR 图（012 -> ps，345 -> rgb）
                mphdc::DataFrame2DStruct* data2D = reinterpret_cast<mphdc::DataFrame2DStruct*>(&data);
                const int height = data2D->Height();
                const int width = data2D->Width();
                std::array<unsigned char*, 6> chPtr{};
                for (int c = 0; c < data2D->Channel() && c < 6; ++c) {
                    unsigned char* ptr = nullptr;
                    int size = data2D->GetRawChannelData(c, &ptr);
                    if (ptr && size == height * width) chPtr[c] = ptr;
                }

                if (!(chPtr[2] && chPtr[1] && chPtr[0] && chPtr[5] && chPtr[4] && chPtr[3])) {
                    LOGE("通道数据不完整，重试当前张");
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }

                cv::Mat ps = mergeBGR(chPtr[2], chPtr[1], chPtr[0], height, width);
                cv::Mat rgb = mergeBGR(chPtr[5], chPtr[4], chPtr[3], height, width);
                if (ps.empty() || rgb.empty()) {
                    LOGE("OpenCV 合成失败，重试当前张");
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }

                ImageFrame frame;
                frame.ps_image = std::move(ps);
                frame.rgb_image = std::move(rgb);
                frame.meta.timestamp_ms = XL::getCurTimestamp();
                frame.meta.device_id = mDeviceId;
                frame.meta.group_id = group_id;
                frame.meta.index_in_group = idx;
                frame.meta.sequence_id = mFrameSeq++;

                if (!g_queue_manager.pushRaw(std::move(frame))) {
                    LOGE("pushRaw 失败（队列可能已停止）");
                    break;
                }

                LOGI("已入队：group=%llu, idx=%d", group_id, idx);

                ++idx; // 成功后才递增到下一张
                if (mDelayMs > 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(mDelayMs));
                }
            }

            LOGI("组 %llu 采集完成，等待允许下一组", group_id);
            // 下一轮 while 会再次等待 allow_next_group()
        }

        LOGI("CameraThread 采集循环结束");
    }

} // namespace XL