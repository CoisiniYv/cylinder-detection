#pragma once

#include "image_types.hpp"
#include "queue_manager.hpp"
#include "Utils/Common.hpp"
#include "Utils/Log.hpp"
#include "MPHdc_API.h"

#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <future>

namespace XL {

	class CameraThread {
	public:
		CameraThread();
		~CameraThread();

		// 启动采集线程；delay_ms 控制每张间隔（毫秒）；device_id 作为标识写入 meta
		bool start(int delay_ms = 0, const std::string& device_id = "");
		void stop();
		void join();
		bool isRunning() const noexcept { return mRunning.load(std::memory_order_relaxed); }

		// 允许开始下一组采集（由 group_manager / plc_thread 调用）
		void allow_next_group();

		void setDelayMs(int ms) { mDelayMs = ms; }
		int getDelayMs() const noexcept { return mDelayMs; }

		// 单张图像采集接口
		std::future<ImageFrame> captureSingleImage(int timeout_ms = 10000);

	private:
		bool initDevice();
		void cleanupDevice();
		void captureLoop();

		// 单张采集实现
		ImageFrame captureSingleImageInternal(int timeout_ms);

		// BGR 通道合成（通道数据为 8bit 单通道，尺寸 h*w）
		static cv::Mat mergeBGR(unsigned char* b, unsigned char* g, unsigned char* r, int h, int w);

	private:
		std::atomic<bool> mRunning{ false };
		std::atomic<bool> mStopRequested{ false };
		std::thread mThread;

		int mDelayMs = 0;
		std::string mDeviceId;

		// 设备 SDK 句柄与设置
		mphdc::IMPHdc* mDevice = nullptr;
		mphdc::BasicSettingsStructType mBasicSettings{};

		// 组与帧序号
		std::uint64_t mGroupSeq = 1;
		std::uint64_t mFrameSeq = 1;

		// 组启动控制
		std::condition_variable mAllowCv;
		std::mutex mAllowMtx;
		bool mAllowNextGroupFlag = true; // 首组可直接开始

		// 单张采集控制
		std::mutex mSingleCaptureMtx;
		std::condition_variable mSingleCaptureCv;
		bool mSingleCaptureRequested = false;
		std::promise<ImageFrame> mSingleCapturePromise;
	};

} // namespace XL