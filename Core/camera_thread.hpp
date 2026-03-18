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

// --- Windows 宏冲突解决策略 ---
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN // 排除不常用的 Windows 头文件
#endif
#ifndef NOMINMAX
#define NOMINMAX // 防止 windows.h 定义 min 和 max 宏，避免与 std::min/max 冲突
#endif
#include <windows.h>
#endif

namespace XL {

	class CameraThread {
	public:
		CameraThread();
		~CameraThread();

		/**
		 * 启动采集线程
		 * @param delay_ms: 每张图拍摄后的额外等待延时
		 * @param device_id: 相机标识
		 * @param com_port: 串口号，例如 "COM3"，为空则不启用串口
		 */
		bool start(int delay_ms = 0, const std::string& device_id = "", const std::string& com_port = "");

		void stop();
		void join();
		bool isRunning() const noexcept { return mRunning.load(std::memory_order_relaxed); }

		// 允许开始下一组采集（由外部逻辑触发）
		void allow_next_group();

		void setDelayMs(int ms) { mDelayMs = ms; }
		int getDelayMs() const noexcept { return mDelayMs; }

		// 单张图像采集接口
		std::future<ImageFrame> captureSingleImage(int timeout_ms = 10000);

	private:
		// 相机 SDK 相关
		bool initDevice();
		void cleanupDevice();
		void captureLoop();
		ImageFrame captureSingleImageInternal(int timeout_ms);

		// BGR 通道合成
		static cv::Mat mergeBGR(unsigned char* b, unsigned char* g, unsigned char* r, int h, int w);

		// 串口通信私有方法
		bool openSerial(const std::string& portName);
		void closeSerial();
		bool sendSerialCommand(const std::string& cmd);
		std::string readSerialResponse();

	private:
		std::atomic<bool> mRunning{ false };
		std::atomic<bool> mStopRequested{ false };
		std::thread mThread;

		int mDelayMs = 0;
		std::string mDeviceId;

		// 相机 SDK 实例与设置
		mphdc::IMPHdc* mDevice = nullptr;
		mphdc::BasicSettingsStructType mBasicSettings{};

		// 采集序列号
		std::uint64_t mGroupSeq = 1;

		// 组逻辑控制
		std::condition_variable mAllowCv;
		std::mutex mAllowMtx;
		bool mAllowNextGroupFlag = true;

		// 单张采集控制
		std::mutex mSingleCaptureMtx;
		std::condition_variable mSingleCaptureCv;
		bool mSingleCaptureRequested = false;
		std::promise<ImageFrame> mSingleCapturePromise;

		// 串口相关成员
#ifdef _WIN32
		HANDLE mSerialHandle = (HANDLE)(uintptr_t)-1; // 对应 INVALID_HANDLE_VALUE
#else
		void* mSerialHandle = nullptr;
#endif
		std::string mComPort;
	};

} // namespace XL