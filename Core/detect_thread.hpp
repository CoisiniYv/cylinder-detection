#pragma once


#include "image_types.hpp"
#include "Detect/crop_image.h"
#include "Detect/trtyolo_slice.hpp"
#include "Detect/HalconProcessor.h"
#include "Detect/Slice.h"
#include "Detect/sam.h"

#include <thread>
#include <atomic>
#include <string>
#include <memory>
#include <vector>
#include <filesystem>
#include "Server.hpp"
namespace XL {

	// 检测线程
	class DetectThread {
	public:
		// thread_index：用于日志/输出命名；params：检测参数（由 Server 下发）
		explicit DetectThread(int thread_index, const DetectParams& params);
		~DetectThread();

		void start();        // 启动线程（内部会预加载模型）
		void stop();         // 请求停止（优雅退出），不会影响全局队列
		void join();         // 等待线程退出
		bool isRunning() const noexcept { return mRunning.load(std::memory_order_relaxed); }

		int index() const noexcept { return mIndex; }

	private:
		void run();          // 线程主体：阻塞式从队列取图并处理

	private:
		int mIndex = 0;
		std::thread mThread;
		std::atomic<bool> mRunning{ false };

		// 配置与模型
		DetectParams mParams;                                   // 可配置参数集合
		std::unique_ptr<trtyolo::SliceDetector> mSliceDetector; // 每个线程独立模型实例
		HalconProcessor mHalcon;                                // Halcon 处理器（线程内复用）
		bool mModelReady = false;

		// SAM 分割
		std::unique_ptr<SamSegmenter> mSam;                     // SAM 分割器
		bool mSamReady = false;

		// 选择输入图像类型
		bool mPreferPsImage = true;
	};

} // namespace XL