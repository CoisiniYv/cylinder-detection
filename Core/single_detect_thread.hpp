#pragma once


#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include "Server.hpp"
#include "image_types.hpp"
#include "Detect/Slice.h"
#include "Detect/trtyolo_slice.hpp"
#include "Detect/crop_image.h"
#include "Detect/sam.h"
#include "Detect/HalconProcessor.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace XL {

	// 单张图片检测专用线程
	class SingleDetectThread {
	public:
		SingleDetectThread();
		~SingleDetectThread();

		void start();
		void stop();
		void join();
		bool isRunning() const { return mRunning.load(); }

		// 提交一个单图检测任务，并阻塞等待结果返回
		// 返回是否成功；成功时 out_result 填充，失败时 err_msg 填充
		bool submitAndWait(const DetectParams& params, SingleImageResult& out_result, std::string& err_msg);

	private:
		struct Task {
			DetectParams params;
			SingleImageResult result;
			bool ok = false;
			bool done = false;
			std::string err;
			std::mutex mtx;
			std::condition_variable cv;
		};

		void worker();

	private:
		std::thread mThread;
		std::atomic<bool> mRunning{ false };
		std::mutex mQueueMutex;
		std::condition_variable mQueueCv;
		std::deque<std::shared_ptr<Task>> mTasks;

		// 线程内资源（模型等）
		std::unique_ptr<trtyolo::SliceDetector> mSliceDetector;
		bool mModelReady = false;
		std::string mLastTrtEngineFile;
		bool mLastSwapRB = false;

		std::unique_ptr<SamSegmenter> mSam;
		bool mSamReady = false;
		std::string mLastEncoderPath;
		std::string mLastDecoderPath;

		HalconProcessor mHalcon;
	};

} // namespace XL