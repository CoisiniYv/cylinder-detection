

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <clocale>
#include "camera_thread.hpp"

namespace XL {


	CameraThread::CameraThread() {}

	CameraThread::~CameraThread() {
		stop();
		join();
		cleanupDevice();
	}

	bool CameraThread::start(int delay_ms,
		const std::string& device_id,
		const std::string& slide_port,
		int slide_axis_id,
		int slide_timeout_ms,
		bool enable_group_capture) {
		if (isRunning()) return true;
		mDelayMs = delay_ms;
		mDeviceId = device_id;
		mSlideCfg = SerialConfig{};
		mSlideCfg.port = slide_port;
		mSlideAxisId = slide_axis_id;
		mSlideTimeoutMs = slide_timeout_ms > 0 ? slide_timeout_ms : 20000;
		mEnableGroupCapture = enable_group_capture;
		if (!initDevice()) {
			LOGE("init Fall");
			return false;
		}
		if (!mSlideCfg.port.empty()) {
			if (!openSlideSerial()) {
				LOGE("滑台串口 %s 打开失败", mSlideCfg.port.c_str());
			}
			else {
				LOGI("滑台串口 %s 已连接", mSlideCfg.port.c_str());
			}
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
		LOGI("CameraThread已启动，delay_ms=%d", mDelayMs);
		return true;
	}

	void CameraThread::stop() {
		mStopRequested.store(true, std::memory_order_relaxed);
		mAllowCv.notify_all();
		mSingleCaptureCv.notify_all();
	}

	void CameraThread::join() {
		if (mThread.joinable()) {
			mThread.join();
		}
		mRunning.store(false, std::memory_order_relaxed);
		g_queue_manager.stop();
		closeSlideSerial();
		cleanupDevice();
	}

	void CameraThread::allow_next_group() {
		{
			std::lock_guard<std::mutex> lk(mAllowMtx);
			mAllowNextGroupFlag = true;
		}
		mAllowCv.notify_one();
		LOGI("Allow next snap");
	}

	// ---------------- 滑台串口实现 ----------------

	bool CameraThread::openSlideSerial() {
		if (mSlideCfg.port.empty()) return false;
		return mSlide.Open(mSlideCfg);
	}

	void CameraThread::closeSlideSerial() {
		mSlide.Close();
	}

	bool CameraThread::lineHasToken(const std::string& line, const std::string& token) {
		if (token.empty()) return false;
		return line.find(token) != std::string::npos;
	}

	void CameraThread::drainSlideLines() {
		if (!mSlide.IsOpen()) return;
		(void)mSlide.DrainLines();
	}

	bool CameraThread::sendSlideCommand(int dir, int steps, int dly) {
		if (!mSlide.IsOpen()) return false;
		std::string line = std::to_string(mSlideAxisId) + "," +
			std::to_string(dir) + "," +
			std::to_string(steps) + "," +
			std::to_string(dly);
		return mSlide.WriteLine(line);
	}

	bool CameraThread::waitSlideResponseAny(const std::vector<std::string>& tokens, int timeout_ms) {
		if (!mSlide.IsOpen()) return false;
		const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
		while (!mStopRequested.load(std::memory_order_relaxed)) {
			for (auto& line : mSlide.DrainLines()) {
				if (lineHasToken(line, "Parse Err") || lineHasToken(line, "Bad Cmd") || lineHasToken(line, "Flow Busy")) {
					LOGE("滑台返回错误: %s", line.c_str());
					return false;
				}
				for (const auto& t : tokens) {
					if (lineHasToken(line, t)) {
						return true;
					}
				}
			}
			if (std::chrono::steady_clock::now() >= deadline) {
				return false;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
		return false;
	}

	bool CameraThread::waitSlideResponse(const std::string& token, int timeout_ms) {
		return waitSlideResponseAny({ token }, timeout_ms);
	}

	// 单张采集接口实现
	std::future<ImageFrame> CameraThread::captureSingleImage(int timeout_ms) {
		std::lock_guard<std::mutex> lk(mSingleCaptureMtx);

		// 检查是否已有进行中的单张采集
		if (mSingleCaptureRequested) {
			std::promise<ImageFrame> failed_promise;
			failed_promise.set_exception(std::make_exception_ptr(
				std::runtime_error("Single capture already in progress")));
			return failed_promise.get_future();
		}

		mSingleCaptureRequested = true;
		mSingleCapturePromise = std::promise<ImageFrame>();

		// 通知采集线程执行单张采集
		mSingleCaptureCv.notify_one();

		return mSingleCapturePromise.get_future();
	}

	ImageFrame CameraThread::captureSingleImageInternal(int timeout_ms) {
		if (!mDevice) {
			throw std::runtime_error("Camera device not initialized");
		}

		mphdc::DataFormatType format;
		mphdc::DataFrameUndefinedStruct data;
		bool ok = mDevice->Snap(true, &format, &data, timeout_ms);
		if (!ok) {
			throw std::runtime_error("Snap failed within timeout");
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
			throw std::runtime_error("Incomplete channel data");
		}

		cv::Mat ps = mergeBGR(chPtr[2], chPtr[1], chPtr[0], height, width);
		cv::Mat rgb = mergeBGR(chPtr[5], chPtr[4], chPtr[3], height, width);
		if (ps.empty() || rgb.empty()) {
			throw std::runtime_error("OpenCV merge failed");
		}

		ImageFrame frame;
		frame.ps_image = std::move(ps);
		frame.rgb_image = std::move(rgb);
		frame.meta.device_id = mDeviceId;
		frame.meta.group_id = 0; // 单张采集组ID设为0
		frame.meta.index_in_group = -1; // 索引设为-1

		return frame;
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
		LOGI("CameraThread采集循环开始");

		while (!mStopRequested.load(std::memory_order_relaxed)) {
			// 单张采集检查逻辑
			{
				std::unique_lock<std::mutex> lk(mSingleCaptureMtx);
				if (mSingleCaptureCv.wait_for(lk, std::chrono::milliseconds(100),
					[this]() { return mSingleCaptureRequested || mStopRequested; })) {

					if (mStopRequested.load(std::memory_order_relaxed)) break;

					if (mSingleCaptureRequested) {
						try {
							ImageFrame frame = captureSingleImageInternal(10000);
							mSingleCapturePromise.set_value(std::move(frame));
							LOGI("单张采集完成");
						}
						catch (const std::exception& e) {
							mSingleCapturePromise.set_exception(std::current_exception());
							LOGE("单张采集失败: %s", e.what());
						}
						mSingleCaptureRequested = false;
						continue;
					}
				}
			}

			if (!mEnableGroupCapture) {
				continue;
			}

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

			const bool slideEnabled = mSlide.IsOpen();
			if (slideEnabled) {
				drainSlideLines();
				// 1) LOAD
				if (!sendSlideCommand(20, 0, 0) || !waitSlideResponse("FLOW LOAD DONE", mSlideTimeoutMs)) {
					LOGE("滑台 LOAD 流程失败");
					allow_next_group();
					std::this_thread::sleep_for(std::chrono::milliseconds(200));
					continue;
				}
				// 2) GOTO_CAM
				if (!sendSlideCommand(21, 0, 0) || !waitSlideResponse("FLOW GOTO_CAM DONE", mSlideTimeoutMs)) {
					LOGE("滑台 GOTO_CAM 流程失败");
					allow_next_group();
					std::this_thread::sleep_for(std::chrono::milliseconds(200));
					continue;
				}
			}

			const std::uint64_t group_id = mGroupSeq++;
			LOGI("开始采集新组group=%llu", group_id);

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
				frame.meta.device_id = mDeviceId;
				frame.meta.group_id = group_id;
				frame.meta.index_in_group = idx;

				//if (!g_queue_manager.pushRaw(std::move(frame))) {
				//	LOGE("pushRaw 失败（队列可能已停止）");
				//	break;
				//}

				//LOGI("已入队：group=%llu, idx=%d", group_id, idx);

				//++idx; // 成功后才递增到下一张
				if (g_queue_manager.pushRaw(std::move(frame))) {
					LOGI("已入队: group=%llu, idx=%d", group_id, idx);

					// 四面采集：拍一张 -> CAM_NEXT -> 拍下一张
					if (slideEnabled) {
						if (idx < static_cast<int>(kQuadImageCount) - 1) {
							if (!sendSlideCommand(23, 0, 0) ||
								!waitSlideResponseAny({ "Cam Next", "POS_DONE" }, mSlideTimeoutMs)) {
								LOGE("滑台 CAM_NEXT 失败，group=%llu, idx=%d", group_id, idx);
								break;
							}
						}
						else {
							// 最后一面完成后 UNLOAD
							if (!sendSlideCommand(22, 0, 0) ||
								!waitSlideResponse("FLOW UNLOAD DONE", mSlideTimeoutMs)) {
								LOGE("滑台 UNLOAD 失败，group=%llu", group_id);
							}
						}
					}
					++idx;
				}
				if (mDelayMs > 0) {
					std::this_thread::sleep_for(std::chrono::milliseconds(mDelayMs));
				}
			}

			LOGI("组%llu采集完成，等待允许下一组", group_id);
			// 下一轮 while 会再次等待 allow_next_group()
		}

		LOGI("CameraThread采集循环结束");
	}

} // namespace XL
