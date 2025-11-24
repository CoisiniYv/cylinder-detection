#include "single_detect_thread.hpp"
#include "Utils/Log.hpp"

#include <cuda_runtime.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>

namespace XL {

	static bool ensure_dir(const std::filesystem::path& p) {
		std::error_code ec; if (std::filesystem::exists(p, ec)) return std::filesystem::is_directory(p, ec);
		return std::filesystem::create_directories(p, ec);
	}

	SingleDetectThread::SingleDetectThread() {
		mHalcon.setSaveCompleteCallback([this](const std::string& filePath,
			bool success,
			HalconProcessor::SaveType type,
			const std::string& message) {
				auto status = mHalcon.getAsyncStatus();
				const char* type_str = (type == HalconProcessor::SaveType::PATH_SAVE) ? "PATH_SAVE" : "UNKNOWN";

				// 打印保存结果以及当前异步状态统计
				LOGI("缺陷骨架图已保存：path=%s，success=%d，type=%s，msg=%s，pending=%d，completed=%d，failed=%d",
					filePath.c_str(),
					success ? 1 : 0,
					type_str,
					message.c_str(),
					status.pendingOperations,
					status.completedOperations,
					status.failedOperations);
			});
	}

	SingleDetectThread::~SingleDetectThread() {
		stop();
		join();
		// 清理回调
		mHalcon.clearSaveCompleteCallback();
	}

	void SingleDetectThread::start() {
		if (mRunning.load()) return;
		mRunning.store(true);
		mThread = std::thread(&SingleDetectThread::worker, this);
	}

	void SingleDetectThread::stop() {
		mRunning.store(false);
		mQueueCv.notify_all();
	}

	void SingleDetectThread::join() {
		if (mThread.joinable()) mThread.join();
	}

	bool SingleDetectThread::submitAndWait(const DetectParams& params, SingleImageResult& out_result, std::string& err_msg) {
		if (!mRunning.load()) {
			// 若未启动则自动启动
			start();
		}
		auto task = std::make_shared<Task>();
		task->params = params;

		{
			std::lock_guard<std::mutex> lk(mQueueMutex);
			mTasks.push_back(task);
		}
		mQueueCv.notify_one();

		std::unique_lock<std::mutex> ul(task->mtx);
		task->cv.wait(ul, [&] { return task->done; });
		if (task->ok) {
			out_result = std::move(task->result);
			return true;
		}
		else {
			err_msg = task->err;
			return false;
		}
	}

	void SingleDetectThread::worker() {
		try {
			cv::cuda::setDevice(0);
		}
		catch (...) {
		}
		cudaError_t __devErr = cudaSetDevice(0);
		if (__devErr != cudaSuccess) {
			LOGE("SingleDetectThread: cudaSetDevice failed: %s", cudaGetErrorString(__devErr));
			mRunning.store(false);
			return;
		}
		cv::cuda::Stream stream;

		while (mRunning.load()) {
			std::shared_ptr<Task> task;
			{
				std::unique_lock<std::mutex> lk(mQueueMutex);
				mQueueCv.wait(lk, [&] { return !mRunning.load() || !mTasks.empty(); });
				if (!mRunning.load() && mTasks.empty()) break;
				task = mTasks.front();
				mTasks.pop_front();
			}

			if (!task) continue;

			try {
				const auto& p = task->params;
				// 初始化/更新检测模型
				if (!mSliceDetector || !mModelReady || mLastTrtEngineFile != p.engine_file) {
					try {
						trtyolo::InferOption infer_opt;
						infer_opt.enableSwapRB();
						mSliceDetector = std::make_unique<trtyolo::SliceDetector>(p.engine_file, infer_opt);
						mModelReady = true;
						mLastTrtEngineFile = p.engine_file;
					}
					catch (const std::exception& e) {
						LOGE("SingleDetectThread: model init failed: %s", e.what());
						throw;
					}
				}

				// 初始化/更新SAM分割器
				{
					std::string actualEncoderPath = p.sam_encoder_engine_file;
					std::string actualDecoderPath = p.sam_decoder_engine_file;
					if (!std::filesystem::exists(actualEncoderPath))
						actualEncoderPath = p.sam_encoder_onnx_file;
					if (!std::filesystem::exists(actualDecoderPath))
						actualDecoderPath = p.sam_decoder_onnx_file;
					const std::string encoderEngine = actualEncoderPath;
					const std::string decoderEngine = actualDecoderPath;

					if (!encoderEngine.empty() && !decoderEngine.empty()) {
						if (!mSam || !mSamReady || mLastEncoderPath != encoderEngine || mLastDecoderPath != decoderEngine) {
							mSam = std::make_unique<SamSegmenter>();
							mSamReady = mSam->init(encoderEngine, decoderEngine);
							mLastEncoderPath = encoderEngine;
							mLastDecoderPath = decoderEngine;
							if (!mSamReady) {
								LOGE("SingleDetectThread: SAM init failed.");
							}
						}
					}
					else {
						mSam.reset();
						mSamReady = false;
						mLastEncoderPath.clear();
						mLastDecoderPath.clear();
					}
				}

				auto total_start = std::chrono::high_resolution_clock::now();

				// 读取输入图像
				cv::Mat src_cpu = cv::imread(p.input_image_path, cv::IMREAD_COLOR);
				if (src_cpu.empty()) {
					throw std::runtime_error(std::string("无法读取输入图片: ") + p.input_image_path);
				}
				cv::cuda::GpuMat d_input; d_input.upload(src_cpu, stream);

				const std::string savePath = (g_server_state.config ? g_server_state.config->outputdir : std::string()) + "\\" + g_server_state.run_id + "\\" + "single";

				auto crop_start = std::chrono::high_resolution_clock::now();
				// ROI/条纹/去噪
				cv::cuda::GpuMat d_cropped = cropImage(
					d_input,
					false,
					0, 0, 0, 0,
					false,
					0, 0, 0, 0, 0,
					"fft_image", "",
					p.enable_fourier_transform,
					p.filter_width,
					p.attenuation_factor,
					p.target_angle,
					p.angle_tolerance,
					p.enable_denoising,
					p.denoise_h,
					p.denoise_hColor,
					p.denoise_search_window,
					p.denoise_template_window
				);

				cudaDeviceSynchronize();

				cv::cuda::GpuMat o_cropped = cropImage(
					d_input,
					false,
					0, 0, 0, 0,
					false,
					0, 0, 0, 0, 0,
					"cropped_image", "",
					false,
					p.filter_width,
					p.attenuation_factor,
					p.target_angle,
					p.angle_tolerance,
					false,
					p.denoise_h,
					p.denoise_hColor,
					p.denoise_search_window,
					p.denoise_template_window
				);

				auto crop_end = std::chrono::high_resolution_clock::now();
				auto crop_duration = std::chrono::duration_cast<std::chrono::milliseconds>(crop_end - crop_start).count();
				std::cout << "SingleDetectThread: crop processing time: " << crop_duration << " ms" << std::endl;
				auto halcon_start = std::chrono::high_resolution_clock::now();

				// Halcon 中心点检测
				std::string skeleton_output_path;
				std::vector<std::pair<double, double>> centers;
				bool halcon_ok = false;
				try {
					halcon_ok = mHalcon.processImage(
						d_cropped,
						skeleton_output_path,
						centers,
						!savePath.empty(),
						savePath
					);
				}
				catch (const std::exception& e) {
					LOGE("HalconProcessor error: %s", e.what());
					halcon_ok = false;
				}

				std::cout << centers.size() << std::endl;
				auto halcon_end = std::chrono::high_resolution_clock::now();
				auto halcon_duration = std::chrono::duration_cast<std::chrono::milliseconds>(halcon_end - halcon_start).count();
				std::cout << "SingleDetectThread: halcon processing time: " << halcon_duration << " ms" << std::endl;
				auto slice_start = std::chrono::high_resolution_clock::now();

				// 提取切片并检测
				auto slices = ImageSlicer::extractSlicesGPU(
					d_cropped,
					centers,
					p.slice_distance,
					false,
					"",
					p.slice_width,
					p.slice_height
				);

				cudaDeviceSynchronize();

				auto slice_end = std::chrono::high_resolution_clock::now();
				auto slice_duration = std::chrono::duration_cast<std::chrono::milliseconds>(slice_end - slice_start).count();
				std::cout << "SingleDetectThread: slice processing time: " << slice_duration << " ms" << std::endl;
				auto trt_start = std::chrono::high_resolution_clock::now();

				bool hasDet = false;
				trtyolo::DetectRes detres;
				if (!slices.empty()) {
					detres = mSliceDetector->process_sliced_images(
						slices,
						p.nms_threshold,
						p.conf_threshold,
						std::vector<int>{0, 2}
					);
					hasDet = true;
				}

				auto trt_end = std::chrono::high_resolution_clock::now();
				auto trt_duration = std::chrono::duration_cast<std::chrono::milliseconds>(trt_end - trt_start).count();
				std::cout << "SingleDetectThread: trt processing time: " << trt_duration << " ms" << std::endl;
				auto sam_start = std::chrono::high_resolution_clock::now();

				// 可视化
				cv::Mat vis_cpu;
				d_cropped.download(vis_cpu, stream);
				stream.waitForCompletion();

				std::vector<double> areas_px;
				std::vector<double> lengths_px;

				std::string saved_path;

				if (hasDet && mSamReady && mSam) {
					double pix_to_mm = p.pix_to_mm;
					if (pix_to_mm <= 1e-9) pix_to_mm = 1.0;
					float minAreaPx = (p.min_area_mm2 > 0.0) ? static_cast<float>(p.min_area_mm2 / (pix_to_mm * pix_to_mm)) : 0.0f;
					float minDiamPx = (p.min_diameter_mm > 0.0) ? static_cast<float>(p.min_diameter_mm / pix_to_mm) : 0.0f;
					auto masks = mSam->inferFromDetections(vis_cpu, detres, minAreaPx, minDiamPx);
					areas_px.assign(detres.num, 0.0);
					lengths_px.assign(detres.num, 0.0);
					const auto& samAreas = mSam->lastAreasPx();
					const auto& samDiameters = mSam->lastDiametersPx();
					for (int i = 0; i < detres.num; ++i) {
						if (i < static_cast<int>(samAreas.size())) areas_px[i] = samAreas[i];
						if (i < static_cast<int>(samDiameters.size())) lengths_px[i] = samDiameters[i];
					}
					cv::Mat seg_vis = mSam->visualize(o_cropped, saved_path, detres, masks, !savePath.empty(), savePath);
				}

				o_cropped.release();
				d_cropped.release();
				d_input.release();
				slices.clear();

				mHalcon.waitForAsyncOperations();

				auto sam_end = std::chrono::high_resolution_clock::now();
				auto sam_duration = std::chrono::duration_cast<std::chrono::milliseconds>(sam_end - sam_start).count();
				std::cout << "SingleDetectThread: sam processing time: " << sam_duration << " ms" << std::endl;
				auto res_start = std::chrono::high_resolution_clock::now();

				// 构造结果
				SingleImageResult result;
				result.meta = FrameMeta{}; // 使用默认值
				result.saved_path = saved_path;
				if (!skeleton_output_path.empty()) result.skeleton_path = skeleton_output_path;

				float pix2 = p.pix_to_mm * p.pix_to_mm;
				float pix1 = p.pix_to_mm;
				if (p.pix_to_mm == 0)
				{
					pix2 = 1.0f;
					pix1 = 1.0f;
				}
				if (hasDet) {
					result.detections.reserve(detres.num);
					for (int i = 0; i < detres.num; ++i) {
						Detection d;
						d.label_id = detres.classes[i];
						d.confidence = detres.scores[i];
						const auto& b = detres.boxes[i];
						d.box.x = static_cast<int>(b.left);
						d.box.y = static_cast<int>(b.top);
						d.box.w = static_cast<int>(b.right - b.left);
						d.box.h = static_cast<int>(b.bottom - b.top);
						if (i < static_cast<int>(areas_px.size())) d.area = areas_px[i] * pix2;
						if (i < static_cast<int>(lengths_px.size())) d.length = lengths_px[i] * pix1;
						result.detections.push_back(std::move(d));
					}
				}

				auto res_end = std::chrono::high_resolution_clock::now();
				auto res_duration = std::chrono::duration_cast<std::chrono::milliseconds>(res_end - res_start).count();
				std::cout << "SingleDetectThread: res processing time: " << res_duration << " ms" << std::endl;

				auto total_end = std::chrono::high_resolution_clock::now();
				auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
				std::cout << "SingleDetectThread: total processing time: " << total_duration << " ms" << std::endl;

				stream.waitForCompletion();

				// 完成任务
				{
					std::lock_guard<std::mutex> lg(task->mtx);
					task->result = std::move(result);
					task->ok = true;
					task->done = true;
				}
				task->cv.notify_one();
			}
			catch (const std::exception& e) {
				std::lock_guard<std::mutex> lg(task->mtx);
				task->err = e.what();
				task->ok = false;
				task->done = true;
				task->cv.notify_one();
			}
			catch (...) {
				std::lock_guard<std::mutex> lg(task->mtx);
				task->err = "unknown error";
				task->ok = false;
				task->done = true;
				task->cv.notify_one();
			}
		}

		mRunning.store(false);
	}

} // namespace XL