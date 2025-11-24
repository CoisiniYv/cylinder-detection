#include "detect_thread.hpp"
#include "queue_manager.hpp"

#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cuda_runtime.h>
#include <iostream>

namespace XL {

	DetectThread::DetectThread(int thread_index, const DetectParams& params)
		: mIndex(thread_index), mParams(params) {
		mPreferPsImage = true; // 使用光度立体图，若没有则回退到RGB
	}

	DetectThread::~DetectThread() {
		stop();
		join();
	}

	void DetectThread::start() {
		if (mRunning.load()) return;
		// 将模型初始化移到工作线程中，确保 CUDA 上下文在该线程创建并使用
		mRunning.store(true);
		mThread = std::thread(&DetectThread::run, this);
	}

	void DetectThread::stop() {
		mRunning.store(false);
	}

	void DetectThread::join() {
		if (mThread.joinable()) mThread.join();
	}

	void DetectThread::run() {
		try {
			cv::cuda::setDevice(0);
		}
		catch (...) {
		}
		cudaError_t __devErr = cudaSetDevice(0);
		if (__devErr != cudaSuccess) {
			std::cerr << "DetectThread[" << mIndex << "]: cudaSetDevice failed: " << cudaGetErrorString(__devErr) << std::endl;
			mRunning.store(false);
			return;
		}

    cv::cuda::Stream stream;
    cv::cuda::GpuMat d_input;
    cv::cuda::GpuMat d_cropped;
    cv::cuda::GpuMat o_cropped;
    cv::Mat vis_cpu;
    std::vector<SliceInfo> slices;

		if (!mModelReady || !mSliceDetector) {
			try {
				trtyolo::InferOption infer_opt;
				infer_opt.enableSwapRB();
				mSliceDetector = std::make_unique<trtyolo::SliceDetector>(mParams.engine_file, infer_opt);
				mModelReady = true;
			}
			catch (const std::exception& e) {
				std::cerr << "DetectThread[" << mIndex << "]: model preload failed in run: " << e.what() << std::endl;
				mModelReady = false;
			}
		}
		std::this_thread::sleep_for(std::chrono::seconds(2));
		// 初始化SAM分割器
		if (!mSamReady || !mSam) {
			try {
				mSam = std::make_unique<SamSegmenter>();

				std::string actualEncoderPath = mParams.sam_encoder_engine_file;
				std::string actualDecoderPath = mParams.sam_decoder_engine_file;
				if (!std::filesystem::exists(actualEncoderPath))
					actualEncoderPath = mParams.sam_encoder_onnx_file;
				if (!std::filesystem::exists(actualDecoderPath))
					actualDecoderPath = mParams.sam_decoder_onnx_file;
				const std::string encoderEngine = actualEncoderPath;
				const std::string decoderEngine = actualDecoderPath;

				if (encoderEngine.empty() || decoderEngine.empty()) {
					std::cerr << "DetectThread[" << mIndex << "]: SAM engine paths are empty. Skip SAM init." << std::endl;
					mSamReady = false;
				}
				else {
					mSamReady = mSam->init(encoderEngine, decoderEngine);
					if (!mSamReady) {
						std::cerr << "DetectThread[" << mIndex << "]: SAM init failed." << std::endl;
					}
				}
			}
			catch (const std::exception& e) {
				std::cerr << "DetectThread[" << mIndex << "]: SAM init exception: " << e.what() << std::endl;
				mSamReady = false;
			}
			catch (...) {
				std::cerr << "DetectThread[" << mIndex << "]: SAM init unknown exception." << std::endl;
				mSamReady = false;
			}
		}
		std::this_thread::sleep_for(std::chrono::seconds(1));
		if (!mModelReady || !mSliceDetector) {
			std::cerr << "DetectThread[" << mIndex << "]: model not ready, abort run." << std::endl;
			mRunning.store(false);
			return;
		}
		std::this_thread::sleep_for(std::chrono::seconds(1));
		while (mRunning.load()) {
			// 以短超时阻塞，便于响应 stop
			ImageFrame frame;
			const bool ok = g_queue_manager.popRawBlocking(frame, 100);
			if (!ok) {
				if (g_queue_manager.stopped()) break; // 全局队列停止
				continue; // 超时或暂时无数据
			}

			// 选择源图
			const cv::Mat& src = (mPreferPsImage && frame.has_ps()) ? frame.ps_image : frame.rgb_image;
			if (src.empty()) {
				continue;
			}

			extern ServerState g_server_state;
			const std::string savePath = (g_server_state.config ? g_server_state.config->outputdir : std::string()) + "\\" + g_server_state.run_id + "\\" + std::to_string(frame.meta.group_id);

			std::filesystem::create_directories(savePath);

			std::string original_name = "original_g" + std::to_string(frame.meta.group_id) + "_i" + std::to_string(frame.meta.index_in_group);
			std::string original_path = savePath + "\\" + original_name + ".png";
			std::string skeleton_path;
			std::string save_path;

            d_input.upload(src, stream);

			if (mParams.qw_index == frame.meta.index_in_group)
			{
                d_cropped = cropImage(
                    d_input,
                    mParams.enable_four_side_crop,
                    mParams.crop_x, mParams.crop_y, mParams.crop_width, mParams.crop_height,
                    mParams.is_qw,
                    mParams.circle_x1, mParams.circle_y1, mParams.circle_x2, mParams.circle_y2, mParams.radius,
                    "fft_image", "",
                    mParams.enable_fourier_transform,
                    mParams.filter_width,
                    mParams.attenuation_factor,
                    mParams.target_angle,
                    mParams.angle_tolerance,
                    mParams.enable_denoising,
                    mParams.denoise_h,
                    mParams.denoise_hColor,
                    mParams.denoise_search_window,
                    mParams.denoise_template_window,
                    stream
                );

                o_cropped = cropImage(
                    d_input,
                    mParams.enable_four_side_crop,
                    mParams.crop_x, mParams.crop_y, mParams.crop_width, mParams.crop_height,
                    mParams.is_qw,
                    mParams.circle_x1, mParams.circle_y1, mParams.circle_x2, mParams.circle_y2, mParams.radius,
                    original_name, savePath,
                    false,
                    mParams.filter_width,
                    mParams.attenuation_factor,
                    mParams.target_angle,
                    mParams.angle_tolerance,
                    false,
                    mParams.denoise_h,
                    mParams.denoise_hColor,
                    mParams.denoise_search_window,
                    mParams.denoise_template_window,
                    stream
                );
			}
			else
			{
                d_cropped = cropImage(
                    d_input,
                    mParams.enable_four_side_crop,
                    mParams.crop_x, mParams.crop_y, mParams.crop_width, mParams.crop_height,
                    false,
                    mParams.circle_x1, mParams.circle_y1, mParams.circle_x2, mParams.circle_y2, mParams.radius,
                    "fft_image", "",
                    mParams.enable_fourier_transform,
                    mParams.filter_width,
                    mParams.attenuation_factor,
                    mParams.target_angle,
                    mParams.angle_tolerance,
                    mParams.enable_denoising,
                    mParams.denoise_h,
                    mParams.denoise_hColor,
                    mParams.denoise_search_window,
                    mParams.denoise_template_window,
                    stream
                );

                o_cropped = cropImage(
                    d_input,
                    mParams.enable_four_side_crop,
                    mParams.crop_x, mParams.crop_y, mParams.crop_width, mParams.crop_height,
                    false,
                    mParams.circle_x1, mParams.circle_y1, mParams.circle_x2, mParams.circle_y2, mParams.radius,
                    original_name, savePath,
                    false,
                    mParams.filter_width,
                    mParams.attenuation_factor,
                    mParams.target_angle,
                    mParams.angle_tolerance,
                    false,
                    mParams.denoise_h,
                    mParams.denoise_hColor,
                    mParams.denoise_search_window,
                    mParams.denoise_template_window,
                    stream
                );
            }
            stream.waitForCompletion();

			// Halcon 处理器：中心点检测（直接传入GPU图像，并传递流以保证下载/同步在同一流上）
			std::vector<std::pair<double, double>> centers;
			bool halcon_ok = false;
			try {
				// 读取上传目录（config.json -> uploadDir），目录规则：uploadDir/YYYYMMDD_HHMMSS_groupid
				halcon_ok = mHalcon.processImage(
					d_cropped,
					skeleton_path,
					centers,
					/*enableSaveToPath*/ !savePath.empty(),
					/*savePath*/ savePath,
					frame.meta.group_id,
					frame.meta.index_in_group
				);
			}
			catch (const std::exception& e) {
				std::cerr << "HalconProcessor error: " << e.what() << std::endl;
				halcon_ok = false;
			}

			if (!halcon_ok || centers.empty()) {
				// 构造空结果并入队（可选）。这里选择跳过。
				continue;
			}

			// 提取切片
            slices = ImageSlicer::extractSlicesGPU(
                d_cropped,
                centers,
                mParams.slice_distance,
                false,
                "",
                mParams.slice_width,
                mParams.slice_height,
                stream
            );
            if (slices.empty()) {
                continue;
            }
            // 切片推理
            trtyolo::DetectRes detres = mSliceDetector->process_sliced_images(
                slices,
                mParams.nms_threshold,
                mParams.conf_threshold,
                std::vector<int>{0, 2},
                cv::cuda::StreamAccessor::getStream(stream)
            );

			// 可视化并保存
            d_cropped.download(vis_cpu, stream);
            stream.waitForCompletion();

			// 使用 SAM 进行分割并可视化；如SAM未就绪则回退到 YOLO 可视化
			cv::Mat to_save;
			// SAM 分割（若已成功初始化）
			std::vector<double> areas_px; // 与 detres 对齐的像素面积
			std::vector<double> lengths_px; // 与 detres 对齐的像素“长度”（以最小外接圆直径近似）
			if (mSamReady && mSam) {
				// 将 mm 阈值转换为像素阈值：
				double pix_to_mm = mParams.pix_to_mm;
				if (pix_to_mm <= 1e-9) {
					// 防止除零；若未配置则按 1mm/pix 处理
					pix_to_mm = 1.0;
				}
				float minAreaPx = 0.0f;
				float minDiamPx = 0.0f;
				if (mParams.min_area_mm2 > 0.0) {
					minAreaPx = static_cast<float>(mParams.min_area_mm2 / (pix_to_mm * pix_to_mm));
				}
				if (mParams.min_diameter_mm > 0.0) {
					minDiamPx = static_cast<float>(mParams.min_diameter_mm / pix_to_mm);
				}
				// 执行分割与筛选（masks 与 detres 一一对应），并直接复用 SAM 已计算的面积/直径
				auto masks = mSam->inferFromDetections(vis_cpu, detres, minAreaPx, minDiamPx);

				// 直接使用 SamSegmenter 的最近一次度量，避免二次计算
				areas_px.assign(detres.num, 0.0);
				lengths_px.assign(detres.num, 0.0);
				const auto& samAreas = mSam->lastAreasPx();
				const auto& samDiameters = mSam->lastDiametersPx();
				for (int i = 0; i < detres.num; ++i) {
					if (i < static_cast<int>(samAreas.size())) {
						areas_px[i] = samAreas[i];
					}
					if (i < static_cast<int>(samDiameters.size())) {
						lengths_px[i] = samDiameters[i];
					}
				}

				cv::Mat seg_vis = mSam->visualize(o_cropped, save_path, detres, masks, !savePath.empty(), savePath, frame.meta.group_id, frame.meta.index_in_group);
				to_save = seg_vis.empty() ? vis_cpu : seg_vis;
			}

			mHalcon.waitForAsyncOperations();

            o_cropped.release();
            d_cropped.release();
            d_input.release();
            vis_cpu.release();
            slices.clear();

			// 构造结果并入队
			SingleImageResult result;
			result.meta = frame.meta;

			result.saved_path = save_path;
			if (!skeleton_path.empty()) {
				result.skeleton_path = skeleton_path;
			}
			result.original_path = original_path;

			result.detections.reserve(detres.num);

			float pix2 = mParams.pix_to_mm * mParams.pix_to_mm;
			float pix1 = mParams.pix_to_mm;
			if (mParams.pix_to_mm == 0)
			{
				pix2 = 1.0f;
				pix1 = 1.0f;
			}
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

			stream.waitForCompletion();

			g_queue_manager.pushResult(std::move(result));
		}

		mRunning.store(false);
	}

} // namespace XL