#include "detect_thread.hpp"
#include "queue_manager.hpp"
#include "detect/crop_image.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <filesystem>

namespace XL {

static bool ensure_dir(const std::filesystem::path& p) {
    std::error_code ec; if (std::filesystem::exists(p, ec)) return std::filesystem::is_directory(p, ec);
    return std::filesystem::create_directories(p, ec);
}

std::string DetectThread::makeOutputFilename(const FrameMeta& meta, int thread_index, const std::string& ext) {
    // 使用默认 ./output 目录生成文件名（当调用方未自定义路径时可用）
    std::filesystem::path base("./output");
    ensure_dir(base);
    const auto ts = meta.timestamp_ms ? meta.timestamp_ms : (std::int64_t)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    auto fname = std::string("det_") + std::to_string(meta.sequence_id) + "_" + std::to_string(thread_index) + "_" + std::to_string(ts) + ext;
    return (base / fname).string();
}

DetectThread::DetectThread(int thread_index, const DetectParams& params)
    : mIndex(thread_index), mParams(params) {
    mPreferPsImage = true; // 先用PS图，若没有则回退到RGB
}

DetectThread::~DetectThread() {
    stop();
    join();
    mHalcon.cleanupTempFiles();
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
    // 每个工作线程绑定 CUDA 设备，并在该线程中初始化 TensorRT/模型上下文
    try {
        cv::cuda::setDevice(0);
        cudaSetDevice(0);
    } catch (...) {
        // 兼容性考虑：忽略可能的异常以避免中断
    }

    // 创建该线程专属的 CUDA Stream，贯穿整条流水线，避免使用默认（legacy）流
    cv::cuda::Stream stream;

    if (!mModelReady || !mSliceDetector) {
        try {
            trtyolo::InferOption infer_opt; 
            if (mParams.enable_swap_rb) {
                infer_opt.enableSwapRB();
            }
            mSliceDetector = std::make_unique<trtyolo::SliceDetector>(mParams.trt_engine_file, infer_opt);
            mModelReady = true;
        } catch (const std::exception& e) {
            std::cerr << "DetectThread[" << mIndex << "]: model preload failed in run: " << e.what() << std::endl;
            mModelReady = false;
        }
    }
    std::this_thread::sleep_for(std::chrono::seconds(2));
    // 初始化 SAM 分割器（一次性）
    if (!mSamReady || !mSam) {
        try {
            mSam = std::make_unique<SamSegmenter>();
            const std::string encoderEngine = mParams.sam_encoder_engine_file;
            const std::string decoderEngine = mParams.sam_decoder_engine_file;
            if (encoderEngine.empty() || decoderEngine.empty()) {
                std::cerr << "DetectThread[" << mIndex << "]: SAM engine paths are empty. Skip SAM init." << std::endl;
                mSamReady = false;
            } else {
                mSamReady = mSam->init(encoderEngine, decoderEngine);
                if (!mSamReady) {
                    std::cerr << "DetectThread[" << mIndex << "]: SAM init failed." << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "DetectThread[" << mIndex << "]: SAM init exception: " << e.what() << std::endl;
            mSamReady = false;
        } catch (...) {
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

        // 上传到GPU（使用线程专属流）
        cv::cuda::GpuMat d_input; d_input.upload(src, stream);

        // ROI裁剪 + 条纹去除 + 去噪（传入统一的流）
        const bool enable_fft = mParams.stripe_radius > 0 || mParams.filter_width > 0; // 有配置即启用
        cv::cuda::GpuMat d_cropped = cropImage(
            d_input,
            true,
            mParams.crop_x, mParams.crop_y, mParams.crop_width, mParams.crop_height,
            true,
            mParams.line_x1, mParams.line_y1, mParams.line_x2, mParams.line_y2, mParams.stripe_radius,
            "cropped_image", "",
            enable_fft,
            mParams.filter_width,
            mParams.attenuation_factor,
            mParams.target_angle,
            mParams.angle_tolerance,
            true,
            mParams.denoise_h,
            mParams.denoise_hColor,
            mParams.denoise_search_window,
            mParams.denoise_template_window
        );

        // Halcon 处理器：中心点检测（直接传入GPU图像，并传递流以保证下载/同步在同一流上）
        std::string outputImagePath;
        std::vector<std::pair<double,double>> centers;
        bool halcon_ok = false;
        try {
            // 读取上传目录（config.json -> uploadDir），目录规则：uploadDir/run_YYYYMMDD_HHMMSS_groupid
            extern ServerState g_server_state;
            const std::string savePath = (g_server_state.config ? g_server_state.config->outputdir : std::string());
            const std::string uuid = std::string("run_") + g_server_state.run_id + "_" + std::to_string(frame.meta.group_id);
            halcon_ok = mHalcon.processImage(
                d_cropped,
                outputImagePath,
                centers,
                /*enableSaveToPath*/ !savePath.empty(),
                /*uuid*/ uuid,
                /*savePath*/ savePath,
                frame.meta.group_id,
                frame.meta.index_in_group
            );
        } catch (const std::exception& e) {
            std::cerr << "HalconProcessor error: " << e.what() << std::endl;
            halcon_ok = false;
        }

        // mHalcon.cleanupTempFiles();
        if (!halcon_ok || centers.empty()) {
            // 构造空结果并入队（可选）。这里选择跳过。
            continue;
        }

        // 提取切片（GPU->CPU），传递流以避免默认流下载
        auto slices = ImageSlicer::extractSlicesGPU(
            d_cropped,
            centers,
            mParams.slice_distance,
            mParams.save_slices,
            mParams.slices_save_dir,
            mParams.slice_width,
            mParams.slice_height
        );
        if (slices.empty()) {
            continue;
        }
        // 切片推理
        trtyolo::DetectRes detres = mSliceDetector->process_sliced_images(
            slices,
            mParams.nms_threshold,
            mParams.conf_threshold,
            std::vector<int>{0,2} // 不限制类别，保留所有检测框
        );

        // 可视化并保存（下载时使用流，并在写盘前等待完成）
        cv::Mat vis_cpu; d_cropped.download(vis_cpu, stream);
        stream.waitForCompletion();

        // 使用 SAM 进行分割并可视化；如 SAM 未就绪则回退到 YOLO 可视化
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
            // 保存 SAM 结果到 uploadDir/run_<run_id>_<group_id>/output_i<index>.png
            extern ServerState g_server_state;
            const std::string savePath = (g_server_state.config ? g_server_state.config->outputdir : std::string());
            const std::string uuid = std::string("run_") + g_server_state.run_id + "_" + std::to_string(frame.meta.group_id);
            cv::Mat seg_vis = mSam->visualize(d_cropped, detres, masks, !savePath.empty(), uuid, savePath, frame.meta.index_in_group);
            to_save = seg_vis.empty() ? vis_cpu : seg_vis;
        } else {
            // 无 SAM 时，简单回退为 bbox 估计，确保字段有值
            areas_px.assign(detres.num, 0.0);
            lengths_px.assign(detres.num, 0.0);
            for (int i = 0; i < detres.num; ++i) {
                const auto& b = detres.boxes[i];
                int w = static_cast<int>(b.right - b.left);
                int h = static_cast<int>(b.bottom - b.top);
                areas_px[i] = static_cast<double>(w) * static_cast<double>(h);
                lengths_px[i] = static_cast<double>(std::max(w, h));
            }
            trtyolo::SliceDetector::visualize_sliced_result(vis_cpu, detres, mParams.labels);
            to_save = vis_cpu;
        }

        // 统一保存到 uploadDir/run_<run_id>_<group_id>/output_i<index>.png
        extern ServerState g_server_state;
        const std::string savePath = (g_server_state.config ? g_server_state.config->outputdir : std::string());
        const std::string uuid = std::string("run_") + g_server_state.run_id + "_" + std::to_string(frame.meta.group_id);
        std::filesystem::path out_dir = std::filesystem::path(savePath) / uuid;
        ensure_dir(out_dir);
        std::string save_path = (out_dir / (std::string("output_i") + std::to_string(frame.meta.index_in_group) + ".png")).string();
        cv::imwrite(save_path, to_save);

        
        // 构造结果并入队
        SingleImageResult result;
        result.meta = frame.meta;
        // 输出图保存路径：uploadDir/run_<run_id>_<group_id>/output_i<index>.png
        result.saved_path = save_path;
        // Halcon 骨架图保存路径：uploadDir/run_<run_id>_<group_id>/skeleton_i<index>.png
        if (!outputImagePath.empty()) {
            result.skeleton_path = outputImagePath;
        }

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
            if (d.label_id >= 0 && d.label_id < static_cast<int>(mParams.labels.size())) {
                d.label = mParams.labels[d.label_id];
            } else {
                d.label = std::string("class_") + std::to_string(d.label_id);
            }
            // 写入与 SAM 掩码对齐的面积/长度（像素单位）
            if (i < static_cast<int>(areas_px.size())) d.area = areas_px[i];
            if (i < static_cast<int>(lengths_px.size())) d.length = lengths_px[i];
            result.detections.push_back(std::move(d));
        }

        g_queue_manager.pushResult(std::move(result));
    }

    mRunning.store(false);
}

} // namespace XL