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

    if (!mModelReady || !mSliceDetector) {
        std::cerr << "DetectThread[" << mIndex << "]: model not ready, abort run." << std::endl;
        mRunning.store(false);
        return;
    }
     std::this_thread::sleep_for(std::chrono::seconds(2));
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
            // halcon_ok = mHalcon.processImage(d_cropped, outputImagePath, centers, false);
            halcon_ok = mHalcon.processImage(
                d_cropped,
                outputImagePath,
                centers,
                false,
                std::string(""),
                std::string(""),
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
        trtyolo::SliceDetector::visualize_sliced_result(vis_cpu, detres, mParams.labels);
        std::filesystem::path base(mParams.result_image_path.empty() ? std::filesystem::path("./output") : std::filesystem::path(mParams.result_image_path));
        std::string save_path;
        // 始终生成唯一文件名，避免覆盖
        const auto ts = frame.meta.timestamp_ms ? frame.meta.timestamp_ms : (std::int64_t)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        const std::string stem = base.has_extension() ? base.stem().string() : std::string("det");
        const std::string ext = base.has_extension() ? base.extension().string() : std::string(".png");
        auto fname = stem + "_" + std::to_string(frame.meta.sequence_id) + "_" + std::to_string(mIndex) + "_" + std::to_string(ts) + ext;
        if (base.has_extension()) {
            ensure_dir(base.parent_path());
            save_path = (base.parent_path() / fname).string();
        } else {
            ensure_dir(base);
            save_path = (base / fname).string();
        }
        cv::imwrite(save_path, vis_cpu);

        // 构造结果并入队
        SingleImageResult result;
        result.meta = frame.meta;
        result.saved_path = save_path;
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
            result.detections.push_back(std::move(d));
        }

        g_queue_manager.pushResult(std::move(result));
    }

    mRunning.store(false);
}

} // namespace XL