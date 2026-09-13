#include "detect_thread.hpp"

#include "Config.hpp"
#include "Utils/Log.hpp"
#include "queue_manager.hpp"
#include "runtime_state.hpp"

#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>

#include <algorithm>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace XL {
namespace {

bool selectCudaDevice(int device, int worker_index) {
    const cudaError_t cuda_error = cudaSetDevice(device);
    if (cuda_error != cudaSuccess) {
        LOGE("DetectThread[%d] cudaSetDevice(%d) failed: %s",
             worker_index, device, cudaGetErrorString(cuda_error));
        return false;
    }

    try {
        cv::cuda::setDevice(device);
        return true;
    }
    catch (const cv::Exception& e) {
        LOGE("DetectThread[%d] cv::cuda::setDevice(%d) failed: %s",
             worker_index, device, e.what());
        return false;
    }
}

std::pair<std::string, std::string> selectSamModels(const DetectParams& params) {
    std::string encoder = params.sam_encoder_engine_file;
    std::string decoder = params.sam_decoder_engine_file;
    if (!encoder.empty() && !std::filesystem::exists(encoder)) encoder = params.sam_encoder_onnx_file;
    if (!decoder.empty() && !std::filesystem::exists(decoder)) decoder = params.sam_decoder_onnx_file;
    return {std::move(encoder), std::move(decoder)};
}

SingleImageResult makeFailedResult(const ImageFrame& frame, const std::string& message) {
    SingleImageResult result;
    result.meta = frame.meta;
    result.processing_ok = false;
    result.error_message = message;
    return result;
}

void publishResult(SingleImageResult&& result) {
    if (!g_queue_manager.pushResult(std::move(result))) {
        LOGE("failed to publish detection result because result queue is stopped");
    }
}

} // namespace

DetectThread::DetectThread(int thread_index, const DetectParams& params)
    : mIndex(thread_index), mParams(params) {}

DetectThread::~DetectThread() {
    stop();
    join();
}

void DetectThread::start() {
    if (mRunning.exchange(true, std::memory_order_relaxed)) return;
    mThread = std::thread(&DetectThread::run, this);
}

void DetectThread::stop() {
    mRunning.store(false, std::memory_order_relaxed);
}

void DetectThread::join() {
    if (mThread.joinable()) mThread.join();
}

void DetectThread::run() {
    if (!selectCudaDevice(mParams.gpu_device, mIndex)) {
        mRunning.store(false, std::memory_order_relaxed);
        return;
    }

    try {
        trtyolo::InferOption infer_option;
        infer_option.enableSwapRB();
        mSliceDetector = std::make_unique<trtyolo::SliceDetector>(mParams.engine_file, infer_option);
        mModelReady = true;
    }
    catch (const std::exception& e) {
        LOGE("DetectThread[%d] detector initialization failed: %s", mIndex, e.what());
        mRunning.store(false, std::memory_order_relaxed);
        return;
    }

    try {
        const auto [encoder, decoder] = selectSamModels(mParams);
        if (!encoder.empty() && !decoder.empty()) {
            mSam = std::make_unique<SamSegmenter>();
            mSamReady = mSam->init(encoder, decoder);
            if (!mSamReady) mSam.reset();
        }
    }
    catch (const std::exception& e) {
        LOGE("DetectThread[%d] SAM initialization failed: %s", mIndex, e.what());
        mSam.reset();
        mSamReady = false;
    }

    cv::cuda::Stream stream;

    while (mRunning.load(std::memory_order_relaxed)) {
        ImageFrame frame;
        if (!g_queue_manager.popRawBlocking(frame, 100)) {
            if (g_queue_manager.stopped()) break;
            continue;
        }

        try {
            const cv::Mat& source = (mPreferPsImage && frame.has_ps()) ? frame.ps_image : frame.rgb_image;
            if (source.empty()) {
                publishResult(makeFailedResult(frame, "input frame is empty"));
                continue;
            }

            const Config* config = g_runtime_state.config;
            if (!config || g_runtime_state.run_id.empty()) {
                publishResult(makeFailedResult(frame, "runtime output context is unavailable"));
                continue;
            }

            const std::filesystem::path output_dir =
                std::filesystem::path(config->outputdir) /
                g_runtime_state.run_id /
                std::to_string(frame.meta.group_id);
            std::filesystem::create_directories(output_dir);

            const std::string original_name =
                "original_g" + std::to_string(frame.meta.group_id) +
                "_i" + std::to_string(frame.meta.index_in_group);
            const std::string original_path = (output_dir / (original_name + ".png")).string();

            cv::cuda::GpuMat input_gpu;
            input_gpu.upload(source, stream);

            const bool apply_qw_mask =
                mParams.is_qw && mParams.qw_index == frame.meta.index_in_group;

            cv::cuda::GpuMat processed_gpu = cropImage(
                input_gpu,
                mParams.enable_four_side_crop,
                mParams.crop_x,
                mParams.crop_y,
                mParams.crop_width,
                mParams.crop_height,
                apply_qw_mask,
                mParams.circle_x1,
                mParams.circle_y1,
                mParams.circle_x2,
                mParams.circle_y2,
                mParams.radius,
                "fft_image",
                "",
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
                stream);

            cv::cuda::GpuMat visualization_source_gpu = cropImage(
                input_gpu,
                mParams.enable_four_side_crop,
                mParams.crop_x,
                mParams.crop_y,
                mParams.crop_width,
                mParams.crop_height,
                apply_qw_mask,
                mParams.circle_x1,
                mParams.circle_y1,
                mParams.circle_x2,
                mParams.circle_y2,
                mParams.radius,
                original_name,
                output_dir.string(),
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
                stream);

            // HALCON downloads from GpuMat using its own/default CUDA context.
            // Make the producer stream complete before crossing that boundary.
            stream.waitForCompletion();

            std::string skeleton_path;
            std::vector<std::pair<double, double>> centers;
            if (!mHalcon.processImage(
                    processed_gpu,
                    skeleton_path,
                    centers,
                    true,
                    output_dir.string(),
                    frame.meta.group_id,
                    frame.meta.index_in_group)) {
                auto failed = makeFailedResult(frame, "HALCON preprocessing failed");
                failed.original_path = original_path;
                publishResult(std::move(failed));
                continue;
            }

            SingleImageResult result;
            result.meta = frame.meta;
            result.original_path = original_path;
            result.skeleton_path = skeleton_path;

            // No centers is a valid no-defect result, not a missing frame.
            if (centers.empty()) {
                mHalcon.waitForAsyncOperations();
                publishResult(std::move(result));
                continue;
            }

            auto slices = ImageSlicer::extractSlicesGPU(
                processed_gpu,
                centers,
                mParams.slice_distance,
                false,
                "",
                mParams.slice_width,
                mParams.slice_height,
                stream);
            if (slices.empty()) {
                auto failed = makeFailedResult(frame, "slice extraction returned no images");
                failed.original_path = original_path;
                failed.skeleton_path = skeleton_path;
                publishResult(std::move(failed));
                continue;
            }

            trtyolo::DetectRes detections = mSliceDetector->process_sliced_images(
                slices,
                mParams.nms_threshold,
                mParams.conf_threshold,
                mParams.allowed_class_indices);

            cv::Mat visualization_cpu;
            processed_gpu.download(visualization_cpu, stream);
            stream.waitForCompletion();

            std::vector<double> areas_px;
            std::vector<double> diameters_px;
            std::string saved_path;

            if (mSamReady && mSam && detections.num > 0) {
                double pixel_to_mm = mParams.pix_to_mm;
                if (pixel_to_mm <= 1e-9) pixel_to_mm = 1.0;
                const float min_area_px = mParams.min_area_mm2 > 0.0
                    ? static_cast<float>(mParams.min_area_mm2 / (pixel_to_mm * pixel_to_mm))
                    : 0.0f;
                const float min_diameter_px = mParams.min_diameter_mm > 0.0
                    ? static_cast<float>(mParams.min_diameter_mm / pixel_to_mm)
                    : 0.0f;

                auto masks = mSam->inferFromDetections(
                    visualization_cpu, detections, min_area_px, min_diameter_px);
                areas_px = mSam->lastAreasPx();
                diameters_px = mSam->lastDiametersPx();
                mSam->visualize(
                    visualization_source_gpu,
                    saved_path,
                    detections,
                    masks,
                    true,
                    output_dir.string(),
                    frame.meta.group_id,
                    frame.meta.index_in_group);
            }

            result.saved_path = saved_path;
            result.detections.reserve(static_cast<std::size_t>(std::max(0, detections.num)));

            const double pixel_scale = mParams.pix_to_mm > 0.0 ? mParams.pix_to_mm : 1.0;
            const double area_scale = pixel_scale * pixel_scale;
            for (int i = 0; i < detections.num; ++i) {
                Detection detection;
                detection.label_id = detections.classes[i];
                detection.confidence = detections.scores[i];
                const auto& box = detections.boxes[i];
                detection.box.x = static_cast<int>(box.left);
                detection.box.y = static_cast<int>(box.top);
                detection.box.w = static_cast<int>(box.right - box.left);
                detection.box.h = static_cast<int>(box.bottom - box.top);
                if (i < static_cast<int>(areas_px.size())) detection.area = areas_px[i] * area_scale;
                if (i < static_cast<int>(diameters_px.size())) detection.length = diameters_px[i] * pixel_scale;
                result.detections.emplace_back(std::move(detection));
            }

            mHalcon.waitForAsyncOperations();
            publishResult(std::move(result));
        }
        catch (const std::exception& e) {
            LOGE("DetectThread[%d] frame processing failed: %s", mIndex, e.what());
            publishResult(makeFailedResult(frame, e.what()));
        }
        catch (...) {
            LOGE("DetectThread[%d] frame processing failed with unknown exception", mIndex);
            publishResult(makeFailedResult(frame, "unknown processing exception"));
        }
    }

    mRunning.store(false, std::memory_order_relaxed);
}

} // namespace XL
