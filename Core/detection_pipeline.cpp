#include "detection_pipeline.hpp"

#include "Utils/Log.hpp"
#include "detect/HalconProcessor.h"
#include "detect/Slice.h"
#include "detect/crop_image.h"
#include "detect/preprocess_options.hpp"
#include "detect/sam.h"
#include "detect/trtyolo_slice.hpp"

#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace XL {
namespace {

bool regularFileExists(const std::string& file) {
    if (file.empty()) return false;
    std::error_code error;
    return std::filesystem::is_regular_file(file, error) && !error;
}

std::pair<std::string, std::string> selectSamModels(const DetectParams& params) {
    std::string encoder = params.sam_encoder_engine_file;
    std::string decoder = params.sam_decoder_engine_file;

    if (!regularFileExists(encoder) && !params.sam_encoder_onnx_file.empty()) {
        encoder = params.sam_encoder_onnx_file;
    }
    if (!regularFileExists(decoder) && !params.sam_decoder_onnx_file.empty()) {
        decoder = params.sam_decoder_onnx_file;
    }
    return {std::move(encoder), std::move(decoder)};
}

PreprocessOptions makePreprocessOptions(
    const DetectParams& params,
    bool apply_qw_mask,
    bool apply_stripe_removal,
    std::string file_name = "cropped_image",
    std::string output_dir = {}) {
    PreprocessOptions options;

    options.crop.enabled = params.enable_four_side_crop;
    options.crop.x = params.crop_x;
    options.crop.y = params.crop_y;
    options.crop.width = params.crop_width;
    options.crop.height = params.crop_height;

    options.qw_mask.enabled = apply_qw_mask;
    options.qw_mask.x1 = params.circle_x1;
    options.qw_mask.y1 = params.circle_y1;
    options.qw_mask.x2 = params.circle_x2;
    options.qw_mask.y2 = params.circle_y2;
    options.qw_mask.radius = params.radius;

    options.stripe.enabled = apply_stripe_removal && params.enable_fourier_transform;
    options.stripe.filter_width = params.filter_width;
    options.stripe.attenuation_factor = params.attenuation_factor;
    options.stripe.target_angle = params.target_angle;
    options.stripe.angle_tolerance = params.angle_tolerance;
    options.stripe.denoise = apply_stripe_removal && params.enable_denoising;
    options.stripe.denoise_h = params.denoise_h;
    options.stripe.denoise_h_color = params.denoise_hColor;
    options.stripe.denoise_search_window = params.denoise_search_window;
    options.stripe.denoise_template_window = params.denoise_template_window;

    options.save.file_name = std::move(file_name);
    options.save.output_dir = std::move(output_dir);
    return options;
}

SingleImageResult failedResult(
    const FrameMeta& meta,
    std::string message,
    std::string original_path = {},
    std::string skeleton_path = {}) {
    SingleImageResult result;
    result.meta = meta;
    result.processing_ok = false;
    result.error_message = std::move(message);
    result.original_path = std::move(original_path);
    result.skeleton_path = std::move(skeleton_path);
    return result;
}

int safeDetectionCount(const trtyolo::DetectRes& detections) {
    return std::max(0, std::min({
        detections.num,
        static_cast<int>(detections.boxes.size()),
        static_cast<int>(detections.classes.size()),
        static_cast<int>(detections.scores.size())}));
}

} // namespace

struct DetectionPipeline::Impl {
    explicit Impl(int id) : pipeline_id(id) {}

    int pipeline_id = -1;
    int current_gpu_device = -1;
    bool is_configured = false;
    DetectParams params;

    std::unique_ptr<trtyolo::SliceDetector> detector;
    std::string detector_engine;

    std::unique_ptr<SamSegmenter> sam;
    bool sam_ready = false;
    std::string sam_encoder;
    std::string sam_decoder;

    HalconProcessor halcon;

    void resetDeviceResources() {
        sam.reset();
        sam_ready = false;
        sam_encoder.clear();
        sam_decoder.clear();

        detector.reset();
        detector_engine.clear();
        is_configured = false;
    }

    bool selectCudaDevice(int device, std::string& error_message) {
        const cudaError_t cuda_error = cudaSetDevice(device);
        if (cuda_error != cudaSuccess) {
            error_message = std::string("cudaSetDevice failed: ") + cudaGetErrorString(cuda_error);
            return false;
        }

        try {
            cv::cuda::setDevice(device);
        }
        catch (const cv::Exception& e) {
            error_message = std::string("cv::cuda::setDevice failed: ") + e.what();
            return false;
        }

        current_gpu_device = device;
        return true;
    }

    bool configureModels(const DetectParams& next, std::string& error_message) {
        if (next.engine_file.empty()) {
            error_message = "detector engine path is empty";
            return false;
        }

        if (current_gpu_device != next.gpu_device) {
            resetDeviceResources();
            current_gpu_device = -1;
            if (!selectCudaDevice(next.gpu_device, error_message)) return false;
        }

        if (!detector || detector_engine != next.engine_file) {
            try {
                trtyolo::InferOption infer_option;
                infer_option.enableSwapRB();
                auto replacement = std::make_unique<trtyolo::SliceDetector>(
                    next.engine_file, infer_option);
                detector = std::move(replacement);
                detector_engine = next.engine_file;
            }
            catch (const std::exception& e) {
                error_message = std::string("detector initialization failed: ") + e.what();
                is_configured = false;
                return false;
            }
        }

        const auto [encoder, decoder] = selectSamModels(next);
        if (encoder.empty() || decoder.empty()) {
            sam.reset();
            sam_ready = false;
            sam_encoder.clear();
            sam_decoder.clear();
        }
        else if (!sam || !sam_ready || sam_encoder != encoder || sam_decoder != decoder) {
            auto replacement = std::make_unique<SamSegmenter>();
            if (replacement->init(encoder, decoder)) {
                sam = std::move(replacement);
                sam_ready = true;
                sam_encoder = encoder;
                sam_decoder = decoder;
            }
            else {
                // Segmentation is an optional refinement stage in the current
                // product behavior. Preserve YOLO results when SAM cannot load.
                sam.reset();
                sam_ready = false;
                sam_encoder = encoder;
                sam_decoder = decoder;
                LOGE("DetectionPipeline[%d] SAM unavailable; continuing without segmentation",
                     pipeline_id);
            }
        }

        params = next;
        is_configured = true;
        return true;
    }

    SingleImageResult processSource(
        const cv::Mat& source,
        const FrameMeta& meta,
        const std::filesystem::path& output_dir,
        const std::string& original_name,
        const std::string& original_path,
        bool save_original_image,
        bool apply_qw_mask) {
        SingleImageResult result;
        result.meta = meta;
        result.original_path = original_path;

        if (!is_configured || !detector) {
            return failedResult(meta, "detection pipeline is not configured", original_path);
        }
        if (source.empty()) {
            return failedResult(meta, "input image is empty", original_path);
        }

        try {
            std::filesystem::create_directories(output_dir);

            cv::cuda::Stream stream;
            cv::cuda::GpuMat input_gpu;
            input_gpu.upload(source, stream);

            const PreprocessOptions analysis_options = makePreprocessOptions(
                params,
                apply_qw_mask,
                true,
                "fft_image",
                {});
            cv::cuda::GpuMat processed_gpu = preprocessImage(
                input_gpu,
                analysis_options,
                stream);

            const PreprocessOptions visualization_options = makePreprocessOptions(
                params,
                apply_qw_mask,
                false,
                original_name,
                save_original_image ? output_dir.string() : std::string());
            cv::cuda::GpuMat visualization_source_gpu = preprocessImage(
                input_gpu,
                visualization_options,
                stream);

            // HALCON crosses from OpenCV's explicit CUDA stream to its own
            // execution path. Synchronize exactly at this ownership boundary.
            stream.waitForCompletion();

            std::vector<std::pair<double, double>> centers;
            if (!halcon.processImage(
                    processed_gpu,
                    result.skeleton_path,
                    centers,
                    true,
                    output_dir.string(),
                    meta.group_id,
                    meta.index_in_group)) {
                return failedResult(
                    meta,
                    "HALCON preprocessing failed",
                    original_path,
                    result.skeleton_path);
            }

            // No candidate centers is a valid GOOD image, not a processing
            // failure and not a missing worker result.
            if (centers.empty()) {
                halcon.waitForAsyncOperations();
                return result;
            }

            auto slices = ImageSlicer::extractSlicesGPU(
                processed_gpu,
                centers,
                params.slice_distance,
                false,
                "",
                params.slice_width,
                params.slice_height,
                stream);
            if (slices.empty()) {
                halcon.waitForAsyncOperations();
                return failedResult(
                    meta,
                    "slice extraction returned no images",
                    original_path,
                    result.skeleton_path);
            }

            trtyolo::DetectRes detections = detector->process_sliced_images(
                slices,
                params.nms_threshold,
                params.conf_threshold,
                params.allowed_class_indices);

            std::vector<double> areas_px;
            std::vector<double> diameters_px;
            if (sam_ready && sam && detections.num > 0) {
                cv::Mat segmentation_source;
                processed_gpu.download(segmentation_source, stream);
                stream.waitForCompletion();

                const double pixel_to_mm = params.pix_to_mm > 1e-9 ? params.pix_to_mm : 1.0;
                const float min_area_px = params.min_area_mm2 > 0.0
                    ? static_cast<float>(params.min_area_mm2 / (pixel_to_mm * pixel_to_mm))
                    : 0.0f;
                const float min_diameter_px = params.min_diameter_mm > 0.0
                    ? static_cast<float>(params.min_diameter_mm / pixel_to_mm)
                    : 0.0f;

                auto masks = sam->inferFromDetections(
                    segmentation_source,
                    detections,
                    min_area_px,
                    min_diameter_px);
                areas_px = sam->lastAreasPx();
                diameters_px = sam->lastDiametersPx();
                sam->visualize(
                    visualization_source_gpu,
                    result.saved_path,
                    detections,
                    masks,
                    true,
                    output_dir.string(),
                    meta.group_id,
                    meta.index_in_group);
            }

            const int detection_count = safeDetectionCount(detections);
            if (detection_count != detections.num) {
                LOGE("DetectionPipeline[%d] inconsistent detector result: num=%d usable=%d",
                     pipeline_id, detections.num, detection_count);
            }

            const double pixel_scale = params.pix_to_mm > 0.0 ? params.pix_to_mm : 1.0;
            const double area_scale = pixel_scale * pixel_scale;
            result.detections.reserve(static_cast<std::size_t>(detection_count));
            for (int i = 0; i < detection_count; ++i) {
                const std::size_t index = static_cast<std::size_t>(i);
                Detection detection;
                detection.label_id = detections.classes[index];
                detection.confidence = detections.scores[index];

                const auto& box = detections.boxes[index];
                detection.box.x = static_cast<int>(box.left);
                detection.box.y = static_cast<int>(box.top);
                detection.box.w = static_cast<int>(box.right - box.left);
                detection.box.h = static_cast<int>(box.bottom - box.top);
                if (index < areas_px.size()) detection.area = areas_px[index] * area_scale;
                if (index < diameters_px.size()) detection.length = diameters_px[index] * pixel_scale;
                result.detections.emplace_back(std::move(detection));
            }

            halcon.waitForAsyncOperations();
            return result;
        }
        catch (const std::exception& e) {
            try { halcon.waitForAsyncOperations(); }
            catch (...) {}
            return failedResult(meta, e.what(), original_path, result.skeleton_path);
        }
        catch (...) {
            try { halcon.waitForAsyncOperations(); }
            catch (...) {}
            return failedResult(
                meta,
                "unknown processing exception",
                original_path,
                result.skeleton_path);
        }
    }
};

DetectionPipeline::DetectionPipeline(int pipeline_id)
    : mImpl(std::make_unique<Impl>(pipeline_id)) {}

DetectionPipeline::~DetectionPipeline() = default;

bool DetectionPipeline::configure(
    const DetectParams& params,
    std::string& error_message) {
    error_message.clear();
    if (params.gpu_device < 0) {
        error_message = "gpu device must be non-negative";
        return false;
    }
    return mImpl->configureModels(params, error_message);
}

SingleImageResult DetectionPipeline::processFrame(
    const ImageFrame& frame,
    const DetectionContext& context) {
    if (!context.valid()) {
        return failedResult(frame.meta, "detection output context is invalid");
    }

    const cv::Mat& source = frame.has_ps() ? frame.ps_image : frame.rgb_image;
    const std::filesystem::path output_dir = context.groupOutputDir(frame.meta.group_id);
    const std::string original_name =
        "original_g" + std::to_string(frame.meta.group_id) +
        "_i" + std::to_string(frame.meta.index_in_group);
    const std::string original_path = (output_dir / (original_name + ".png")).string();
    const bool apply_qw_mask =
        mImpl->params.is_qw && mImpl->params.qw_index == frame.meta.index_in_group;

    return mImpl->processSource(
        source,
        frame.meta,
        output_dir,
        original_name,
        original_path,
        true,
        apply_qw_mask);
}

SingleImageResult DetectionPipeline::processFile(
    const std::string& input_file,
    const DetectionContext& context) {
    FrameMeta meta;
    meta.device_id = mImpl->params.device_id;

    if (!context.valid()) {
        return failedResult(meta, "detection output context is invalid", input_file);
    }

    cv::Mat source = cv::imread(input_file, cv::IMREAD_COLOR);
    if (source.empty()) {
        return failedResult(meta, "unable to read input image: " + input_file, input_file);
    }

    return mImpl->processSource(
        source,
        meta,
        context.singleOutputDir(),
        "single_original",
        input_file,
        false,
        mImpl->params.is_qw);
}

bool DetectionPipeline::configured() const noexcept {
    return mImpl && mImpl->is_configured;
}

int DetectionPipeline::gpuDevice() const noexcept {
    return mImpl ? mImpl->current_gpu_device : -1;
}

} // namespace XL
