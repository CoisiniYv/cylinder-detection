#include "single_detect_thread.hpp"

#include "Config.hpp"
#include "Utils/Log.hpp"
#include "detect/Slice.h"
#include "detect/crop_image.h"
#include "runtime_state.hpp"

#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <filesystem>
#include <utility>
#include <vector>

namespace XL {
namespace {

std::pair<std::string, std::string> selectSamModels(const DetectParams& params) {
    std::string encoder = params.sam_encoder_engine_file;
    std::string decoder = params.sam_decoder_engine_file;
    if (!encoder.empty() && !std::filesystem::exists(encoder)) encoder = params.sam_encoder_onnx_file;
    if (!decoder.empty() && !std::filesystem::exists(decoder)) decoder = params.sam_decoder_onnx_file;
    return {std::move(encoder), std::move(decoder)};
}

} // namespace

SingleDetectThread::SingleDetectThread() {
    mHalcon.setSaveCompleteCallback(
        [this](const std::string& file_path,
               bool success,
               HalconProcessor::SaveType,
               const std::string& message) {
            const auto status = mHalcon.getAsyncStatus();
            LOGI("HALCON save completed: path=%s success=%d message=%s pending=%d completed=%d failed=%d",
                 file_path.c_str(),
                 success ? 1 : 0,
                 message.c_str(),
                 status.pendingOperations,
                 status.completedOperations,
                 status.failedOperations);
        });
}

SingleDetectThread::~SingleDetectThread() {
    stop();
    join();
    mHalcon.clearSaveCompleteCallback();
}

void SingleDetectThread::start() {
    if (mRunning.exchange(true, std::memory_order_relaxed)) return;
    mThread = std::thread(&SingleDetectThread::worker, this);
}

void SingleDetectThread::stop() {
    mRunning.store(false, std::memory_order_relaxed);
    failPendingTasks("single-image worker is stopping");
    mQueueCondition.notify_all();
}

void SingleDetectThread::join() {
    if (mThread.joinable()) mThread.join();
}

void SingleDetectThread::completeTask(
    const std::shared_ptr<Task>& task,
    bool ok,
    SingleImageResult result,
    std::string error) {
    if (!task) return;
    {
        std::lock_guard<std::mutex> lock(task->mutex);
        if (task->done) return;
        task->ok = ok;
        task->result = std::move(result);
        task->error = std::move(error);
        task->done = true;
    }
    task->condition.notify_one();
}

void SingleDetectThread::failPendingTasks(const std::string& message) {
    std::deque<std::shared_ptr<Task>> pending;
    {
        std::lock_guard<std::mutex> lock(mQueueMutex);
        pending.swap(mTasks);
    }
    for (auto& task : pending) {
        completeTask(task, false, {}, message);
    }
}

bool SingleDetectThread::submitAndWait(
    const DetectParams& params,
    SingleImageResult& out_result,
    std::string& error_message) {
    if (!mRunning.load(std::memory_order_relaxed)) start();

    auto task = std::make_shared<Task>();
    task->params = params;
    {
        std::lock_guard<std::mutex> lock(mQueueMutex);
        if (!mRunning.load(std::memory_order_relaxed)) {
            error_message = "single-image worker is not running";
            return false;
        }
        mTasks.push_back(task);
    }
    mQueueCondition.notify_one();

    std::unique_lock<std::mutex> lock(task->mutex);
    task->condition.wait(lock, [&task] { return task->done; });
    if (!task->ok) {
        error_message = task->error;
        return false;
    }

    out_result = std::move(task->result);
    return true;
}

void SingleDetectThread::worker() {
    while (true) {
        std::shared_ptr<Task> task;
        {
            std::unique_lock<std::mutex> lock(mQueueMutex);
            mQueueCondition.wait(lock, [this] {
                return !mRunning.load(std::memory_order_relaxed) || !mTasks.empty();
            });
            if (!mRunning.load(std::memory_order_relaxed) && mTasks.empty()) break;
            if (mTasks.empty()) continue;
            task = mTasks.front();
            mTasks.pop_front();
        }

        try {
            const DetectParams& params = task->params;

            if (mCurrentGpuDevice != params.gpu_device) {
                // Destroy device-bound resources before switching contexts.
                mSam.reset();
                mSamReady = false;
                mSliceDetector.reset();
                mLastEncoderPath.clear();
                mLastDecoderPath.clear();
                mLastTrtEngineFile.clear();

                const cudaError_t cuda_error = cudaSetDevice(params.gpu_device);
                if (cuda_error != cudaSuccess) {
                    throw std::runtime_error(
                        std::string("cudaSetDevice failed: ") + cudaGetErrorString(cuda_error));
                }
                cv::cuda::setDevice(params.gpu_device);
                mCurrentGpuDevice = params.gpu_device;
            }

            if (!mSliceDetector || mLastTrtEngineFile != params.engine_file) {
                trtyolo::InferOption infer_option;
                infer_option.enableSwapRB();
                mSliceDetector = std::make_unique<trtyolo::SliceDetector>(
                    params.engine_file, infer_option);
                mLastTrtEngineFile = params.engine_file;
            }

            const auto [encoder, decoder] = selectSamModels(params);
            if (!encoder.empty() && !decoder.empty()) {
                if (!mSam || !mSamReady ||
                    mLastEncoderPath != encoder || mLastDecoderPath != decoder) {
                    mSam = std::make_unique<SamSegmenter>();
                    mSamReady = mSam->init(encoder, decoder);
                    mLastEncoderPath = encoder;
                    mLastDecoderPath = decoder;
                    if (!mSamReady) mSam.reset();
                }
            }
            else {
                mSam.reset();
                mSamReady = false;
                mLastEncoderPath.clear();
                mLastDecoderPath.clear();
            }

            cv::Mat source = cv::imread(params.input_image_path, cv::IMREAD_COLOR);
            if (source.empty()) {
                throw std::runtime_error("unable to read input image: " + params.input_image_path);
            }

            const Config* config = g_runtime_state.config;
            if (!config || g_runtime_state.run_id.empty()) {
                throw std::runtime_error("runtime output context is unavailable");
            }
            const std::filesystem::path output_dir =
                std::filesystem::path(config->outputdir) / g_runtime_state.run_id / "single";
            std::filesystem::create_directories(output_dir);

            cv::cuda::Stream stream;
            cv::cuda::GpuMat input_gpu;
            input_gpu.upload(source, stream);

            cv::cuda::GpuMat processed_gpu = cropImage(
                input_gpu,
                params.enable_four_side_crop,
                params.crop_x,
                params.crop_y,
                params.crop_width,
                params.crop_height,
                params.is_qw,
                params.circle_x1,
                params.circle_y1,
                params.circle_x2,
                params.circle_y2,
                params.radius,
                "fft_image",
                "",
                params.enable_fourier_transform,
                params.filter_width,
                params.attenuation_factor,
                params.target_angle,
                params.angle_tolerance,
                params.enable_denoising,
                params.denoise_h,
                params.denoise_hColor,
                params.denoise_search_window,
                params.denoise_template_window,
                stream);

            cv::cuda::GpuMat visualization_source_gpu = cropImage(
                input_gpu,
                params.enable_four_side_crop,
                params.crop_x,
                params.crop_y,
                params.crop_width,
                params.crop_height,
                params.is_qw,
                params.circle_x1,
                params.circle_y1,
                params.circle_x2,
                params.circle_y2,
                params.radius,
                "single_original",
                "",
                false,
                params.filter_width,
                params.attenuation_factor,
                params.target_angle,
                params.angle_tolerance,
                false,
                params.denoise_h,
                params.denoise_hColor,
                params.denoise_search_window,
                params.denoise_template_window,
                stream);

            stream.waitForCompletion();

            std::string skeleton_path;
            std::vector<std::pair<double, double>> centers;
            if (!mHalcon.processImage(
                    processed_gpu,
                    skeleton_path,
                    centers,
                    true,
                    output_dir.string())) {
                throw std::runtime_error("HALCON preprocessing failed");
            }

            SingleImageResult result;
            result.meta.device_id = params.device_id;
            result.original_path = params.input_image_path;
            result.skeleton_path = skeleton_path;

            if (!centers.empty()) {
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
                    throw std::runtime_error("slice extraction returned no images");
                }

                trtyolo::DetectRes detections = mSliceDetector->process_sliced_images(
                    slices,
                    params.nms_threshold,
                    params.conf_threshold,
                    params.allowed_class_indices);

                cv::Mat visualization_cpu;
                processed_gpu.download(visualization_cpu, stream);
                stream.waitForCompletion();

                std::vector<double> areas_px;
                std::vector<double> diameters_px;
                if (mSamReady && mSam && detections.num > 0) {
                    double pixel_to_mm = params.pix_to_mm;
                    if (pixel_to_mm <= 1e-9) pixel_to_mm = 1.0;
                    const float min_area_px = params.min_area_mm2 > 0.0
                        ? static_cast<float>(params.min_area_mm2 / (pixel_to_mm * pixel_to_mm))
                        : 0.0f;
                    const float min_diameter_px = params.min_diameter_mm > 0.0
                        ? static_cast<float>(params.min_diameter_mm / pixel_to_mm)
                        : 0.0f;

                    auto masks = mSam->inferFromDetections(
                        visualization_cpu, detections, min_area_px, min_diameter_px);
                    areas_px = mSam->lastAreasPx();
                    diameters_px = mSam->lastDiametersPx();
                    mSam->visualize(
                        visualization_source_gpu,
                        result.saved_path,
                        detections,
                        masks,
                        true,
                        output_dir.string());
                }

                const double pixel_scale = params.pix_to_mm > 0.0 ? params.pix_to_mm : 1.0;
                const double area_scale = pixel_scale * pixel_scale;
                result.detections.reserve(static_cast<std::size_t>(std::max(0, detections.num)));
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
            }

            mHalcon.waitForAsyncOperations();
            completeTask(task, true, std::move(result), {});
        }
        catch (const std::exception& e) {
            LOGE("single-image task failed: %s", e.what());
            completeTask(task, false, {}, e.what());
        }
        catch (...) {
            LOGE("single-image task failed with unknown exception");
            completeTask(task, false, {}, "unknown processing exception");
        }
    }

    failPendingTasks("single-image worker stopped before processing task");
    mRunning.store(false, std::memory_order_relaxed);
}

} // namespace XL
