#include "HalconProcessor.h"

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <iostream>

HalconProcessor::HalconProcessor() {
    HalconCpp::SetSystem("parallelize_operators", "true");
    mAsyncWorkerThread = std::thread(&HalconProcessor::asyncWorkerFunction, this);
}

HalconProcessor::~HalconProcessor() {
    mStopAsyncThread.store(true, std::memory_order_relaxed);
    mQueueCondition.notify_all();
    if (mAsyncWorkerThread.joinable()) mAsyncWorkerThread.join();
}

void HalconProcessor::setSaveCompleteCallback(SaveCompleteCallback callback) {
    std::lock_guard<std::mutex> lock(mCallbackMutex);
    mSaveCompleteCallback = std::move(callback);
}

void HalconProcessor::clearSaveCompleteCallback() {
    std::lock_guard<std::mutex> lock(mCallbackMutex);
    mSaveCompleteCallback = nullptr;
}

void HalconProcessor::invokeSaveCallback(
    const std::string& file_path,
    bool success,
    SaveType type,
    const std::string& message) {
    SaveCompleteCallback callback;
    {
        std::lock_guard<std::mutex> lock(mCallbackMutex);
        callback = mSaveCompleteCallback;
    }

    if (!callback) return;
    try {
        callback(file_path, success, type, message);
    }
    catch (const std::exception& e) {
        std::cerr << "HALCON save callback failed: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "HALCON save callback failed with unknown exception" << std::endl;
    }
}

void HalconProcessor::waitForAsyncOperations() {
    std::unique_lock<std::mutex> lock(mQueueMutex);
    mQueueCondition.wait(lock, [this] {
        return mPendingAsyncOperations.load(std::memory_order_relaxed) == 0;
    });
}

HalconProcessor::AsyncStatus HalconProcessor::getAsyncStatus() const {
    return AsyncStatus{
        mPendingAsyncOperations.load(std::memory_order_relaxed),
        mCompletedAsyncOperations.load(std::memory_order_relaxed),
        mFailedAsyncOperations.load(std::memory_order_relaxed)};
}

std::string HalconProcessor::generateSavePathFilename(
    const std::string& save_path,
    std::uint64_t group_id,
    int index_in_group) const {
    const std::filesystem::path directory(save_path);
    std::filesystem::create_directories(directory);
    return (directory /
            ("skeleton_g" + std::to_string(group_id) +
             "_i" + std::to_string(index_in_group) + ".png"))
        .string();
}

std::future<bool> HalconProcessor::saveOverlayImageAsync(
    const HalconCpp::HObject& image,
    const HalconCpp::HObject& overlay_region,
    const std::string& output_path) {
    auto task = std::make_shared<AsyncSaveTask>();
    try {
        HalconCpp::CopyImage(image, &task->image);
        if (overlay_region.CountObj() > 0) {
            HalconCpp::CopyObj(overlay_region, &task->overlay_region, 1, -1);
        }
    }
    catch (const HalconCpp::HException& e) {
        std::promise<bool> failed;
        failed.set_value(false);
        std::cerr << "Failed to copy HALCON objects for async save: "
                  << e.ErrorMessage().TextA() << std::endl;
        return failed.get_future();
    }

    task->target_path = output_path;
    auto future = task->promise.get_future();
    {
        std::lock_guard<std::mutex> lock(mQueueMutex);
        if (mStopAsyncThread.load(std::memory_order_relaxed)) {
            task->promise.set_value(false);
            return future;
        }
        mAsyncSaveQueue.push(task);
        mPendingAsyncOperations.fetch_add(1, std::memory_order_relaxed);
    }
    mQueueCondition.notify_one();
    return future;
}

void HalconProcessor::asyncWorkerFunction() {
    while (true) {
        std::shared_ptr<AsyncSaveTask> task;
        {
            std::unique_lock<std::mutex> lock(mQueueMutex);
            mQueueCondition.wait(lock, [this] {
                return mStopAsyncThread.load(std::memory_order_relaxed) || !mAsyncSaveQueue.empty();
            });

            if (mAsyncSaveQueue.empty()) {
                if (mStopAsyncThread.load(std::memory_order_relaxed)) break;
                continue;
            }

            task = mAsyncSaveQueue.front();
            mAsyncSaveQueue.pop();
        }

        bool success = false;
        std::string message;
        try {
            success = performSaveOperation(task->image, task->overlay_region, task->target_path);
            message = success ? "save succeeded" : "save failed";
        }
        catch (const std::exception& e) {
            success = false;
            message = e.what();
        }
        catch (...) {
            success = false;
            message = "unknown save exception";
        }

        if (!success) mFailedAsyncOperations.fetch_add(1, std::memory_order_relaxed);
        mCompletedAsyncOperations.fetch_add(1, std::memory_order_relaxed);
        task->promise.set_value(success);
        invokeSaveCallback(task->target_path, success, SaveType::PATH_SAVE, message);
        mPendingAsyncOperations.fetch_sub(1, std::memory_order_relaxed);
        mQueueCondition.notify_all();
    }

    mQueueCondition.notify_all();
}

bool HalconProcessor::performSaveOperation(
    const HalconCpp::HObject& image,
    const HalconCpp::HObject& overlay_region,
    const std::string& output_path) {
    try {
        HalconCpp::HObject red;
        HalconCpp::HObject green;
        HalconCpp::HObject blue;
        HalconCpp::HObject overlay_image;
        HalconCpp::HTuple channels;
        HalconCpp::CountChannels(image, &channels);

        if (static_cast<int>(channels[0].I()) == 3) {
            HalconCpp::Decompose3(image, &red, &green, &blue);
        }
        else {
            HalconCpp::HObject color_image;
            HalconCpp::Compose3(image, image, image, &color_image);
            HalconCpp::Decompose3(color_image, &red, &green, &blue);
        }

        if (overlay_region.CountObj() > 0) {
            HalconCpp::PaintRegion(overlay_region, red, &red, 255, "fill");
            HalconCpp::PaintRegion(overlay_region, green, &green, 0, "fill");
            HalconCpp::PaintRegion(overlay_region, blue, &blue, 0, "fill");
        }

        HalconCpp::Compose3(red, green, blue, &overlay_image);
        HalconCpp::WriteImage(overlay_image, "png", 0, output_path.c_str());
        return true;
    }
    catch (const HalconCpp::HException& e) {
        std::cerr << "HALCON image save failed: " << e.ErrorMessage().TextA() << std::endl;
        return false;
    }
}

bool HalconProcessor::gpuMatToHalconHObject(
    const cv::cuda::GpuMat& gpu_mat,
    HalconCpp::HObject& image) {
    try {
        if (gpu_mat.empty()) {
            std::cerr << "gpuMatToHalconHObject: empty input" << std::endl;
            return false;
        }

        cv::Mat cpu_image;
        gpu_mat.download(cpu_image);
        if (cpu_image.empty()) return false;

        if (cpu_image.type() == CV_8UC3) {
            std::vector<cv::Mat> bgr(3);
            cv::split(cpu_image, bgr);

            HalconCpp::HObject red;
            HalconCpp::HObject green;
            HalconCpp::HObject blue;
            HalconCpp::HObject color;
            HalconCpp::GenImage1(
                &red, "byte", cpu_image.cols, cpu_image.rows,
                reinterpret_cast<Hlong>(bgr[2].data));
            HalconCpp::GenImage1(
                &green, "byte", cpu_image.cols, cpu_image.rows,
                reinterpret_cast<Hlong>(bgr[1].data));
            HalconCpp::GenImage1(
                &blue, "byte", cpu_image.cols, cpu_image.rows,
                reinterpret_cast<Hlong>(bgr[0].data));
            HalconCpp::Compose3(red, green, blue, &color);
            HalconCpp::CopyImage(color, &image);
            return true;
        }

        if (cpu_image.type() == CV_8UC1) {
            HalconCpp::HObject gray;
            HalconCpp::GenImage1(
                &gray, "byte", cpu_image.cols, cpu_image.rows,
                reinterpret_cast<Hlong>(cpu_image.data));
            HalconCpp::CopyImage(gray, &image);
            return true;
        }

        std::cerr << "gpuMatToHalconHObject: unsupported OpenCV type "
                  << cpu_image.type() << std::endl;
        return false;
    }
    catch (const HalconCpp::HException& e) {
        std::cerr << "HALCON conversion failed: " << e.ErrorMessage().TextA() << std::endl;
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "OpenCV/HALCON conversion failed: " << e.what() << std::endl;
        return false;
    }
}

bool HalconProcessor::processImage(
    const cv::cuda::GpuMat& input_image,
    std::string& output_image_path,
    std::vector<std::pair<double, double>>& center_points,
    bool enable_save_to_path,
    const std::string& save_path,
    std::uint64_t group_id,
    int index_in_group) {
    center_points.clear();
    output_image_path.clear();

    if (input_image.empty()) return false;
    if (enable_save_to_path && save_path.empty()) enable_save_to_path = false;

    try {
        const std::string output_path = enable_save_to_path
            ? generateSavePathFilename(save_path, group_id, index_in_group)
            : std::string();
        output_image_path = output_path;

        HalconCpp::HObject image;
        if (!gpuMatToHalconHObject(input_image, image)) return false;

        // The effective detection path only depends on vertical texture. Older
        // code also calculated a horizontal branch and several morphology
        // results that were never consumed by the final region selection.
        HalconCpp::HObject mean_image;
        HalconCpp::HObject vertical_texture;
        HalconCpp::HObject enhanced_image;
        HalconCpp::HObject bright_regions;
        HalconCpp::HObject connected_regions;
        HalconCpp::HObject final_regions;

        HalconCpp::MeanImage(image, &mean_image, 6, 6);
        HalconCpp::TextureLaws(mean_image, &vertical_texture, "le", 3, 5);
        HalconCpp::ScaleImage(vertical_texture, &enhanced_image, 3, 0);
        HalconCpp::Threshold(enhanced_image, &bright_regions, 128, 255);
        HalconCpp::Connection(bright_regions, &connected_regions);
        HalconCpp::SelectShape(connected_regions, &final_regions, "area", "and", 100, 2000);

        HalconCpp::HTuple count;
        HalconCpp::CountObj(final_regions, &count);
        const bool defects_detected = count.Length() > 0 && count[0].I() > 0;

        if (defects_detected) {
            HalconCpp::HTuple area;
            HalconCpp::HTuple row;
            HalconCpp::HTuple column;
            HalconCpp::AreaCenter(final_regions, &area, &row, &column);
            center_points.reserve(static_cast<std::size_t>(row.Length()));
            for (Hlong i = 0; i < row.Length(); ++i) {
                center_points.emplace_back(column[i].D(), row[i].D());
            }
        }

        if (enable_save_to_path && !output_path.empty()) {
            if (defects_detected) {
                saveOverlayImageAsync(image, final_regions, output_path);
            }
            else {
                HalconCpp::HObject empty_region;
                HalconCpp::GenEmptyObj(&empty_region);
                saveOverlayImageAsync(image, empty_region, output_path);
            }
        }

        return true;
    }
    catch (const HalconCpp::HException& e) {
        std::cerr << "HALCON Error #" << e.ErrorCode() << " in "
                  << e.ProcName().TextA() << ": " << e.ErrorMessage().TextA() << std::endl;
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "HALCON processing exception: " << e.what() << std::endl;
        return false;
    }
}
