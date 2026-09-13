#include "Slice.h"

#include <opencv2/imgcodecs.hpp>

#include <array>
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace {

constexpr std::array<std::pair<int, int>, 5> kSliceOffsets{{
    {0, 0},
    {0, -1},
    {0, 1},
    {-1, 0},
    {1, 0},
}};

void validateSliceArguments(int slice_width, int slice_height, int& distance) {
    if (slice_width <= 0 || slice_height <= 0) {
        throw std::invalid_argument("slice width and height must be positive");
    }
    distance = std::abs(distance);
}

} // namespace

std::vector<SliceInfo> ImageSlicer::extractSlices(
    const cv::Mat& original_image,
    const std::vector<std::pair<double, double>>& center_points,
    int distance,
    bool save_slices,
    const std::string& save_path,
    int slice_width,
    int slice_height) {
    validateSliceArguments(slice_width, slice_height, distance);

    std::vector<SliceInfo> slices;
    if (original_image.empty() || center_points.empty()) return slices;
    if (save_slices && save_path.empty()) save_slices = false;
    if (save_slices) std::filesystem::create_directories(save_path);

    slices.reserve(center_points.size() * kSliceOffsets.size());
    for (std::size_t center_index = 0; center_index < center_points.size(); ++center_index) {
        const auto [center_x, center_y] = center_points[center_index];
        const std::size_t first_slice = slices.size();

        for (const auto [x_direction, y_direction] : kSliceOffsets) {
            slices.emplace_back(extractSingleSlice(
                original_image,
                center_x + static_cast<double>(x_direction * distance),
                center_y + static_cast<double>(y_direction * distance),
                slice_width,
                slice_height));
        }

        if (save_slices) {
            for (std::size_t slice_index = 0; slice_index < kSliceOffsets.size(); ++slice_index) {
                saveSliceImage(
                    slices[first_slice + slice_index].slice,
                    save_path,
                    center_index,
                    slice_index);
            }
        }
    }
    return slices;
}

SliceInfo ImageSlicer::extractSingleSlice(
    const cv::Mat& original_image,
    double center_x,
    double center_y,
    int slice_width,
    int slice_height) {
    const int start_x = static_cast<int>(center_x - slice_width / 2.0);
    const int start_y = static_cast<int>(center_y - slice_height / 2.0);
    const int end_x = start_x + slice_width;
    const int end_y = start_y + slice_height;

    cv::Mat slice = cv::Mat::zeros(slice_height, slice_width, original_image.type());

    const int source_x0 = std::max(start_x, 0);
    const int source_y0 = std::max(start_y, 0);
    const int source_x1 = std::min(end_x, original_image.cols);
    const int source_y1 = std::min(end_y, original_image.rows);

    const int copy_width = source_x1 - source_x0;
    const int copy_height = source_y1 - source_y0;
    if (copy_width > 0 && copy_height > 0) {
        const int target_x = source_x0 - start_x;
        const int target_y = source_y0 - start_y;
        original_image(cv::Rect(source_x0, source_y0, copy_width, copy_height))
            .copyTo(slice(cv::Rect(target_x, target_y, copy_width, copy_height)));
    }

    return SliceInfo(std::move(slice), cv::Point(start_x, start_y));
}

std::vector<SliceInfo> ImageSlicer::extractSlicesGPU(
    const cv::cuda::GpuMat& original_image,
    const std::vector<std::pair<double, double>>& center_points,
    int distance,
    bool save_slices,
    const std::string& save_path,
    int slice_width,
    int slice_height,
    cv::cuda::Stream& stream) {
    validateSliceArguments(slice_width, slice_height, distance);

    std::vector<SliceInfo> slices;
    if (original_image.empty() || center_points.empty()) return slices;
    if (save_slices && save_path.empty()) save_slices = false;
    if (save_slices) std::filesystem::create_directories(save_path);

    slices.reserve(center_points.size() * kSliceOffsets.size());
    for (const auto& center : center_points) {
        for (const auto [x_direction, y_direction] : kSliceOffsets) {
            slices.emplace_back(extractSingleSliceGPU(
                original_image,
                center.first + static_cast<double>(x_direction * distance),
                center.second + static_cast<double>(y_direction * distance),
                slice_width,
                slice_height,
                stream));
        }
    }

    // Every SliceInfo contains a CPU cv::Mat populated by an asynchronous GPU
    // download. The caller immediately hands those Mats to TensorRT, so the
    // ownership boundary must guarantee that all downloads are complete.
    stream.waitForCompletion();

    if (save_slices) {
        for (std::size_t center_index = 0; center_index < center_points.size(); ++center_index) {
            const std::size_t first_slice = center_index * kSliceOffsets.size();
            for (std::size_t slice_index = 0; slice_index < kSliceOffsets.size(); ++slice_index) {
                saveSliceImage(
                    slices[first_slice + slice_index].slice,
                    save_path,
                    center_index,
                    slice_index);
            }
        }
    }

    return slices;
}

SliceInfo ImageSlicer::extractSingleSliceGPU(
    const cv::cuda::GpuMat& original_image,
    double center_x,
    double center_y,
    int slice_width,
    int slice_height,
    cv::cuda::Stream& stream) {
    const int start_x = static_cast<int>(center_x - slice_width / 2.0);
    const int start_y = static_cast<int>(center_y - slice_height / 2.0);
    const int end_x = start_x + slice_width;
    const int end_y = start_y + slice_height;

    cv::cuda::GpuMat slice_gpu(slice_height, slice_width, original_image.type());
    slice_gpu.setTo(cv::Scalar::all(0), stream);

    const int source_x0 = std::max(start_x, 0);
    const int source_y0 = std::max(start_y, 0);
    const int source_x1 = std::min(end_x, original_image.cols);
    const int source_y1 = std::min(end_y, original_image.rows);
    const int copy_width = source_x1 - source_x0;
    const int copy_height = source_y1 - source_y0;

    if (copy_width > 0 && copy_height > 0) {
        const int target_x = source_x0 - start_x;
        const int target_y = source_y0 - start_y;
        cv::cuda::GpuMat source_roi(
            original_image,
            cv::Rect(source_x0, source_y0, copy_width, copy_height));
        cv::cuda::GpuMat target_roi(
            slice_gpu,
            cv::Rect(target_x, target_y, copy_width, copy_height));
        source_roi.copyTo(target_roi, stream);
    }

    cv::Mat slice;
    slice_gpu.download(slice, stream);
    return SliceInfo(std::move(slice), cv::Point(start_x, start_y));
}

void ImageSlicer::saveSliceImage(
    const cv::Mat& slice,
    const std::string& save_path,
    std::size_t center_index,
    std::size_t slice_index) {
    const std::filesystem::path output =
        std::filesystem::path(save_path) /
        ("center_" + std::to_string(center_index) +
         "_slice_" + std::to_string(slice_index) + ".png");
    if (!cv::imwrite(output.string(), slice)) {
        throw std::runtime_error("failed to save slice: " + output.string());
    }
}
