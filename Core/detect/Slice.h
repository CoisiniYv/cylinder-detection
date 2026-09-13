#pragma once

#include "trtyolo_slice.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <string>
#include <utility>
#include <vector>

using SliceInfo = trtyolo::SliceInfo;

// Stateless image slicing utility used before slice-level YOLO inference.
class ImageSlicer {
public:
    static std::vector<SliceInfo> extractSlices(
        const cv::Mat& original_image,
        const std::vector<std::pair<double, double>>& center_points,
        int distance = 50,
        bool save_slices = false,
        const std::string& save_path = {},
        int slice_width = 100,
        int slice_height = 100);

    static std::vector<SliceInfo> extractSlicesGPU(
        const cv::cuda::GpuMat& original_image,
        const std::vector<std::pair<double, double>>& center_points,
        int distance = 50,
        bool save_slices = false,
        const std::string& save_path = {},
        int slice_width = 100,
        int slice_height = 100,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());

private:
    static SliceInfo extractSingleSlice(
        const cv::Mat& original_image,
        double center_x,
        double center_y,
        int slice_width,
        int slice_height);

    static SliceInfo extractSingleSliceGPU(
        const cv::cuda::GpuMat& original_image,
        double center_x,
        double center_y,
        int slice_width,
        int slice_height,
        cv::cuda::Stream& stream);

    static void saveSliceImage(
        const cv::Mat& slice,
        const std::string& save_path,
        std::size_t center_index,
        std::size_t slice_index);
};
