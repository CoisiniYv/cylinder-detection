#pragma once

#include "speedSam.h"
#include "trtyolo_slice.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class SamSegmenter {
public:
    SamSegmenter() = default;
    ~SamSegmenter() = default;

    SamSegmenter(const SamSegmenter&) = delete;
    SamSegmenter& operator=(const SamSegmenter&) = delete;

    bool init(const std::string& encoder_engine_path, const std::string& decoder_engine_path);

    std::vector<cv::Mat> inferFromDetections(
        cv::Mat& image,
        trtyolo::DetectRes& detections,
        float area_threshold_px = 0.0f,
        float diameter_threshold_px = 0.0f);

    cv::Mat visualize(
        const cv::cuda::GpuMat& gpu_image,
        std::string& output_image_path,
        const trtyolo::DetectRes& detections,
        const std::vector<cv::Mat>& masks,
        bool enable_save_to_path = false,
        const std::string& save_path = {},
        std::uint64_t group_id = 0,
        int index_in_group = -1);

    const std::vector<double>& lastAreasPx() const noexcept { return mLastAreasPx; }
    const std::vector<double>& lastDiametersPx() const noexcept { return mLastDiametersPx; }

private:
    static cv::Mat combineBinaryMask(const std::vector<cv::Mat>& masks, float threshold = 0.5f);
    static float calculateMaskCircleDiameter(const cv::Mat& mask, float threshold = 0.5f);

private:
    std::unique_ptr<SpeedSam> mSam;
    std::vector<double> mLastAreasPx;
    std::vector<double> mLastDiametersPx;
};
