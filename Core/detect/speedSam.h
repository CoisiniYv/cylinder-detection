#pragma once

#include "engineTRT.h"

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

// Thin SAM encoder/decoder pipeline built on EngineTRT.
class SpeedSam {
public:
    SpeedSam(const std::string& encoder_path, const std::string& decoder_path);
    ~SpeedSam() = default;

    SpeedSam(const SpeedSam&) = delete;
    SpeedSam& operator=(const SpeedSam&) = delete;

    cv::Mat predict(
        cv::Mat& image,
        const std::vector<cv::Point>& points,
        const std::vector<float>& labels);

private:
    void upscaleMask(cv::Mat& mask, int target_width, int target_height, int size = 256);
    cv::Mat resizeImage(const cv::Mat& image, int model_width, int model_height);
    void prepareDecoderInput(
        const std::vector<cv::Point>& points,
        float* point_data,
        int image_width,
        int image_height);

private:
    std::vector<float> mFeatures;
    std::vector<float> mMaskInput;
    float mHasMaskInput = 0.0f;
    std::vector<float> mIouPrediction;
    std::vector<float> mLowResMasks;

    std::unique_ptr<EngineTRT> mImageEncoder;
    std::unique_ptr<EngineTRT> mMaskDecoder;
};
