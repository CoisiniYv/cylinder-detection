#pragma once

#include <opencv2/core/cuda.hpp>

#include <string>

// Frequency-domain stripe suppression. The class is stateless; all tuning
// values are explicit per call so concurrent workers do not share mutable state.
class StripeRemoval {
public:
    int detect_strongest_direction(
        const cv::cuda::GpuMat& magnitude_spectrum,
        double inner_ratio = 0.12,
        double outer_ratio = 0.18,
        int target_angle = 90,
        int angle_tolerance = 10);

    cv::cuda::GpuMat remove_image_stripes(
        const cv::cuda::GpuMat& image,
        const std::string& output_path = {},
        int filter_width = 10,
        double attenuation_factor = 0.0001,
        int target_angle = 90,
        int angle_tolerance = 10,
        bool enable_denoising = false,
        float denoise_h = 3.0f,
        float denoise_h_color = 8.0f,
        int denoise_search_window_size = 17,
        int denoise_template_window_size = 11,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());
};
