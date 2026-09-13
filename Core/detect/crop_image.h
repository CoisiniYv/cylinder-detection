#pragma once

#include "preprocess_options.hpp"

#include <opencv2/core/cuda.hpp>

#include <string>

// Preferred preprocessing entry point used by detection pipelines.
// Processing order: optional ROI crop -> optional stripe removal/denoise ->
// optional QW circular masking -> optional PNG save.
cv::cuda::GpuMat preprocessImage(
    const cv::cuda::GpuMat& image_input,
    const PreprocessOptions& options,
    cv::cuda::Stream& stream = cv::cuda::Stream::Null());

// Compatibility wrapper for older call sites. New code should construct a
// PreprocessOptions object and call preprocessImage() instead of relying on this
// long positional parameter list.
cv::cuda::GpuMat cropImage(
    const cv::cuda::GpuMat& image_input,
    bool enable_four_side_crop = false,
    int x = 0,
    int y = 0,
    int width = 0,
    int height = 0,
    bool is_qw = false,
    int x1_circle = 0,
    int y1_circle = 0,
    int x2_circle = 0,
    int y2_circle = 0,
    int radius = 0,
    const std::string& file_name = "cropped_image",
    const std::string& output_dir = {},
    bool enable_fourier_transform = false,
    int filter_width = 10,
    double attenuation_factor = 0.0001,
    int target_angle = 90,
    int angle_tolerance = 10,
    bool enable_denoising = false,
    float denoise_h = 5.0f,
    float denoise_h_color = 8.0f,
    int denoise_search_window_size = 17,
    int denoise_template_window_size = 11,
    cv::cuda::Stream& stream = cv::cuda::Stream::Null());
