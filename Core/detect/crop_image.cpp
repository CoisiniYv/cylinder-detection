#include "crop_image.h"

#include "StripeRemoval.h"

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <stdexcept>
#include <vector>

cv::cuda::GpuMat cropImage(
    const cv::cuda::GpuMat& image_input,
    bool enable_four_side_crop,
    int x,
    int y,
    int width,
    int height,
    bool is_qw,
    int x1_circle,
    int y1_circle,
    int x2_circle,
    int y2_circle,
    int radius,
    const std::string& file_name,
    const std::string& output_dir,
    bool enable_fourier_transform,
    int filter_width,
    double attenuation_factor,
    int target_angle,
    int angle_tolerance,
    bool enable_denoising,
    float denoise_h,
    float denoise_h_color,
    int denoise_search_window_size,
    int denoise_template_window_size,
    cv::cuda::Stream& stream) {
    if (image_input.empty()) throw std::invalid_argument("preprocess input image is empty");

    cv::cuda::GpuMat processed;
    if (enable_four_side_crop) {
        if (x < 0 || y < 0 || width <= 0 || height <= 0 ||
            x + width > image_input.cols || y + height > image_input.rows) {
            throw std::out_of_range("ROI crop is outside the input image");
        }
        cv::cuda::GpuMat roi(image_input, cv::Rect(x, y, width, height));
        roi.copyTo(processed, stream);
    }
    else {
        image_input.copyTo(processed, stream);
    }

    if (enable_fourier_transform) {
        StripeRemoval stripe_removal;
        processed = stripe_removal.remove_image_stripes(
            processed,
            {},
            filter_width,
            attenuation_factor,
            target_angle,
            angle_tolerance,
            enable_denoising,
            denoise_h,
            denoise_h_color,
            denoise_search_window_size,
            denoise_template_window_size,
            stream);
    }

    if (is_qw) {
        if (x1_circle < 0 || y1_circle < 0 || x2_circle < 0 || y2_circle < 0 || radius <= 0) {
            throw std::invalid_argument("QW mask requires valid circle centers and radius");
        }

        const int left_center_x = enable_four_side_crop ? x1_circle - x : x1_circle;
        const int left_center_y = enable_four_side_crop ? y1_circle - y : y1_circle;
        const int right_center_x = enable_four_side_crop ? x2_circle - x : x2_circle;
        const int right_center_y = enable_four_side_crop ? y2_circle - y : y2_circle;

        cv::Mat mask(processed.size(), CV_8UC1, cv::Scalar(255));
        cv::circle(mask, cv::Point(left_center_x, left_center_y), radius, cv::Scalar(0), -1);
        cv::circle(mask, cv::Point(right_center_x, right_center_y), radius, cv::Scalar(0), -1);

        cv::cuda::GpuMat mask_gpu;
        mask_gpu.upload(mask, stream);
        cv::cuda::GpuMat mask_float;
        mask_gpu.convertTo(mask_float, CV_32F, 1.0 / 255.0, 0.0, stream);

        const int channels = processed.channels();
        cv::cuda::GpuMat processed_float;
        processed.convertTo(processed_float, CV_MAKETYPE(CV_32F, channels), 1.0, 0.0, stream);

        if (channels == 1) {
            cv::cuda::multiply(processed_float, mask_float, processed_float, 1.0, -1, stream);
        }
        else {
            std::vector<cv::cuda::GpuMat> mask_channels(
                static_cast<std::size_t>(channels), mask_float);
            cv::cuda::GpuMat expanded_mask;
            cv::cuda::merge(mask_channels, expanded_mask, stream);
            cv::cuda::multiply(processed_float, expanded_mask, processed_float, 1.0, -1, stream);
        }

        processed_float.convertTo(processed, processed.type(), 1.0, 0.0, stream);
    }

    if (!output_dir.empty()) {
        const std::filesystem::path directory(output_dir);
        std::filesystem::create_directories(directory);
        const std::filesystem::path output_file = directory / (file_name + ".png");

        cv::Mat output_cpu;
        processed.download(output_cpu, stream);
        stream.waitForCompletion();
        if (!cv::imwrite(output_file.string(), output_cpu)) {
            throw std::runtime_error("failed to save preprocessed image: " + output_file.string());
        }
    }

    return processed;
}
