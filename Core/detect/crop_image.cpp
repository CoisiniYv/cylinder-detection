#include "crop_image.h"

#include "StripeRemoval.h"

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <stdexcept>
#include <vector>

cv::cuda::GpuMat preprocessImage(
    const cv::cuda::GpuMat& image_input,
    const PreprocessOptions& options,
    cv::cuda::Stream& stream) {
    if (image_input.empty()) throw std::invalid_argument("preprocess input image is empty");

    cv::cuda::GpuMat processed;
    if (options.crop.enabled) {
        const auto& crop = options.crop;
        if (crop.x < 0 || crop.y < 0 || crop.width <= 0 || crop.height <= 0 ||
            crop.x + crop.width > image_input.cols ||
            crop.y + crop.height > image_input.rows) {
            throw std::out_of_range("ROI crop is outside the input image");
        }
        cv::cuda::GpuMat roi(
            image_input,
            cv::Rect(crop.x, crop.y, crop.width, crop.height));
        roi.copyTo(processed, stream);
    }
    else {
        image_input.copyTo(processed, stream);
    }

    if (options.stripe.enabled) {
        const auto& stripe = options.stripe;
        StripeRemoval stripe_removal;
        processed = stripe_removal.remove_image_stripes(
            processed,
            {},
            stripe.filter_width,
            stripe.attenuation_factor,
            stripe.target_angle,
            stripe.angle_tolerance,
            stripe.denoise,
            stripe.denoise_h,
            stripe.denoise_h_color,
            stripe.denoise_search_window,
            stripe.denoise_template_window,
            stream);
    }

    if (options.qw_mask.enabled) {
        const auto& mask_options = options.qw_mask;
        if (mask_options.x1 < 0 || mask_options.y1 < 0 ||
            mask_options.x2 < 0 || mask_options.y2 < 0 ||
            mask_options.radius <= 0) {
            throw std::invalid_argument("QW mask requires valid circle centers and radius");
        }

        const int crop_x = options.crop.enabled ? options.crop.x : 0;
        const int crop_y = options.crop.enabled ? options.crop.y : 0;
        const int left_center_x = mask_options.x1 - crop_x;
        const int left_center_y = mask_options.y1 - crop_y;
        const int right_center_x = mask_options.x2 - crop_x;
        const int right_center_y = mask_options.y2 - crop_y;

        cv::Mat mask(processed.size(), CV_8UC1, cv::Scalar(255));
        cv::circle(
            mask,
            cv::Point(left_center_x, left_center_y),
            mask_options.radius,
            cv::Scalar(0),
            -1);
        cv::circle(
            mask,
            cv::Point(right_center_x, right_center_y),
            mask_options.radius,
            cv::Scalar(0),
            -1);

        cv::cuda::GpuMat mask_gpu;
        mask_gpu.upload(mask, stream);
        cv::cuda::GpuMat mask_float;
        mask_gpu.convertTo(mask_float, CV_32F, 1.0 / 255.0, 0.0, stream);

        const int channels = processed.channels();
        cv::cuda::GpuMat processed_float;
        processed.convertTo(
            processed_float,
            CV_MAKETYPE(CV_32F, channels),
            1.0,
            0.0,
            stream);

        if (channels == 1) {
            cv::cuda::multiply(processed_float, mask_float, processed_float, 1.0, -1, stream);
        }
        else {
            std::vector<cv::cuda::GpuMat> mask_channels(
                static_cast<std::size_t>(channels),
                mask_float);
            cv::cuda::GpuMat expanded_mask;
            cv::cuda::merge(mask_channels, expanded_mask, stream);
            cv::cuda::multiply(
                processed_float,
                expanded_mask,
                processed_float,
                1.0,
                -1,
                stream);
        }

        processed_float.convertTo(processed, processed.type(), 1.0, 0.0, stream);
    }

    if (!options.save.output_dir.empty()) {
        const std::filesystem::path directory(options.save.output_dir);
        std::filesystem::create_directories(directory);
        const std::filesystem::path output_file =
            directory / (options.save.file_name + ".png");

        cv::Mat output_cpu;
        processed.download(output_cpu, stream);
        stream.waitForCompletion();
        if (!cv::imwrite(output_file.string(), output_cpu)) {
            throw std::runtime_error(
                "failed to save preprocessed image: " + output_file.string());
        }
    }

    return processed;
}

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
    PreprocessOptions options;
    options.crop.enabled = enable_four_side_crop;
    options.crop.x = x;
    options.crop.y = y;
    options.crop.width = width;
    options.crop.height = height;

    options.qw_mask.enabled = is_qw;
    options.qw_mask.x1 = x1_circle;
    options.qw_mask.y1 = y1_circle;
    options.qw_mask.x2 = x2_circle;
    options.qw_mask.y2 = y2_circle;
    options.qw_mask.radius = radius;

    options.save.file_name = file_name;
    options.save.output_dir = output_dir;

    options.stripe.enabled = enable_fourier_transform;
    options.stripe.filter_width = filter_width;
    options.stripe.attenuation_factor = attenuation_factor;
    options.stripe.target_angle = target_angle;
    options.stripe.angle_tolerance = angle_tolerance;
    options.stripe.denoise = enable_denoising;
    options.stripe.denoise_h = denoise_h;
    options.stripe.denoise_h_color = denoise_h_color;
    options.stripe.denoise_search_window = denoise_search_window_size;
    options.stripe.denoise_template_window = denoise_template_window_size;

    return preprocessImage(image_input, options, stream);
}
