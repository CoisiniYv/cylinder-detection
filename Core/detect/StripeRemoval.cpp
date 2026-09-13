#include "StripeRemoval.h"

#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/photo/cuda.hpp>

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void swapFrequencyQuadrants(
    cv::cuda::GpuMat& real,
    cv::cuda::GpuMat& imaginary,
    cv::cuda::Stream& stream) {
    const int half_width = real.cols / 2;
    const int half_height = real.rows / 2;
    if (half_width <= 0 || half_height <= 0) return;

    cv::cuda::GpuMat q0_real(real, cv::Rect(0, 0, half_width, half_height));
    cv::cuda::GpuMat q1_real(real, cv::Rect(half_width, 0, half_width, half_height));
    cv::cuda::GpuMat q2_real(real, cv::Rect(0, half_height, half_width, half_height));
    cv::cuda::GpuMat q3_real(real, cv::Rect(half_width, half_height, half_width, half_height));

    cv::cuda::GpuMat q0_imag(imaginary, cv::Rect(0, 0, half_width, half_height));
    cv::cuda::GpuMat q1_imag(imaginary, cv::Rect(half_width, 0, half_width, half_height));
    cv::cuda::GpuMat q2_imag(imaginary, cv::Rect(0, half_height, half_width, half_height));
    cv::cuda::GpuMat q3_imag(imaginary, cv::Rect(half_width, half_height, half_width, half_height));

    cv::cuda::GpuMat temp_real;
    cv::cuda::GpuMat temp_imag;

    q0_real.copyTo(temp_real, stream);
    q0_imag.copyTo(temp_imag, stream);
    q3_real.copyTo(q0_real, stream);
    q3_imag.copyTo(q0_imag, stream);
    temp_real.copyTo(q3_real, stream);
    temp_imag.copyTo(q3_imag, stream);

    q1_real.copyTo(temp_real, stream);
    q1_imag.copyTo(temp_imag, stream);
    q2_real.copyTo(q1_real, stream);
    q2_imag.copyTo(q1_imag, stream);
    temp_real.copyTo(q2_real, stream);
    temp_imag.copyTo(q2_imag, stream);
}

} // namespace

int StripeRemoval::detect_strongest_direction(
    const cv::cuda::GpuMat& magnitude_spectrum,
    double inner_ratio,
    double outer_ratio,
    int target_angle,
    int angle_tolerance) {
    if (magnitude_spectrum.empty()) {
        throw std::invalid_argument("magnitude spectrum is empty");
    }
    if (inner_ratio < 0.0 || outer_ratio <= inner_ratio || outer_ratio > 1.0) {
        throw std::invalid_argument("invalid stripe frequency radius ratios");
    }
    if (angle_tolerance < 0 || angle_tolerance > 180) {
        throw std::invalid_argument("invalid stripe angle tolerance");
    }

    cv::Mat spectrum_cpu;
    magnitude_spectrum.download(spectrum_cpu);

    const int rows = spectrum_cpu.rows;
    const int cols = spectrum_cpu.cols;
    const int center_row = rows / 2;
    const int center_col = cols / 2;
    const double min_radius = std::min(rows, cols) * inner_ratio;
    const double max_radius = std::min(rows, cols) * outer_ratio;

    std::vector<double> angular_profile(360, 0.0);
    std::vector<int> angle_counts(360, 0);
    std::vector<double> x_offsets(static_cast<std::size_t>(cols));
    for (int column = 0; column < cols; ++column) {
        x_offsets[static_cast<std::size_t>(column)] = static_cast<double>(column - center_col);
    }

#pragma omp parallel
    {
        std::vector<double> local_profile(360, 0.0);
        std::vector<int> local_counts(360, 0);

#pragma omp for schedule(static)
        for (int row = 0; row < rows; ++row) {
            const double y = static_cast<double>(center_row - row);
            const float* values = spectrum_cpu.ptr<float>(row);
            for (int column = 0; column < cols; ++column) {
                const double x = x_offsets[static_cast<std::size_t>(column)];
                const double radius = std::sqrt(x * x + y * y);
                if (radius < min_radius || radius > max_radius) continue;

                double angle = std::atan2(-y, x) * 180.0 / CV_PI;
                angle = std::fmod(angle + 360.0, 360.0);
                const int angle_index = static_cast<int>(angle);
                if (angle_index < 0 || angle_index >= 360) continue;

                local_profile[static_cast<std::size_t>(angle_index)] += values[column];
                local_counts[static_cast<std::size_t>(angle_index)] += 1;
            }
        }

#pragma omp critical
        {
            for (int angle = 0; angle < 360; ++angle) {
                angular_profile[static_cast<std::size_t>(angle)] +=
                    local_profile[static_cast<std::size_t>(angle)];
                angle_counts[static_cast<std::size_t>(angle)] +=
                    local_counts[static_cast<std::size_t>(angle)];
            }
        }
    }

    for (int angle = 0; angle < 360; ++angle) {
        const int count = angle_counts[static_cast<std::size_t>(angle)];
        if (count > 0) angular_profile[static_cast<std::size_t>(angle)] /= count;
    }

    constexpr double kSigma = 5.0;
    constexpr int kKernelRadius = 15;
    const int normalized_target = (target_angle % 360 + 360) % 360;
    const int start_angle = (normalized_target - angle_tolerance + 360) % 360;
    const int end_angle = (normalized_target + angle_tolerance) % 360;

    auto smoothedValue = [&](int angle_index) {
        double sum = 0.0;
        double weight_sum = 0.0;
        for (int offset = -kKernelRadius; offset <= kKernelRadius; ++offset) {
            const int index = (angle_index + offset + 360) % 360;
            const double weight = std::exp(
                -(offset * offset) / (2.0 * kSigma * kSigma));
            sum += angular_profile[static_cast<std::size_t>(index)] * weight;
            weight_sum += weight;
        }
        return weight_sum > 0.0 ? sum / weight_sum : 0.0;
    };

    int strongest_angle = normalized_target;
    double strongest_value = -std::numeric_limits<double>::infinity();
    auto inspectAngle = [&](int angle) {
        const double value = smoothedValue(angle);
        if (value > strongest_value) {
            strongest_value = value;
            strongest_angle = angle;
        }
    };

    if (start_angle <= end_angle) {
        for (int angle = start_angle; angle <= end_angle; ++angle) inspectAngle(angle);
    }
    else {
        for (int angle = start_angle; angle < 360; ++angle) inspectAngle(angle);
        for (int angle = 0; angle <= end_angle; ++angle) inspectAngle(angle);
    }

    return -strongest_angle;
}

cv::cuda::GpuMat StripeRemoval::remove_image_stripes(
    const cv::cuda::GpuMat& image,
    const std::string& output_path,
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
    if (image.empty()) throw std::invalid_argument("stripe-removal input is empty");
    if (filter_width < 0) throw std::invalid_argument("filter_width must be non-negative");
    if (attenuation_factor < 0.0 || attenuation_factor > 1.0) {
        throw std::invalid_argument("attenuation_factor must be in [0, 1]");
    }
    if (image.channels() != 1 && image.channels() != 3 && image.channels() != 4) {
        throw std::invalid_argument("stripe removal supports only 1, 3 or 4 channel images");
    }

    cv::cuda::GpuMat source = image;
    if (source.channels() == 1) {
        cv::cuda::GpuMat converted;
        cv::cuda::cvtColor(source, converted, cv::COLOR_GRAY2BGR, 0, stream);
        source = converted;
    }
    else if (source.channels() == 4) {
        cv::cuda::GpuMat converted;
        cv::cuda::cvtColor(source, converted, cv::COLOR_BGRA2BGR, 0, stream);
        source = converted;
    }

    cv::cuda::GpuMat source_float;
    source.convertTo(source_float, CV_32FC3, 1.0, 0.0, stream);

    std::vector<cv::cuda::GpuMat> channels;
    cv::cuda::split(source_float, channels, stream);
    std::vector<cv::cuda::GpuMat> real_parts(channels.size());
    std::vector<cv::cuda::GpuMat> imaginary_parts(channels.size());

    cv::cuda::GpuMat magnitude_spectrum;
    cv::Size padded_size;

    for (std::size_t channel_index = 0; channel_index < channels.size(); ++channel_index) {
        const cv::cuda::GpuMat& channel = channels[channel_index];
        const int padded_rows = cv::getOptimalDFTSize(channel.rows);
        const int padded_cols = cv::getOptimalDFTSize(channel.cols);
        padded_size = cv::Size(padded_cols, padded_rows);

        cv::cuda::GpuMat padded(padded_rows, padded_cols, CV_32F);
        padded.setTo(cv::Scalar::all(0), stream);
        channel.copyTo(
            padded(cv::Rect(0, 0, channel.cols, channel.rows)),
            stream);

        cv::cuda::GpuMat planes[2];
        padded.copyTo(planes[0], stream);
        planes[1].create(padded_rows, padded_cols, CV_32F);
        planes[1].setTo(cv::Scalar::all(0), stream);

        cv::cuda::GpuMat complex_image;
        cv::cuda::merge(planes, 2, complex_image, stream);
        cv::cuda::dft(complex_image, complex_image, complex_image.size(), 0, stream);
        cv::cuda::split(complex_image, planes, stream);

        swapFrequencyQuadrants(planes[0], planes[1], stream);
        planes[0].copyTo(real_parts[channel_index], stream);
        planes[1].copyTo(imaginary_parts[channel_index], stream);

        if (channel_index == 0) {
            cv::cuda::magnitude(planes[0], planes[1], magnitude_spectrum, stream);
        }
    }

    // Direction analysis runs on CPU. Synchronize the producer stream before
    // the default-stream download in detect_strongest_direction().
    stream.waitForCompletion();
    const int strongest_angle = detect_strongest_direction(
        magnitude_spectrum, 0.12, 0.18, target_angle, angle_tolerance);

    cv::Mat mask(padded_size, CV_32F, cv::Scalar(1.0f));
    const int center_x = padded_size.width / 2;
    const int center_y = padded_size.height / 2;
    const double theta = strongest_angle * (CV_PI / 180.0);
    const double cosine = std::cos(theta);
    const double sine = std::sin(theta);
    for (int row = 0; row < mask.rows; ++row) {
        float* mask_row = mask.ptr<float>(row);
        const double v = static_cast<double>(center_y - row);
        for (int column = 0; column < mask.cols; ++column) {
            const double u = static_cast<double>(column - center_x);
            const double distance = std::abs(u * sine - v * cosine);
            if (distance <= filter_width && !(row == center_y && column == center_x)) {
                mask_row[column] = static_cast<float>(attenuation_factor);
            }
        }
    }

    cv::cuda::GpuMat mask_gpu;
    mask_gpu.upload(mask, stream);

    std::vector<cv::cuda::GpuMat> processed_channels(channels.size());
    for (std::size_t channel_index = 0; channel_index < channels.size(); ++channel_index) {
        cv::cuda::GpuMat real = real_parts[channel_index];
        cv::cuda::GpuMat imaginary = imaginary_parts[channel_index];
        cv::cuda::multiply(real, mask_gpu, real, 1.0, -1, stream);
        cv::cuda::multiply(imaginary, mask_gpu, imaginary, 1.0, -1, stream);
        swapFrequencyQuadrants(real, imaginary, stream);

        cv::cuda::GpuMat filtered_planes[2]{real, imaginary};
        cv::cuda::GpuMat filtered_complex;
        cv::cuda::merge(filtered_planes, 2, filtered_complex, stream);

        cv::cuda::GpuMat inverse_complex;
        cv::cuda::dft(
            filtered_complex,
            inverse_complex,
            filtered_complex.size(),
            cv::DFT_INVERSE | cv::DFT_SCALE,
            stream);

        cv::cuda::GpuMat time_planes[2];
        cv::cuda::split(inverse_complex, time_planes, stream);
        cv::cuda::GpuMat cropped = time_planes[0](
            cv::Rect(0, 0, channels[channel_index].cols, channels[channel_index].rows));
        cv::cuda::max(cropped, 0.0, cropped, stream);
        cropped.copyTo(processed_channels[channel_index], stream);
    }

    cv::cuda::GpuMat processed_float;
    cv::cuda::merge(processed_channels, processed_float, stream);
    cv::cuda::GpuMat processed_u8;
    processed_float.convertTo(processed_u8, CV_8UC3, 1.0, 0.0, stream);

    cv::cuda::GpuMat result;
    if (enable_denoising) {
        cv::cuda::fastNlMeansDenoisingColored(
            processed_u8,
            result,
            denoise_h,
            denoise_h_color,
            denoise_search_window_size,
            denoise_template_window_size,
            stream);
    }
    else {
        result = processed_u8;
    }

    if (!output_path.empty()) {
        const std::filesystem::path file(output_path);
        if (!file.parent_path().empty()) {
            std::filesystem::create_directories(file.parent_path());
        }
        cv::Mat result_cpu;
        result.download(result_cpu, stream);
        stream.waitForCompletion();
        if (!cv::imwrite(file.string(), result_cpu)) {
            throw std::runtime_error("failed to save stripe-removal output: " + file.string());
        }
    }

    return result;
}
