#include "StripeRemoval.h"

#include <limits>

StripeRemoval::StripeRemoval(int fw, double af)
	: filter_width(fw), attenuation_factor(af) {
}

int StripeRemoval::detect_strongest_direction(const cv::cuda::GpuMat& d_magnitude_spectrum,
	double inner_ratio,
	double outer_ratio,
	int target_angle,
	int angle_tolerance) {
	// 将幅度谱下载到CPU进行高效并行分析（OpenMP），总体上对大图性能更稳定
	if (d_magnitude_spectrum.empty()) {
		throw std::invalid_argument("幅度谱为空");
	}
	cv::Mat magnitude_spectrum;
	d_magnitude_spectrum.download(magnitude_spectrum);

	int rows = magnitude_spectrum.rows;
	int cols = magnitude_spectrum.cols;
	int center_row = rows / 2;
	int center_col = cols / 2;

	double min_radius = std::min(rows, cols) * inner_ratio;
	double max_radius = std::min(rows, cols) * outer_ratio;

	std::vector<double> angular_profile(360, 0.0);
	std::vector<int> angle_counts(360, 0);

	// 预计算列偏移以减少内层计算量
	std::vector<double> x_offsets(cols);
	for (int j = 0; j < cols; ++j) x_offsets[j] = static_cast<double>(j - center_col);

	// OpenMP 并行累加角度剖面
#pragma omp parallel
	{
		std::vector<double> local_profile(360, 0.0);
		std::vector<int> local_counts(360, 0);

#pragma omp for schedule(static)
		for (int i = 0; i < rows; ++i) {
			double y = static_cast<double>(center_row - i);
			const float* mag_row = magnitude_spectrum.ptr<float>(i);
			for (int j = 0; j < cols; ++j) {
				double x = x_offsets[j];
				double r = std::sqrt(x * x + y * y);
				if (r >= min_radius && r <= max_radius) {
					double angle = std::atan2(-y, x) * 180.0 / CV_PI;
					angle = std::fmod(angle + 360.0, 360.0);
					int angle_idx = static_cast<int>(angle);
					if (angle_idx >= 0 && angle_idx < 360) {
						float mag_val = mag_row[j];
						local_profile[angle_idx] += static_cast<double>(mag_val);
						local_counts[angle_idx]++;
					}
				}
			}
		}

		// 归并到全局
#pragma omp critical
		{
			for (int k = 0; k < 360; ++k) {
				angular_profile[k] += local_profile[k];
				angle_counts[k] += local_counts[k];
			}
		}
	}

	// 计算平均值
	for (int i = 0; i < 360; ++i) {
		if (angle_counts[i] > 0) {
			angular_profile[i] /= angle_counts[i];
		}
	}

	// 高斯平滑并在目标窗口内查找最强角度（on-the-fly 最大值）
	double sigma = 5.0;
	int kernel_radius = 15;
	int start_angle = (target_angle - angle_tolerance + 360) % 360;
	int end_angle = (target_angle + angle_tolerance) % 360;

	auto smooth_at = [&](int angle_index) -> double {
		double sum = 0.0, weight_sum = 0.0;
		for (int j = -kernel_radius; j <= kernel_radius; ++j) {
			int idx = (angle_index + j + 360) % 360;
			double weight = std::exp(-(j * j) / (2.0 * sigma * sigma));
			sum += angular_profile[idx] * weight;
			weight_sum += weight;
		}
		return sum / weight_sum;
		};

	int strongest_angle = target_angle;
	double max_val = -std::numeric_limits<double>::infinity();
	if (start_angle <= end_angle) {
		for (int i = start_angle; i <= end_angle; ++i) {
			double val = smooth_at(i);
			if (val > max_val) { max_val = val; strongest_angle = i; }
		}
	}
	else {
		for (int i = start_angle; i < 360; ++i) {
			double val = smooth_at(i);
			if (val > max_val) { max_val = val; strongest_angle = i; }
		}
		for (int i = 0; i <= end_angle; ++i) {
			double val = smooth_at(i);
			if (val > max_val) { max_val = val; strongest_angle = i; }
		}
	}

	return -strongest_angle;
}

cv::cuda::GpuMat StripeRemoval::remove_image_stripes(const cv::cuda::GpuMat& d_image,
    const std::string& output_path,
    int filter_width,
    double attenuation_factor,
    int target_angle,
    int angle_tolerance,
    bool enable_denoising,
    float denoise_h,
    float denoise_hColor,
    int denoise_searchWindowSize,
    int denoise_templateWindowSize,
    cv::cuda::Stream& stream) {

	auto total_start_time = std::chrono::high_resolution_clock::now();
	this->filter_width = filter_width;
	this->attenuation_factor = attenuation_factor;

	try {
		// 输入统一为 GpuMat，确保为3通道BGR
		cv::cuda::GpuMat d_src = d_image;
		if (d_src.empty()) {
			throw std::runtime_error("Input GpuMat is empty");
		}
        if (d_src.channels() == 1) {
            cv::cuda::GpuMat d_tmp;
            cv::cuda::cvtColor(d_src, d_tmp, cv::COLOR_GRAY2BGR, 0, stream);
            d_src = d_tmp;
        }
        else if (d_src.channels() == 4) {
            cv::cuda::GpuMat d_tmp;
            cv::cuda::cvtColor(d_src, d_tmp, cv::COLOR_BGRA2BGR, 0, stream);
            d_src = d_tmp;
        }

		// 转换到32F以进行频域处理
		cv::cuda::GpuMat d_src_f32;
        d_src.convertTo(d_src_f32, CV_32FC3, 1.0, 0.0, stream);

		// 拆分通道
		std::vector<cv::cuda::GpuMat> d_channels;
        cv::cuda::split(d_src_f32, d_channels, stream);

		std::vector<cv::cuda::GpuMat> d_real_parts(d_channels.size());
		std::vector<cv::cuda::GpuMat> d_imag_parts(d_channels.size());

		cv::cuda::GpuMat d_magnitude_spectrum;
		cv::Size padded_size;

		for (int c = 0; c < static_cast<int>(d_channels.size()); ++c) {
			cv::cuda::GpuMat d_channel = d_channels[c];

			// 计算最佳DFT尺寸并进行填充
			int m = cv::getOptimalDFTSize(d_channel.rows);
			int n = cv::getOptimalDFTSize(d_channel.cols);
			padded_size = cv::Size(n, m);

			cv::cuda::GpuMat d_padded(m, n, CV_32F);
            d_padded.setTo(cv::Scalar::all(0), stream);
			cv::Rect roi(0, 0, d_channel.cols, d_channel.rows);
            d_channel.copyTo(d_padded(roi), stream);

			// 创建复数矩阵（实部+虚部）
			cv::cuda::GpuMat d_planes[2];
            d_padded.copyTo(d_planes[0], stream);
            d_planes[1].create(m, n, CV_32F);
            d_planes[1].setTo(cv::Scalar::all(0), stream);

			cv::cuda::GpuMat d_complexI;
            cv::cuda::merge(d_planes, 2, d_complexI, stream);

			// 执行GPU版DFT
            cv::cuda::dft(d_complexI, d_complexI, d_complexI.size(), 0, stream);

			// 分离实部和虚部
            cv::cuda::split(d_complexI, d_planes, stream);
			cv::cuda::GpuMat d_real_part = d_planes[0];
			cv::cuda::GpuMat d_imag_part = d_planes[1];

			// 频谱中心化（GPU上执行）
			int cx = d_real_part.cols / 2;
			int cy = d_real_part.rows / 2;

			cv::cuda::GpuMat q0_real(d_real_part, cv::Rect(0, 0, cx, cy));
			cv::cuda::GpuMat q1_real(d_real_part, cv::Rect(cx, 0, cx, cy));
			cv::cuda::GpuMat q2_real(d_real_part, cv::Rect(0, cy, cx, cy));
			cv::cuda::GpuMat q3_real(d_real_part, cv::Rect(cx, cy, cx, cy));

			cv::cuda::GpuMat q0_imag(d_imag_part, cv::Rect(0, 0, cx, cy));
			cv::cuda::GpuMat q1_imag(d_imag_part, cv::Rect(cx, 0, cx, cy));
			cv::cuda::GpuMat q2_imag(d_imag_part, cv::Rect(0, cy, cx, cy));
			cv::cuda::GpuMat q3_imag(d_imag_part, cv::Rect(cx, cy, cx, cy));

			cv::cuda::GpuMat tmp_real, tmp_imag;
            q0_real.copyTo(tmp_real, stream); q0_imag.copyTo(tmp_imag, stream);
            q3_real.copyTo(q0_real, stream); q3_imag.copyTo(q0_imag, stream);
            tmp_real.copyTo(q3_real, stream); tmp_imag.copyTo(q3_imag, stream);

            q1_real.copyTo(tmp_real, stream); q1_imag.copyTo(tmp_imag, stream);
            q2_real.copyTo(q1_real, stream); q2_imag.copyTo(q1_imag, stream);
            tmp_real.copyTo(q2_real, stream); tmp_imag.copyTo(q2_imag, stream);

            d_real_part.copyTo(d_real_parts[c], stream);
            d_imag_part.copyTo(d_imag_parts[c], stream);

			// 第一个通道计算幅度谱用于方向检测（保留在GPU）
			if (c == 0) {
                cv::cuda::magnitude(d_real_part, d_imag_part, d_magnitude_spectrum, stream);
			}
		}

		// 检测最强方向
		int strongest_angle = detect_strongest_direction(d_magnitude_spectrum, 0.12, 0.18, target_angle, angle_tolerance);

		// 预先构建频域掩模（CPU一次性计算，GPU应用）
		cv::Mat mask(padded_size, CV_32F, cv::Scalar(1.0f));
		int center_x = padded_size.width / 2;
		int center_y = padded_size.height / 2;
		double theta_rad = strongest_angle * (CV_PI / 180.0);
		double cos_theta = std::cos(theta_rad);
		double sin_theta = std::sin(theta_rad);
		for (int i = 0; i < mask.rows; ++i) {
			float* mptr = mask.ptr<float>(i);
			double v = static_cast<double>(center_y - i);
			for (int j = 0; j < mask.cols; ++j) {
				double u = static_cast<double>(j - center_x);
				double distance = std::abs(u * sin_theta - v * cos_theta);
				if (distance <= filter_width && !(i == center_y && j == center_x)) {
					mptr[j] = static_cast<float>(attenuation_factor);
				}
			}
		}
		cv::cuda::GpuMat d_mask;
        d_mask.upload(mask, stream);

		// 应用掩模并逆中心化、逆变换
		std::vector<cv::cuda::GpuMat> d_processed_channels(d_channels.size());
		for (int c = 0; c < static_cast<int>(d_channels.size()); ++c) {
			cv::cuda::GpuMat d_real = d_real_parts[c];
			cv::cuda::GpuMat d_imag = d_imag_parts[c];

            cv::cuda::multiply(d_real, d_mask, d_real, 1.0, -1, stream);
            cv::cuda::multiply(d_imag, d_mask, d_imag, 1.0, -1, stream);

			// 逆中心化
			int cx = d_real.cols / 2;
			int cy = d_real.rows / 2;
			cv::cuda::GpuMat q0_real(d_real, cv::Rect(0, 0, cx, cy));
			cv::cuda::GpuMat q1_real(d_real, cv::Rect(cx, 0, cx, cy));
			cv::cuda::GpuMat q2_real(d_real, cv::Rect(0, cy, cx, cy));
			cv::cuda::GpuMat q3_real(d_real, cv::Rect(cx, cy, cx, cy));

			cv::cuda::GpuMat q0_imag(d_imag, cv::Rect(0, 0, cx, cy));
			cv::cuda::GpuMat q1_imag(d_imag, cv::Rect(cx, 0, cx, cy));
			cv::cuda::GpuMat q2_imag(d_imag, cv::Rect(0, cy, cx, cy));
			cv::cuda::GpuMat q3_imag(d_imag, cv::Rect(cx, cy, cx, cy));

			cv::cuda::GpuMat tmp_real, tmp_imag;
            q0_real.copyTo(tmp_real, stream); q0_imag.copyTo(tmp_imag, stream);
            q3_real.copyTo(q0_real, stream); q3_imag.copyTo(q0_imag, stream);
            tmp_real.copyTo(q3_real, stream); tmp_imag.copyTo(q3_imag, stream);

            q1_real.copyTo(tmp_real, stream); q1_imag.copyTo(tmp_imag, stream);
            q2_real.copyTo(q1_real, stream); q2_imag.copyTo(q1_imag, stream);
            tmp_real.copyTo(q2_real, stream); tmp_imag.copyTo(q2_imag, stream);

			// 合并并逆DFT
			cv::cuda::GpuMat d_filtered_planes[2]{ d_real, d_imag };
			cv::cuda::GpuMat d_complex_filtered;
            cv::cuda::merge(d_filtered_planes, 2, d_complex_filtered, stream);

			cv::cuda::GpuMat d_inverse_complex;
            cv::cuda::dft(d_complex_filtered, d_inverse_complex, d_complex_filtered.size(), cv::DFT_INVERSE | cv::DFT_SCALE, stream);

			// 取实部作为时域结果
			cv::cuda::GpuMat d_time_planes[2];
            cv::cuda::split(d_inverse_complex, d_time_planes, stream);
			cv::cuda::GpuMat d_time = d_time_planes[0];

			// 裁剪到原始大小并阈值到非负
			cv::cuda::GpuMat d_processed = d_time(cv::Rect(0, 0, d_channels[c].cols, d_channels[c].rows));
            cv::cuda::max(d_processed, 0.0, d_processed, stream);
            d_processed.copyTo(d_processed_channels[c], stream);
		}

		// 合并通道并转换到8U
		cv::cuda::GpuMat d_processed_image;
        cv::cuda::merge(d_processed_channels, d_processed_image, stream);
		cv::cuda::GpuMat d_output_u8;
        d_processed_image.convertTo(d_output_u8, CV_8UC3, 1.0, 0.0, stream);

		// 非局部均值去噪
		cv::cuda::GpuMat d_final;
        if (enable_denoising) {
            cv::cuda::fastNlMeansDenoisingColored(d_output_u8, d_final, denoise_h, denoise_hColor, denoise_searchWindowSize, denoise_templateWindowSize, stream);
        }
		else {
			d_final = d_output_u8;
		}

		// 可选保存输出
		if (!output_path.empty()) {
			std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());
			cv::Mat out_cpu;
            d_final.download(out_cpu, stream);
			if (!cv::imwrite(output_path, out_cpu)) {
				throw std::runtime_error("Failed to save processed image to: " + output_path);
			}
		}

		return d_final;
	}
	catch (const std::exception&) {
		throw;
	}
}