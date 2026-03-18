#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/photo/cuda.hpp>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <filesystem>
#include <omp.h>
#include <fstream>

class StripeRemoval {
private:
	int filter_width;
	double attenuation_factor;

public:
	StripeRemoval(int fw = 3, double af = 0.0001);
	int detect_strongest_direction(const cv::cuda::GpuMat& d_magnitude_spectrum,
		double inner_ratio = 0.12,
		double outer_ratio = 0.18,
		int target_angle = 90,
		int angle_tolerance = 10);
    cv::cuda::GpuMat remove_image_stripes(const cv::cuda::GpuMat& d_image,
        const std::string& output_path = "",
        int filter_width = 10,
        double attenuation_factor = 0.0001,
        int target_angle = 90,
        int angle_tolerance = 10,
        bool enable_denoising = false,
        float denoise_h = 3.0f,
        float denoise_hColor = 8.0f,
        int denoise_searchWindowSize = 17,
        int denoise_templateWindowSize = 11,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());
};