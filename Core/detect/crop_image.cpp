#include "crop_image.h"
#include "StripeRemoval.h"

#include <stdexcept>

cv::cuda::GpuMat cropImage(
    const cv::cuda::GpuMat& d_imageInput,
    bool enableFourSideCrop,
    int x,
    int y,
    int width,
	int height,
	bool isQw,
	int x1Circle,
	int y1Circle,
	int x2Circle,
	int y2Circle,
	int radius,
	const std::string& fileName,
	const std::string& outputDir,
	bool enableFourierTransform,
	int filter_width,
	double attenuation_factor,
	int target_angle,
	int angle_tolerance,
	bool enable_denoising,
	float denoise_h,
	float denoise_hColor,
    int denoise_searchWindowSize,
    int denoise_templateWindowSize,
    cv::cuda::Stream& stream)
{
	// 输入检查
	if (d_imageInput.empty()) {
		throw std::invalid_argument("输入图像为空");
	}

	int imgCols = d_imageInput.cols;
	int imgRows = d_imageInput.rows;

	cv::cuda::GpuMat d_cropped;

    if (enableFourSideCrop) {
        if (x < 0 || y < 0 || width <= 0 || height <= 0 ||
            (x + width) > imgCols || (y + height) > imgRows) {
            throw std::out_of_range("裁剪区域超出图像范围");
        }
        cv::Rect roi(x, y, width, height);
        cv::cuda::GpuMat roi_view(d_imageInput, roi);
        roi_view.copyTo(d_cropped, stream);
    }
    else {
        d_imageInput.copyTo(d_cropped, stream);
    }

	// 傅里叶条纹去除
	if (enableFourierTransform) {
		StripeRemoval sr;
		d_cropped = sr.remove_image_stripes(
			d_cropped,
			"",
			filter_width,     // filter_width
			attenuation_factor, // attenuation_factor
			target_angle,     // target_angle
			angle_tolerance,  // angle_tolerance
			enable_denoising, // enable_denoising
			denoise_h,       // denoise_h
			denoise_hColor,  // denoise_hColor
			denoise_searchWindowSize, // denoise_searchWindowSize
			denoise_templateWindowSize,
			stream
		);
	}

	if (isQw) {
		if (x1Circle < 0 || y1Circle < 0 || x2Circle < 0 || y2Circle < 0 || radius <= 0) {
			throw std::invalid_argument("裁剪QW区域需要提供有效的x1Circle, y1Circle, x2Circle, y2Circle和radius参数");
		}

		// 计算圆相对裁剪区域的坐标
		const int leftCenterX = enableFourSideCrop ? (x1Circle - x) : x1Circle;
		const int leftCenterY = enableFourSideCrop ? (y1Circle - y) : y1Circle;
		const int rightCenterX = enableFourSideCrop ? (x2Circle - x) : x2Circle;
		const int rightCenterY = enableFourSideCrop ? (y2Circle - y) : y2Circle;

		// 构建二值掩模（CPU），上传后在GPU进行逐元素乘法实现遮罩
		cv::Mat mask(d_cropped.size(), CV_8UC1, cv::Scalar(255));
		cv::circle(mask, cv::Point(leftCenterX, leftCenterY), radius, cv::Scalar(0), -1);
		cv::circle(mask, cv::Point(rightCenterX, rightCenterY), radius, cv::Scalar(0), -1);

        cv::cuda::GpuMat d_mask;
        d_mask.upload(mask, stream);

		// 根据图像通道数扩展掩模并进行乘法
		cv::cuda::GpuMat d_mask_f;
        d_mask.convertTo(d_mask_f, CV_32F, 1.0 / 255.0, 0.0, stream);

		int ch = d_cropped.channels();
		int float_type = CV_MAKETYPE(CV_32F, ch);
        cv::cuda::GpuMat d_cropped_f;
        d_cropped.convertTo(d_cropped_f, float_type, 1.0, 0.0, stream);

        if (ch == 1) {
            cv::cuda::multiply(d_cropped_f, d_mask_f, d_cropped_f, 1.0, -1, stream);
        }
        else {
            std::vector<cv::cuda::GpuMat> channelsMask(ch, d_mask_f);
            cv::cuda::GpuMat d_mask_fN;
            cv::cuda::merge(channelsMask, d_mask_fN, stream);
            cv::cuda::multiply(d_cropped_f, d_mask_fN, d_cropped_f, 1.0, -1, stream);
        }

        d_cropped_f.convertTo(d_cropped, d_cropped.type(), 1.0, 0.0, stream);
    }

	// 保存到磁盘（如需要）
    if (!outputDir.empty()) {
        std::filesystem::path outputPath(outputDir);
        if (!std::filesystem::exists(outputPath)) {
            std::filesystem::create_directories(outputPath);
        }
        std::string fullOutputPath = (outputPath / (fileName + ".png")).string();
        cv::Mat h_output;
        d_cropped.download(h_output, stream);
        cv::imwrite(fullOutputPath, h_output);
    }

	return d_cropped;
}