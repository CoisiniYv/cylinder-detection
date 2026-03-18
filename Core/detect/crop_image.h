#pragma once

#include <opencv2/core/cuda.hpp>
#include <string>

/**
 * @brief 从图像中裁剪指定区域
 *
 * @param d_imageInput 输入图像 (cv::cuda::GpuMat)
 * @param enableFourSideCrop 是否启用四边裁剪，可选，默认false
 * @param x 裁剪区域的左上角 x 坐标，可选，默认0
 * @param y 裁剪区域的左上角 y 坐标，可选，默认0
 * @param width 裁剪区域的宽度，可选，默认0
 * @param height 裁剪区域的高度，可选，默认0
 * @param isQw 是否裁剪QW区域，可选，默认false
 * @param x1Circle 第一个圆中心点对于原图的x位置，可选，默认0
 * @param y1Circle 第一个圆中心点对于原图的y位置，可选，默认0
 * @param x2Circle 第二个圆中心点对于原图的x位置，可选，默认0
 * @param y2Circle 第二个圆中心点对于原图的y位置，可选，默认0
 * @param radius 圆裁剪区域的半径，可选，默认0
 * @param fileName 文件名，可选，默认"cropped_image"
 * @param outputDir 输出目录，可选，默认空字符串
 * @param enableFourierTransform 是否启用傅里叶变换条纹去除，可选，默认false
 * @param filter_width 滤波器宽度，可选，默认值由StripeRemoval类决定
 * @param attenuation_factor 衰减因子，可选，默认值由StripeRemoval类决定
 * @param target_angle 目标角度，可选，默认值由StripeRemoval类决定
 * @param angle_tolerance 角度容差，可选，默认值由StripeRemoval类决定
 * @param enable_denoising 是否启用去噪，可选，默认false
 * @param denoise_h 去噪参数h，可选，默认值由StripeRemoval类决定
 * @param denoise_hColor 去噪参数hColor，可选，默认值由StripeRemoval类决定
 * @param denoise_searchWindowSize 去噪搜索窗口大小，可选，默认值由StripeRemoval类决定
 * @param denoise_templateWindowSize 去噪模板窗口大小，可选，默认值由StripeRemoval类决定
 * @return cv::cuda::GpuMat 裁剪后的图像
 */
cv::cuda::GpuMat cropImage(
    const cv::cuda::GpuMat& d_imageInput,
    bool enableFourSideCrop = false,
    int x = 0,
    int y = 0,
    int width = 0,
    int height = 0,
    bool isQw = false,
    int x1Circle = 0,
    int y1Circle = 0,
    int x2Circle = 0,
    int y2Circle = 0,
    int radius = 0,
    const std::string& fileName = "cropped_image",
    const std::string& outputDir = "",
    bool enableFourierTransform = false,
    int filter_width = 10,
    double attenuation_factor = 0.5,
    int target_angle = 90,
    int angle_tolerance = 15,
    bool enable_denoising = false,
    float denoise_h = 3.0f,
    float denoise_hColor = 3.0f,
    int denoise_searchWindowSize = 21,
    int denoise_templateWindowSize = 7,
    cv::cuda::Stream& stream = cv::cuda::Stream::Null()
);