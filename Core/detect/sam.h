#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "speedSam.h"
#include "trtyolo_slice.hpp"
#include "../Utils/Log.hpp"


#include <memory>
#include <vector>
#include <string>

// SamSegmenter: 使用 SpeedSam 进行初始化、推理与可视化
class SamSegmenter {
public:
	SamSegmenter();
	~SamSegmenter();

	// 导入 SAM 模型（Encoder/Decoder 的 TensorRT 引擎）
	// 返回是否初始化成功
	bool init(const std::string& encoderEnginePath, const std::string& decoderEnginePath);

	// 推理：基于 SliceDetector 的检测结果，将每个检测框作为框提示，输出每个框对应的 mask
	// image: 原始图像（CPU Mat，BGR）
	// detections: SliceDetector 生成的检测结果（包含 left/top/right/bottom）。会在本函数内根据掩码过滤后就地修改（同步过滤）。
	// area_threshold_px: 面积阈值（单位：像素）。仅当 > 0 时生效。
	// diameter_threshold_px: 长度阈值（单位：像素，按“最小外接圆直径”计算）。仅当 > 0 时生效。
	// 返回：与过滤后的检测框一一对应的 mask 列表（每个 mask 与原图同尺寸，类型 CV_32FC1）
	std::vector<cv::Mat> inferFromDetections(cv::Mat& image,
		trtyolo::DetectRes& detections,
		float area_threshold_px = 0.0f,
		float diameter_threshold_px = 0.0f);

	// 可视化：将所有 mask 以透明红色覆盖到原图上，并绘制检测框
	// gpuImage: 原图（GPU Mat）
	// detections: 检测结果（用于绘制框）
	// masks: inferFromDetections 的输出
	// 返回：可视化后的 CPU Mat 图像
	cv::Mat visualize(const cv::cuda::GpuMat& gpuImage,
		std::string& outputImagePath,
		const trtyolo::DetectRes& detections,
		const std::vector<cv::Mat>& masks,
		bool enableSaveToPath = false,
		const std::string& savePath = std::string(),
		std::uint64_t group_id = 0,
		std::int8_t index_in_group = -1);

	// 获取最近一次 inferFromDetections 计算得到的每个检测的像素面积与直径（长度近似），与返回的 masks/过滤后的 detections 一一对齐
	const std::vector<double>& lastAreasPx() const { return mLastAreasPx; }
	const std::vector<double>& lastDiametersPx() const { return mLastDiametersPx; }

private:
	std::unique_ptr<SpeedSam> mSam;

	// 最近一次推理的度量（与过滤后的 detections 对齐）
	std::vector<double> mLastAreasPx;
	std::vector<double> mLastDiametersPx;

	// 将多个 float 掩码组合成一个二值掩码（0/255）
	cv::Mat combineBinaryMask(const std::vector<cv::Mat>& masks, float thresh = 0.5f);

	// 计算掩码的最小外接圆直径（单位：像素）。mask 为 CV_32FC1，内部以阈值 0.5 二值化后再计算。
	static float calculateMaskCircleDiameter(const cv::Mat& mask, float bin_thresh = 0.5f);
};