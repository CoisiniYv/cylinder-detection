#pragma once
#include "engineTRT.h"


//#include "./"
#include <string>

class SpeedSam
{

public:
	/// @brief SpeedSam类的构造函数
	/// 
	/// @param encoderPath 编码器模型路径
	/// @param decoderPath 解码器模型路径
	SpeedSam(std::string encoderPath, std::string decoderPath);

	/// @brief SpeedSam类的析构函数
	~SpeedSam();

	/// @brief 基于输入图像和点预测掩码
	/// 
	/// @param image 用于预测的输入图像
	/// @param points 用于掩码预测的点
	/// @param labels 与点关联的标签
	/// @return 包含预测掩码的矩阵
	cv::Mat predict(cv::Mat& image, std::vector<cv::Point> points, std::vector<float> labels);

private:
	// 变量
	float* mFeatures;        ///< 特征数据指针
	float* mMaskInput;       ///< 掩码输入数据指针
	float* mHasMaskInput;    ///< 掩码存在性输入数据指针
	float* mIouPrediction;   ///< IoU预测数据指针
	float* mLowResMasks;     ///< 低分辨率掩码指针

	EngineTRT* mImageEncoder;  ///< 图像编码模块指针
	EngineTRT* mMaskDecoder;   ///< 掩码解码模块指针

	/// @brief 将给定掩码上采样到目标宽度和高度
	/// 
	/// @param mask 待上采样的掩码
	/// @param targetWidth 上采样的目标宽度
	/// @param targetHeight 上采样的目标高度
	/// @param size 掩码尺寸（默认为256）
	void upscaleMask(cv::Mat& mask, int targetWidth, int targetHeight, int size = 256);

	/// @brief 调整输入图像尺寸以匹配模型要求
	/// 
	/// @param img 待调整尺寸的图像
	/// @param modelWidth 模型要求的宽度
	/// @param modelHeight 模型要求的高度
	/// @return 调整尺寸后的图像矩阵
	cv::Mat resizeImage(cv::Mat& img, int modelWidth, int modelHeight);

	/// @brief 从提供的点准备解码器输入
	/// 
	/// @param points 待转换为输入数据的点
	/// @param pointData 点数据数组指针
	/// @param numPoints 点的数量
	/// @param imageWidth 输入图像的宽度
	/// @param imageHeight 输入图像的高度
	void prepareDecoderInput(std::vector<cv::Point>& points, float* pointData, int numPoints, int imageWidth, int imageHeight);
};