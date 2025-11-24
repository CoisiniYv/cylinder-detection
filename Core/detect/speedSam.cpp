#include "speedSam.h"
#include "config.h"

SpeedSam::SpeedSam(std::string encoderPath, std::string decoderPath)
{
	// 使用提供的模型路径初始化图像编码器和掩码解码器
	mImageEncoder = new EngineTRT(encoderPath,
		{ "image" },                       // 编码器的输入名称
		{ "image_embeddings" },           // 编码器的输出名称
		false,                             // 不使用动态形状
		true);                             // 使用FP16精度

	mMaskDecoder = new EngineTRT(decoderPath,
		{ "image_embeddings", "point_coords", "point_labels", "mask_input", "has_mask_input" }, // 解码器的输入名称
		{ "iou_predictions", "low_res_masks" }, // 解码器的输出名称
		true,                               // 使用动态形状
		false);                             // 不使用FP16精度

	/*if (encoderPath.find(".onnx") != std::string::npos && g_server_state.config != nullptr) {
		std::string encoderEnginePath = g_server_state.config->modelDir + "\\SAM\\SAM_encoder.engine";
		mImageEncoder->saveEngine(encoderEnginePath);
	}
	if (decoderPath.find(".onnx") != std::string::npos && g_server_state.config != nullptr) {
		std::string decoderEnginePath = g_server_state.config->modelDir + "\\SAM\\SAM_mask_decoder.engine";
		mMaskDecoder->saveEngine(decoderEnginePath);
	}*/

	// 为模型特征和输入分配内存
	mFeatures = new float[HIDDEN_DIM * FEATURE_WIDTH * FEATURE_HEIGHT];
	mMaskInput = new float[HIDDEN_DIM * HIDDEN_DIM];
	mHasMaskInput = new float;             // 掩码输入存在的指针
	mIouPrediction = new float[NUM_LABELS]; // IOU预测输出
	mLowResMasks = new float[NUM_LABELS * HIDDEN_DIM * HIDDEN_DIM]; // 低分辨率掩码输出
}

SpeedSam::~SpeedSam()
{
	// 清理动态分配的内存
	if (mFeatures)      delete[] mFeatures;
	if (mMaskInput)     delete[] mMaskInput;
	if (mIouPrediction) delete[] mIouPrediction;
	if (mLowResMasks)   delete[] mLowResMasks;

	if (mImageEncoder)  delete mImageEncoder;
	if (mMaskDecoder)   delete mMaskDecoder;
}

cv::Mat SpeedSam::predict(cv::Mat& image, std::vector<cv::Point> points, std::vector<float> labels)
{
	// 如果没有提供点，返回空掩码
	if (points.size() == 0) return cv::Mat(image.rows, image.cols, CV_32FC1);

	// 为编码器预处理输入图像
	auto resizedImage = resizeImage(image, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);

	// 使用图像编码器进行推理
	mImageEncoder->setInput(resizedImage);
	mImageEncoder->infer();
	mImageEncoder->getOutput(mFeatures);

	// 为指定点准备解码器输入数据
	auto pointData = new float[2 * points.size()]; // 用于保存缩放后点坐标的数组
	prepareDecoderInput(points, pointData, points.size(), image.cols, image.rows);

	// 使用掩码解码器进行推理
	mMaskDecoder->setInput(mFeatures, pointData, labels.data(), mMaskInput, mHasMaskInput, points.size());
	mMaskDecoder->infer();
	mMaskDecoder->getOutput(mIouPrediction, mLowResMasks);

	// 后处理输出掩码
	cv::Mat imgMask(HIDDEN_DIM, HIDDEN_DIM, CV_32FC1, mLowResMasks);
	upscaleMask(imgMask, image.cols, image.rows); // 上采样到原始图像尺寸

	delete[] pointData; // 清理为点数据动态分配的内存

	return imgMask; // 返回分割后的掩码
}

void SpeedSam::prepareDecoderInput(std::vector<cv::Point>& points, float* pointData, int numPoints, int imageWidth, int imageHeight)
{
	float scale = MODEL_INPUT_WIDTH / std::max(imageWidth, imageHeight); // 计算缩放因子

	// 缩放点坐标
	for (int i = 0; i < numPoints; i++)
	{
		pointData[i * 2] = (float)points[i].x * scale; // X坐标
		pointData[i * 2 + 1] = (float)points[i].y * scale; // Y坐标
	}

	// 初始化掩码输入数据
	for (int i = 0; i < HIDDEN_DIM * HIDDEN_DIM; i++)
	{
		mMaskInput[i] = 0; // 将掩码输入设置为零
	}
	*mHasMaskInput = 0; // 将存在掩码输入设置为假
}

cv::Mat SpeedSam::resizeImage(cv::Mat& img, int inputWidth, int inputHeight)
{
	int w, h;
	float aspectRatio = (float)img.cols / (float)img.rows; // 计算宽高比

	// 在保持宽高比的同时确定新尺寸
	if (aspectRatio >= 1)
	{
		w = inputWidth;
		h = int(inputHeight / aspectRatio);
	}
	else
	{
		w = int(inputWidth * aspectRatio);
		h = inputHeight;
	}

	// 创建新尺寸的图像
	cv::Mat re(h, w, CV_8UC3);
	cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR); // 调整原始图像大小
	cv::Mat out(inputHeight, inputWidth, CV_8UC3, 0.0); // 初始化输出图像
	re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows))); // 将调整大小后的图像复制到输出

	return out; // 返回调整大小后的图像
}

void SpeedSam::upscaleMask(cv::Mat& mask, int targetWidth, int targetHeight, int size)
{
	int limX, limY;
	// 根据目标尺寸计算上采样的限制
	if (targetWidth > targetHeight)
	{
		limX = size;
		limY = size * targetHeight / targetWidth;
	}
	else
	{
		limX = size * targetWidth / targetHeight;
		limY = size;
	}

	// 将掩码调整到目标尺寸
	cv::resize(mask(cv::Rect(0, 0, limX, limY)), mask, cv::Size(targetWidth, targetHeight));
}