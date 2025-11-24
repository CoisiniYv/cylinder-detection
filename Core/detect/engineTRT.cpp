#include "engineTRT.h"
#include "logging.h"
#include "cuda_utils.h"
#include "config.h"

#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

static Logger gLogger;

// 辅助函数：将Dims64转换为Dims（保持头文件不变）
static nvinfer1::Dims toDims32(const nvinfer1::Dims64& d64)
{
	nvinfer1::Dims d{};
	d.nbDims = static_cast<int>(d64.nbDims);
	for (int i = 0; i < d.nbDims && i < nvinfer1::Dims::MAX_DIMS; ++i)
	{
		d.d[i] = static_cast<int>(d64.d[i]);
	}
	return d;
}

// 辅助函数：根据Dims64计算大小
static size_t getSizeByDim64(const nvinfer1::Dims64& dims)
{
	size_t size = 1;
	for (int i = 0; i < dims.nbDims; ++i)
	{
		const int64_t di = dims.d[i];
		if (di == -1)
			size *= MAX_NUM_PROMPTS;
		else
			size *= static_cast<size_t>(di);
	}
	return size;
}

std::string getFileExtension(const std::string& filePath) {
	// 在文件路径中查找最后一个点的位置
	size_t dotPos = filePath.find_last_of(".");
	// 如果找到点，则提取点后面的子串作为文件扩展名并返回
	if (dotPos != std::string::npos) {
		return filePath.substr(dotPos + 1);
	}
	// 如果未找到扩展名，则返回空字符串
	return "";
}

EngineTRT::EngineTRT(std::string modelPath, std::vector<std::string> inputNames, std::vector<std::string> outputNames, bool isDynamicShape, bool isFP16) {
	// 检查模型文件是否具有".onnx"扩展名
	if (getFileExtension(modelPath) == "onnx") {
		// 如果文件是ONNX模型，则使用提供的参数构建引擎
		build(modelPath, inputNames, outputNames, isDynamicShape, isFP16);
	}
	else {
		// 如果文件不是ONNX模型，则反序列化现有引擎
		deserializeEngine(modelPath, inputNames, outputNames);
	}
}

EngineTRT::~EngineTRT() {
	// 释放CUDA流
	cudaStreamDestroy(mCudaStream);
	// 释放为推理分配的GPU缓冲区
	for (int i = 0; i < mGpuBuffers.size(); i++)
		CUDA_CHECK(cudaFree(mGpuBuffers[i]));
	// 释放CPU缓冲区
	for (int i = 0; i < mCpuBuffers.size(); i++)
		delete[] mCpuBuffers[i];

	// 清理并销毁TensorRT引擎组件
	delete mContext;  // 销毁执行上下文
	delete mEngine;   // 销毁引擎
	delete mRuntime;  // 销毁运行时
}

void EngineTRT::build(std::string onnxPath, std::vector<std::string> inputNames, std::vector<std::string> outputNames, bool isDynamicShape, bool isFP16)
{
	// 检查ONNX文件是否存在。如果不存在，打印错误消息并返回。
	if (!std::filesystem::exists(onnxPath)) {
		std::cerr << "ONNX file not found: " << onnxPath << std::endl;
		return;  // 如果ONNX文件缺失，提前退出
	}

	// 创建一个推理构建器，用于构建TensorRT引擎。
	auto builder = nvinfer1::createInferBuilder(gLogger);
	assert(builder != nullptr);  // 确保构建器成功创建

	// 使用显式批处理大小，这是ONNX模型所需的。
	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
	assert(network != nullptr);  // 确保网络成功创建

	// 创建一个构建器配置对象，用于设置选项，如FP16精度。
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	assert(config != nullptr);  // 确保配置成功创建

	// 如果需要动态形状支持，配置优化配置文件。
	if (isDynamicShape) // 仅设计用于NanoSAM掩码解码器
	{
		// 为动态输入形状创建一个优化配置文件。
		auto profile = builder->createOptimizationProfile();

		// 为第一个输入设置最小、最优和最大维度。
		profile->setDimensions(inputNames[1].c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3{ 1, 1, 2 });
		profile->setDimensions(inputNames[1].c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3{ 1, 1, 2 });
		profile->setDimensions(inputNames[1].c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3{ 1, 10, 2 });

		// 为第二个输入设置最小、最优和最大维度。
		profile->setDimensions(inputNames[2].c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{ 1, 1 });
		profile->setDimensions(inputNames[2].c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2{ 1, 1 });
		profile->setDimensions(inputNames[2].c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2{ 1, 10 });

		// 将优化配置文件添加到构建器配置中。
		config->addOptimizationProfile(profile);
	}

	// 如果指定，启用FP16模式。
	if (isFP16)
	{
		config->setFlag(nvinfer1::BuilderFlag::kFP16);  // 使用混合精度以获得更快的推理速度
	}

	// 创建一个解析器，将ONNX模型转换为TensorRT网络。
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
	assert(parser != nullptr);  // 确保解析器成功创建

	// 从指定文件解析ONNX模型。
	bool parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(gLogger.getReportableSeverity()));

	// 确保用于性能分析的CUDA流有效。
	assert(mCudaStream != nullptr);

	// 将构建的网络序列化为二进制计划以便执行。
	nvinfer1::IHostMemory* plan{ builder->buildSerializedNetwork(*network, *config) };
	assert(plan != nullptr);  // 确保网络成功序列化

	// 创建一个运行时对象，用于反序列化引擎。
	mRuntime = nvinfer1::createInferRuntime(gLogger);
	assert(mRuntime != nullptr);  // 确保运行时成功创建

	// 反序列化序列化的计划以创建执行引擎（TensorRT 10 API）。
	mEngine = mRuntime->deserializeCudaEngine(plan->data(), plan->size());
	assert(mEngine != nullptr);  // 确保引擎成功反序列化

	// 创建一个执行上下文以运行推理。
	mContext = mEngine->createExecutionContext();
	assert(mContext != nullptr);  // 确保上下文成功创建

	// 清理资源。
	delete network;
	delete config;
	delete parser;
	delete plan;

	// 使用输入和输出名称初始化引擎。
	initialize(inputNames, outputNames);
}

void EngineTRT::saveEngine(const std::string& engineFilePath) {
	if (mEngine) {
		// 将引擎序列化为二进制格式。
		nvinfer1::IHostMemory* serializedEngine = mEngine->serialize();
		std::ofstream engineFile(engineFilePath, std::ios::binary);
		if (engineFile) {
			// 将序列化的引擎数据写入指定文件。
			engineFile.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
			std::cout << "Serialized engine saved to " << engineFilePath << std::endl;
		}
		// 在TensorRT 10中，对象通过C++析构函数管理。
		delete serializedEngine;  // 释放序列化的引擎内存
	}
}

void EngineTRT::deserializeEngine(std::string engine_name, std::vector<std::string> inputNames, std::vector<std::string> outputNames)
{
	// 以二进制模式打开引擎文件。
	std::ifstream file(engine_name, std::ios::binary);
	if (!file.good()) {
		std::cerr << "read " << engine_name << " error!" << std::endl;
		assert(false);  // 如果无法打开文件，触发断言失败
	}

	// 确定文件大小并读取序列化的引擎数据。
	size_t size = 0;
	file.seekg(0, file.end);
	size = file.tellg();
	file.seekg(0, file.beg);
	char* serializedEngine = new char[size];
	assert(serializedEngine);  // 确保内存分配成功
	file.read(serializedEngine, size);
	file.close();

	// 创建一个运行时对象并反序列化引擎。
	mRuntime = nvinfer1::createInferRuntime(gLogger);
	assert(mRuntime);  // 确保运行时成功创建
	mEngine = mRuntime->deserializeCudaEngine(serializedEngine, size);
	mContext = mEngine->createExecutionContext();
	delete[] serializedEngine;  // 释放序列化的引擎内存

	// 确保IO张量的数量与预期的输入和输出数量匹配。
	assert(mEngine->getNbIOTensors() == inputNames.size() + outputNames.size());

	// 使用输入和输出名称初始化引擎。
	initialize(inputNames, outputNames);
}

void EngineTRT::initialize(std::vector<std::string> inputNames, std::vector<std::string> outputNames)
{
	// 调整GPU和CPU缓冲区向量的大小以容纳所有引擎IO张量
	const int numIOTensors = mEngine->getNbIOTensors();
	mGpuBuffers.resize(numIOTensors);
	mCpuBuffers.resize(numIOTensors);

	// 循环遍历所有IO张量以分配内存并存储维度信息
	for (int i = 0; i < numIOTensors; ++i)
	{
		const char* tensorName = mEngine->getIOTensorName(i);
		// 根据张量的维度（Dims64）计算所需的大小
		nvinfer1::Dims64 shape64 = mEngine->getTensorShape(tensorName);
		size_t tensor_size = getSizeByDim64(shape64);
		mBufferBindingSizes.push_back(tensor_size);  // 存储张量的大小
		mBufferBindingBytes.push_back(tensor_size * sizeof(float));  // 计算字节大小

		// 为CPU缓冲区分配主机内存
		mCpuBuffers[i] = new float[tensor_size];

		// 为GPU缓冲区分配设备内存
		cudaMalloc(&mGpuBuffers[i], mBufferBindingBytes[i]);

		// 根据张量是输入还是输出，分别存储输入和输出维度
		if (mEngine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT)
		{
			mInputDims.push_back(toDims32(shape64));
		}
		else
		{
			mOutputDims.push_back(toDims32(shape64));
		}
	}

	// 为异步操作创建一个CUDA流
	CUDA_CHECK(cudaStreamCreate(&mCudaStream));
}

bool EngineTRT::infer()
{
	// 将数据从主机（CPU）输入缓冲区异步复制到设备（GPU）输入缓冲区
	copyInputToDeviceAsync(mCudaStream);

	// 使用TensorRT执行推理，传递GPU缓冲区
	bool status = mContext->executeV2(mGpuBuffers.data());

	if (!status)
	{
		// 如果推理失败，打印错误消息并返回false
		std::cout << "inference error!" << std::endl;
		return false;
	}

	// 将结果从设备（GPU）输出缓冲区异步复制到主机（CPU）输出缓冲区
	copyOutputToHostAsync(mCudaStream);

	// 如果推理成功，返回true
	return true;
}

void EngineTRT::copyInputToDeviceAsync(const cudaStream_t& stream)
{
	// 为输入缓冲区执行从CPU到GPU的异步内存复制
	memcpyBuffers(true, false, true, stream);
}

void EngineTRT::copyOutputToHostAsync(const cudaStream_t& stream)
{
	// 调用memcpyBuffers来处理从GPU到CPU内存的数据复制。
	// 参数：false（不复制输入缓冲区），true（从设备复制数据到主机），
	// true（异步执行复制），以及给定的CUDA流。
	memcpyBuffers(false, true, true, stream);
}

void EngineTRT::memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t& stream)
{
	// 循环遍历TensorRT引擎中的所有IO张量（输入和输出）。
	for (int i = 0; i < mEngine->getNbIOTensors(); i++)
	{
		const char* tensorName = mEngine->getIOTensorName(i);
		const bool isInput = (mEngine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT);
		// 根据复制方向确定目标指针和源指针。
		void* dstPtr = deviceToHost ? mCpuBuffers[i] : mGpuBuffers[i];
		const void* srcPtr = deviceToHost ? mGpuBuffers[i] : mCpuBuffers[i];
		// 获取缓冲区的字节大小。
		const size_t byteSize = mBufferBindingBytes[i];
		// 根据方向设置内存复制操作的类型。
		const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;

		// 检查当前张量是输入还是输出，并相应地进行复制。
		if ((copyInput && isInput) || (!copyInput && !isInput))
		{
			if (async)
			{
				// 使用CUDA流执行异步内存复制。
				CUDA_CHECK(cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream));
			}
			else
			{
				// 执行同步内存复制。
				CUDA_CHECK(cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType));
			}
		}
	}
}

size_t EngineTRT::getSizeByDim(const nvinfer1::Dims& dims)
{
	size_t size = 1;

	// 循环遍历每个维度并相乘以计算总大小。
	for (size_t i = 0; i < dims.nbDims; ++i)
	{
		// 如果维度为-1（动态），则使用预定义的最大大小。
		if (dims.d[i] == -1)
			size *= MAX_NUM_PROMPTS;
		else
			size *= dims.d[i];
	}

	return size;
}

void EngineTRT::setInput(cv::Mat& image)
{
	// 从模型的输入形状中提取输入维度（高度和宽度）
	const int inputH = mInputDims[0].d[2];
	const int inputW = mInputDims[0].d[3];

	int i = 0;  // 缓冲区放置的索引计数器

	// 遍历输入图像中的每个像素
	for (int row = 0; row < image.rows; ++row)
	{
		// 指向图像数据中行起始位置的指针
		uchar* uc_pixel = image.data + row * image.step;

		for (int col = 0; col < image.cols; ++col)
		{
			// 对RGB通道的像素值进行归一化
			// 将BGR图像转换为归一化的RGB并存储在mCpuBuffers中
			mCpuBuffers[0][i] = ((float)uc_pixel[2] / 255.0f - 0.485f) / 0.229f; // 红色通道
			mCpuBuffers[0][i + image.rows * image.cols] = ((float)uc_pixel[1] / 255.0f - 0.456f) / 0.224f; // 绿色通道
			mCpuBuffers[0][i + 2 * image.rows * image.cols] = ((float)uc_pixel[0] / 255.0f - 0.406f) / 0.225f; // 蓝色通道

			uc_pixel += 3;  // 移动到下一个像素
			++i;  // 递增索引
		}
	}
}

void EngineTRT::setInput(float* features, float* imagePointCoords, float* imagePointLabels, float* maskInput, float* hasMaskInput, int numPoints)
{
	// 清理旧缓冲区并为输入数据分配新缓冲区
	delete[] mCpuBuffers[1];
	delete[] mCpuBuffers[2];
	mCpuBuffers[1] = new float[numPoints * 2];  // 点坐标缓冲区
	mCpuBuffers[2] = new float[numPoints];      // 点标签缓冲区

	// 在GPU上为输入数据分配内存
	cudaMalloc(&mGpuBuffers[1], sizeof(float) * numPoints * 2); // 坐标
	cudaMalloc(&mGpuBuffers[2], sizeof(float) * numPoints);     // 标签

	// 为TensorRT设置数据绑定的字节大小
	mBufferBindingBytes[1] = sizeof(float) * numPoints * 2;
	mBufferBindingBytes[2] = sizeof(float) * numPoints;

	// 将输入数据复制到CPU缓冲区
	memcpy(mCpuBuffers[0], features, mBufferBindingBytes[0]);
	memcpy(mCpuBuffers[1], imagePointCoords, sizeof(float) * numPoints * 2);
	memcpy(mCpuBuffers[2], imagePointLabels, sizeof(float) * numPoints);
	memcpy(mCpuBuffers[3], maskInput, mBufferBindingBytes[3]);
	memcpy(mCpuBuffers[4], hasMaskInput, mBufferBindingBytes[4]);

	// 配置TensorRT使用动态输入形状（TensorRT 10 IO张量API）
	mContext->setOptimizationProfileAsync(0, mCudaStream); // 设置优化配置文件
	const char* coordsTensorName = mEngine->getIOTensorName(1);
	const char* labelsTensorName = mEngine->getIOTensorName(2);
	mContext->setInputShape(coordsTensorName, nvinfer1::Dims3{ 1, numPoints, 2 }); // 为坐标设置输入维度
	mContext->setInputShape(labelsTensorName, nvinfer1::Dims2{ 1, numPoints });    // 为标签设置输入维度
}

void EngineTRT::getOutput(float* features)
{
	// 查找输出张量索引并复制其数据
	int outIndex = -1;
	const int numIOTensors = mEngine->getNbIOTensors();
	for (int i = 0; i < numIOTensors; ++i)
	{
		const char* tensorName = mEngine->getIOTensorName(i);
		if (mEngine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kOUTPUT)
		{
			outIndex = i;
			break;
		}
	}

	if (outIndex < 0)
	{
		std::cerr << "No output tensor found for features." << std::endl;
		return;
	}

	memcpy(features, mCpuBuffers[outIndex], mBufferBindingBytes[outIndex]);
}

void EngineTRT::getOutput(float* iouPrediction, float* lowResolutionMasks)
{
	// 识别两个输出张量并按大小映射：小的->IoU，大的->低分辨率掩码
	int outIdxA = -1, outIdxB = -1;
	const int numIOTensors = mEngine->getNbIOTensors();
	for (int i = 0; i < numIOTensors; ++i)
	{
		const char* tensorName = mEngine->getIOTensorName(i);
		if (mEngine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kOUTPUT)
		{
			if (outIdxA == -1) outIdxA = i;
			else { outIdxB = i; break; }
		}
	}

	if (outIdxA < 0 || outIdxB < 0)
	{
		std::cerr << "Expected two output tensors (IoU and low-res masks)." << std::endl;
		return;
	}

	const size_t bytesA = mBufferBindingBytes[outIdxA];
	const size_t bytesB = mBufferBindingBytes[outIdxB];

	int iouIdx, masksIdx;
	if (bytesA <= bytesB)
	{
		iouIdx = outIdxA;
		masksIdx = outIdxB;
	}
	else
	{
		iouIdx = outIdxB;
		masksIdx = outIdxA;
	}

	memcpy(iouPrediction, mCpuBuffers[iouIdx], mBufferBindingBytes[iouIdx]);
	memcpy(lowResolutionMasks, mCpuBuffers[masksIdx], mBufferBindingBytes[masksIdx]);
}