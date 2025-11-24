#pragma once
#include <opencv2/opencv.hpp>
#include "NvInfer.h"



class EngineTRT
{

public:
	/// \brief TRTModule类的构造函数
	/// 
	/// \param modelPath ONNX模型文件的路径
	/// \param inputNames 输入张量的名称
	/// \param outputNames 输出张量的名称
	/// \param isDynamicShape 指示模型是否使用动态形状
	/// \param isFP16 指示模型是否应使用FP16精度
	EngineTRT(std::string modelPath, std::vector<std::string> inputNames, std::vector<std::string> outputNames, bool isDynamicShape, bool isFP16);

	/// \brief 对输入数据执行推理
	/// \return 如果推理成功返回true，否则返回false
	bool infer();

	/// \brief 设置用于推理的输入图像
	/// 
	/// \param image 要处理的输入图像
	void setInput(cv::Mat& image);

	/// \brief 从原始数据设置多个推理输入
	/// 
	/// \param features 指向特征数据的指针
	/// \param imagePointCoords 指向图像点坐标的指针
	/// \param imagePointLabels 指向图像点标签的指针
	/// \param maskInput 指向掩码输入数据的指针
	/// \param hasMaskInput 指向掩码数据存在性的指针
	/// \param numPoints 输入中的点数
	void setInput(float* features, float* imagePointCoords, float* imagePointLabels, float* maskInput, float* hasMaskInput, int numPoints);

	/// \brief 获取IoU和低分辨率掩码的输出预测
	/// 
	/// \param iouPrediction 指向存储IoU预测输出的指针
	/// \param lowResolutionMasks 指向存储低分辨率掩码输出的指针
	void getOutput(float* iouPrediction, float* lowResolutionMasks);

	/// \brief 从推理中获取输出特征
	/// 
	/// \param features 指向存储输出特征的指针
	void getOutput(float* features);

	/// \brief 保存引擎到文件
	/// 
	/// \param engineFilePath 引擎文件路径
	void saveEngine(const std::string& engineFilePath);

	/// \brief TRTModule类的析构函数
	~EngineTRT();

private:
	/// \brief 从ONNX模型构建TensorRT引擎
	/// 
	/// \param onnxPath ONNX模型文件的路径
	/// \param inputNames 输入张量的名称
	/// \param outputNames 输出张量的名称
	/// \param isDynamicShape 指示模型是否使用动态形状
	/// \param isFP16 指示模型是否应使用FP16精度
	void build(std::string onnxPath, std::vector<std::string> inputNames, std::vector<std::string> outputNames, bool isDynamicShape = false, bool isFP16 = false);

	/// \brief 从文件反序列化引擎
	/// 
	/// \param engineName 引擎文件的名称
	/// \param inputNames 输入张量的名称
	/// \param outputNames 输出张量的名称
	void deserializeEngine(std::string engineName, std::vector<std::string> inputNames, std::vector<std::string> outputNames);

	/// \brief 使用输入和输出名称初始化TensorRT模块
	/// 
	/// \param inputNames 输入张量的名称
	/// \param outputNames 输出张量的名称
	void initialize(std::vector<std::string> inputNames, std::vector<std::string> outputNames);

	/// \brief 根据维度获取缓冲区的大小
	/// 
	/// \param dims 缓冲区的维度
	/// \return 缓冲区的字节大小
	size_t getSizeByDim(const nvinfer1::Dims& dims);

	/// \brief 在设备和主机内存之间复制缓冲区
	/// 
	/// \param copyInput 指示是否复制输入数据
	/// \param deviceToHost 指示复制的方向
	/// \param async 指示复制是否应为异步
	/// \param stream 用于复制操作的CUDA流
	void memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t& stream = 0);

	/// \brief 异步将输入数据复制到设备
	/// 
	/// \param stream 用于复制操作的CUDA流
	void copyInputToDeviceAsync(const cudaStream_t& stream = 0);

	/// \brief 异步将输出数据从设备复制到主机
	/// 
	/// \param stream 用于复制操作的CUDA流
	void copyOutputToHostAsync(const cudaStream_t& stream = 0);

	std::vector<nvinfer1::Dims> mInputDims;            //!< 网络输入的维度
	std::vector<nvinfer1::Dims> mOutputDims;           //!< 网络输出的维度
	std::vector<void*> mGpuBuffers;          //!< 引擎执行所需的设备缓冲区向量
	std::vector<float*> mCpuBuffers;         //!< 输入/输出的CPU缓冲区向量
	std::vector<size_t> mBufferBindingBytes; //!< 每个缓冲区绑定的字节大小
	std::vector<size_t> mBufferBindingSizes; //!< 缓冲区绑定的大小
	cudaStream_t mCudaStream;           //!< 用于异步操作的CUDA流

	nvinfer1::IRuntime* mRuntime;                 //!< 用于反序列化引擎的TensorRT运行时
	nvinfer1::ICudaEngine* mEngine;               //!< 用于运行网络的TensorRT引擎
	nvinfer1::IExecutionContext* mContext;        //!< 使用ICudaEngine执行推理的上下文
};