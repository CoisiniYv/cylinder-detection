#pragma once
#include <opencv2/core/cuda.hpp>
#include <HalconCpp.h>

#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <future>
#include <filesystem>

class HalconProcessor
{
public:
	// 保存类型枚举
	enum class SaveType {
		PATH_SAVE
	};

	// 回调函数类型定义
	using SaveCompleteCallback = std::function<void(const std::string& filePath,
		bool success,
		SaveType type,
		const std::string& message)>;

	// 异步操作状态结构体
	struct AsyncStatus {
		int pendingOperations;
		int completedOperations;
		int failedOperations;
	};

	// 构造函数和析构函数
	HalconProcessor();
	~HalconProcessor();

	// 删除拷贝构造函数和赋值操作符
	HalconProcessor(const HalconProcessor&) = delete;
	HalconProcessor& operator=(const HalconProcessor&) = delete;

	// 主要图像处理函数
	bool processImage(const cv::cuda::GpuMat& inputImage,
		std::string& outputImagePath,
		std::vector<std::pair<double, double>>& centerPoints,
		bool enableSaveToPath = false,
		const std::string& savePath = "",
		std::uint64_t group_id = 0,
		std::int8_t index_in_group = -1);

	// 异步操作管理
	void waitForAsyncOperations();
	AsyncStatus getAsyncStatus() const;

	// 回调函数管理
	void setSaveCompleteCallback(SaveCompleteCallback callback);
	void clearSaveCompleteCallback();

	// 工具函数
	static bool gpuMatToHalconHObject(const cv::cuda::GpuMat& gpuMat,
		HalconCpp::HObject& ho_Image);

private:
	// 异步任务结构体
	struct AsyncSaveTask {
		enum class TaskType {
			PATH_SAVE
		};

		TaskType type;
		HalconCpp::HObject image;
		HalconCpp::HObject scratchesAndDots;
		std::string targetPath;
		std::promise<bool> promise;
	};

	// 成员变量
	std::atomic<int> pendingAsyncOperations;
	std::atomic<int> completedAsyncOperations;
	std::atomic<int> failedAsyncOperations;

	std::thread asyncWorkerThread;
	std::atomic<bool> stopAsyncThread;

	std::queue<std::shared_ptr<AsyncSaveTask>> asyncSaveQueue;
	std::mutex queueMutex;
	std::condition_variable queueCondition;

	SaveCompleteCallback saveCompleteCallback;
	std::mutex callbackMutex;

	// 私有方法
	void asyncWorkerFunction();
	void invokeSaveCallback(const std::string& filePath,
		bool success,
		SaveType type,
		const std::string& message);

	std::future<bool> saveOverlayImageAsync(const HalconCpp::HObject& ho_Image,
		const HalconCpp::HObject& ho_ScratchesAndDots,
		const std::string& outputPath);

	bool performSaveOperation(const HalconCpp::HObject& image,
		const HalconCpp::HObject& scratchesAndDots,
		const std::string& outputPath);

	std::string generateSavePathFilename(const std::string& savePath,
		std::uint64_t group_id,
		std::int8_t index_in_group) const;
};