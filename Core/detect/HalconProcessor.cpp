#include <opencv2/opencv.hpp>
#include "HalconProcessor.h"

#include <iostream>


HalconProcessor::HalconProcessor()
	: pendingAsyncOperations(0),
	completedAsyncOperations(0),
	failedAsyncOperations(0),
	stopAsyncThread(false)
{
	// HALCON系统设置
	HalconCpp::SetSystem("parallelize_operators", "true");

	// 启动异步工作线程
	asyncWorkerThread = std::thread(&HalconProcessor::asyncWorkerFunction, this);
}

HalconProcessor::~HalconProcessor()
{
	// 停止异步工作线程
	stopAsyncThread = true;
	queueCondition.notify_all();

	if (asyncWorkerThread.joinable())
	{
		asyncWorkerThread.join();
	}

	// 等待所有异步操作完成（最多等待5秒）
	auto start = std::chrono::steady_clock::now();
	while (pendingAsyncOperations > 0)
	{
		auto now = std::chrono::steady_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start);
		if (elapsed.count() >= 5)
		{
			std::cerr << "Warning: Timeout waiting for async operations to complete" << std::endl;
			break;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
}

void HalconProcessor::setSaveCompleteCallback(SaveCompleteCallback callback)
{
	std::lock_guard<std::mutex> lock(callbackMutex);
	saveCompleteCallback = callback;
}

void HalconProcessor::clearSaveCompleteCallback()
{
	std::lock_guard<std::mutex> lock(callbackMutex);
	saveCompleteCallback = nullptr;
}

void HalconProcessor::invokeSaveCallback(const std::string& filePath,
	bool success,
	SaveType type,
	const std::string& message)
{
	std::lock_guard<std::mutex> lock(callbackMutex);
	if (saveCompleteCallback)
	{
		try
		{
			saveCompleteCallback(filePath, success, type, message);
		}
		catch (const std::exception& e)
		{
			std::cerr << "Error in save complete callback: " << e.what() << std::endl;
		}
	}
}

void HalconProcessor::waitForAsyncOperations()
{
	while (pendingAsyncOperations > 0)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
}

HalconProcessor::AsyncStatus HalconProcessor::getAsyncStatus() const
{
	return AsyncStatus{
		pendingAsyncOperations.load(),
		completedAsyncOperations.load(),
		failedAsyncOperations.load() };
}

std::string HalconProcessor::generateSavePathFilename(const std::string& savePath,
	std::uint64_t group_id,
	std::int8_t index_in_group) const
{
	std::string directoryPath = savePath;
	std::filesystem::create_directories(directoryPath);

	return directoryPath + "\\" + "skeleton" + "_g" + std::to_string(group_id) + "_i" + std::to_string(index_in_group) + ".png";
}

std::future<bool> HalconProcessor::saveOverlayImageAsync(const HalconCpp::HObject& ho_Image,
	const HalconCpp::HObject& ho_ScratchesAndDots,
	const std::string& outputPath)
{
	auto task = std::make_shared<AsyncSaveTask>();
	task->type = AsyncSaveTask::TaskType::PATH_SAVE;

	// 复制HALCON对象到任务中
	try {
		// 复制图像对象 - 创建独立的副本
		HalconCpp::CopyImage(ho_Image, &(task->image));

		// 复制区域对象 - 创建独立的副本
		if (ho_ScratchesAndDots.CountObj() > 0) {
			HalconCpp::CopyObj(ho_ScratchesAndDots, &(task->scratchesAndDots), 1, -1);
		}
	}
	catch (const HalconCpp::HException& e) {
		std::cerr << "Error copying HALCON objects for async save: "
			<< e.ErrorMessage().TextA() << std::endl;
		std::promise<bool> promise;
		promise.set_value(false);
		return promise.get_future();
	}

	task->targetPath = outputPath;
	auto future = task->promise.get_future();

	{
		std::lock_guard<std::mutex> lock(queueMutex);
		asyncSaveQueue.push(task);
		pendingAsyncOperations++;
	}

	queueCondition.notify_one();

	return future;
}

void HalconProcessor::asyncWorkerFunction()
{
	while (!stopAsyncThread)
	{
		std::shared_ptr<AsyncSaveTask> task;

		{
			std::unique_lock<std::mutex> lock(queueMutex);
			queueCondition.wait(lock, [this]()
				{ return !asyncSaveQueue.empty() || stopAsyncThread; });

			if (stopAsyncThread && asyncSaveQueue.empty())
			{
				break;
			}

			if (!asyncSaveQueue.empty())
			{
				task = asyncSaveQueue.front();
				asyncSaveQueue.pop();
			}
		}

		if (task)
		{
			bool success = false;
			std::string message;
			try
			{
				// 直接保存到指定路径
				success = performSaveOperation(task->image, task->scratchesAndDots, task->targetPath);
				message = success ? "路径保存成功" : "路径保存失败";

				if (!success)
				{
					std::cerr << "Async operation failed: " << task->targetPath << std::endl;
					failedAsyncOperations++;
					message = "操作失败";
				}
			}
			catch (const std::exception& e)
			{
				std::cerr << "Exception in async operation: " << e.what() << std::endl;
				failedAsyncOperations++;
				success = false;
				message = std::string("异常: ") + e.what();
			}

			// 设置promise值
			task->promise.set_value(success);

			// 调用回调函数通知保存完成
			invokeSaveCallback(task->targetPath, success, SaveType::PATH_SAVE, message);

			pendingAsyncOperations--;
			completedAsyncOperations++;
		}
	}
}

bool HalconProcessor::performSaveOperation(const HalconCpp::HObject& image,
    const HalconCpp::HObject& scratchesAndDots,
    const std::string& outputPath)
{
    try {
        HalconCpp::HObject ho_Red, ho_Green, ho_Blue;
        HalconCpp::HObject ho_OverlayImage;
        HalconCpp::HTuple channels;
        HalconCpp::CountChannels(image, &channels);
        if ((int)channels[0].I() == 3) {
            HalconCpp::Decompose3(image, &ho_Red, &ho_Green, &ho_Blue);
        }
        else {
            HalconCpp::HObject color_image;
            HalconCpp::Compose3(image, image, image, &color_image);
            HalconCpp::Decompose3(color_image, &ho_Red, &ho_Green, &ho_Blue);
        }

        if (scratchesAndDots.CountObj() > 0) {
            HalconCpp::PaintRegion(scratchesAndDots, ho_Red, &ho_Red, 255, "fill");
            HalconCpp::PaintRegion(scratchesAndDots, ho_Green, &ho_Green, 0, "fill");
            HalconCpp::PaintRegion(scratchesAndDots, ho_Blue, &ho_Blue, 0, "fill");
        }

        HalconCpp::Compose3(ho_Red, ho_Green, ho_Blue, &ho_OverlayImage);
        HalconCpp::WriteImage(ho_OverlayImage, "png", 0, outputPath.c_str());

        return true;
    }
    catch (const HalconCpp::HException& e) {
        std::cerr << "HALCON Error in async saveOverlayImage: " << e.ErrorMessage().TextA() << std::endl;
        return false;
    }
	catch (const std::exception& e) {
		std::cerr << "Standard exception in async saveOverlayImage: " << e.what() << std::endl;
		return false;
	}
}

// 将cv::cuda::GpuMat转换为Halcon图像对象（内部进行深拷贝，避免外部内存失效）
bool HalconProcessor::gpuMatToHalconHObject(const cv::cuda::GpuMat& gpuMat, HalconCpp::HObject& ho_Image)
{
	try {
		if (gpuMat.empty()) {
			std::cerr << "gpuMatToHalconHObject: 输入图像为空" << std::endl;
			return false;
		}

		cv::Mat mat;
		gpuMat.download(mat);
		if (mat.empty()) {
			std::cerr << "gpuMatToHalconHObject: 下载失败" << std::endl;
			return false;
		}

		if (mat.type() == CV_8UC3) {
			// OpenCV BGR -> HALCON 三通道
			HalconCpp::HObject hcolor;
			HalconCpp::GenImageConst(&hcolor, "byte", mat.cols, mat.rows);

			// 拆分 BGR
			std::vector<cv::Mat> bgr(3);
			cv::split(mat, bgr);

			HalconCpp::HObject hR, hG, hB;
			HalconCpp::GenImage1(&hR, "byte", mat.cols, mat.rows, (Hlong)bgr[2].data);
			HalconCpp::GenImage1(&hG, "byte", mat.cols, mat.rows, (Hlong)bgr[1].data);
			HalconCpp::GenImage1(&hB, "byte", mat.cols, mat.rows, (Hlong)bgr[0].data);
			HalconCpp::Compose3(hR, hG, hB, &hcolor);

			HalconCpp::CopyImage(hcolor, &ho_Image);
			return true;
		}
		else if (mat.type() == CV_8UC1) {
			// 单通道灰度
			HalconCpp::HObject hgray;
			HalconCpp::GenImage1(&hgray, "byte", mat.cols, mat.rows, (Hlong)mat.data);
			HalconCpp::CopyImage(hgray, &ho_Image);
			return true;
		}
		else {
			std::cerr << "gpuMatToHalconHObject: 不支持的图像类型: " << mat.type() << std::endl;
			return false;
		}
	}
	catch (const HalconCpp::HException& e) {
		std::cerr << "HALCON Error in gpuMatToHalconHObject: " << e.ErrorMessage().TextA() << std::endl;
		return false;
	}
	catch (const std::exception& e) {
		std::cerr << "Standard exception in gpuMatToHalconHObject: " << e.what() << std::endl;
		return false;
	}
}

bool HalconProcessor::processImage(const cv::cuda::GpuMat& inputImage,
	std::string& outputImagePath,
	std::vector<std::pair<double, double>>& centerPoints,
	bool enableSaveToPath,
	const std::string& savePath,
	std::uint64_t group_id,
	std::int8_t index_in_group)
{
	try {
		// 清空输出
		centerPoints.clear();
		outputImagePath.clear();

		// 验证参数
		if (enableSaveToPath) {
			if (savePath.empty()) {
				std::cerr << "Warning: Save path is empty when saveToPath is enabled" << std::endl;
				enableSaveToPath = false;
			}
		}

		// 生成保存路径
		std::string saveOutputPath;
		const bool doSaveToPath = enableSaveToPath && !savePath.empty();
		if (doSaveToPath) {
			saveOutputPath = generateSavePathFilename(savePath, group_id, index_in_group);
			outputImagePath = saveOutputPath;
		}

		// Local iconic variables
		HalconCpp::HObject  ho_Image, ho_ImageMean, ho_ImageVTexture;
		HalconCpp::HObject  ho_R, ho_G, ho_B, ho_RG, ho_RGoverlap, ho_RegionOverlap;
		HalconCpp::HObject  ho_ImageHTexture, ho_R1, ho_G1, ho_B1, ho_RG1, ho_RGoverlap1;
		HalconCpp::HObject  ho_RegionOverlap1, ho_RegionDilated, ho_RegionEroded;
		HalconCpp::HObject  ho_ConnectedRegions, ho_FinalDefects, ho_ObjectsConcat;
		HalconCpp::HObject  ho_MultipliedImage, ho_EnhancedImage, ho_BrightRegions;
		HalconCpp::HObject  ho_ConnectedRegions1, ho_FinalRegions;

		// Local control variables
		HalconCpp::HTuple  hv_DeviceIdentifier;
		HalconCpp::HTuple hv_Area, hv_Row, hv_Column;
		HalconCpp::HTuple hv_Number, hv_NumberSelected, hv_NumberErrors, hv_NumberScratchesAndDots;
		// 转换GPU图像为HALCON图像
		if (!gpuMatToHalconHObject(inputImage, ho_Image)) {
			std::cerr << "错误: 无法将GpuMat转换为HALCON图像" << std::endl;
			return false;
		}

		HalconCpp::SetSystem("parallelize_operators", "true");
		HalconCpp::QueryAvailableComputeDevices(&hv_DeviceIdentifier);

		HalconCpp::MeanImage(ho_Image, &ho_ImageMean, 6, 6);

		HalconCpp::TextureLaws(ho_ImageMean, &ho_ImageVTexture, "le", 3, 5);
		HalconCpp::Decompose3(ho_ImageVTexture, &ho_R, &ho_G, &ho_B);
		HalconCpp::ConcatObj(ho_R, ho_G, &ho_RG);
		HalconCpp::MinImage(ho_RG, ho_B, &ho_RGoverlap);
		HalconCpp::Threshold(ho_RGoverlap, &ho_RegionOverlap, 64, 255);

		HalconCpp::TextureLaws(ho_ImageMean, &ho_ImageHTexture, "el", 3, 5);
		HalconCpp::Decompose3(ho_ImageHTexture, &ho_R1, &ho_G1, &ho_B1);
		HalconCpp::ConcatObj(ho_R1, ho_G1, &ho_RG1);
		HalconCpp::MinImage(ho_RG1, ho_RG, &ho_RGoverlap1);
		HalconCpp::Threshold(ho_RGoverlap1, &ho_RegionOverlap1, 50, 255);

		HalconCpp::DilationCircle(ho_RegionOverlap1, &ho_RegionDilated, 4);
		HalconCpp::ErosionCircle(ho_RegionDilated, &ho_RegionEroded, 2);
		HalconCpp::Connection(ho_RegionEroded, &ho_ConnectedRegions);
		HalconCpp::SelectShape(ho_ConnectedRegions, &ho_FinalDefects, "area", "and", 100, 2000);
		HalconCpp::ConcatObj(ho_ImageVTexture, ho_ImageHTexture, &ho_ObjectsConcat);

		HalconCpp::MultImage(ho_ImageHTexture, ho_ImageVTexture, &ho_MultipliedImage, 0.1, 0);
		HalconCpp::ScaleImage(ho_ImageVTexture, &ho_EnhancedImage, 3, 0);
		HalconCpp::Threshold(ho_EnhancedImage, &ho_BrightRegions, 128, 255);
		HalconCpp::Connection(ho_BrightRegions, &ho_ConnectedRegions1);
		HalconCpp::SelectShape(ho_ConnectedRegions1, &ho_FinalRegions, "area", "and", 100, 2000);
		HalconCpp::Connection(ho_FinalRegions, &ho_ConnectedRegions1);

		HalconCpp::CountObj(ho_FinalRegions, &hv_NumberScratchesAndDots);

		bool defectsDetected = false;

		if (0 != (int(hv_NumberScratchesAndDots > 0)))
		{
			HalconCpp::AreaCenter(ho_FinalRegions, &hv_Area, &hv_Row, &hv_Column);

			// 保存中心点坐标
			for (int i = 0; i < hv_Row.Length(); i++) {
				centerPoints.emplace_back(
					hv_Column[i].D(),
					hv_Row[i].D()
				);
			}

			defectsDetected = true;

			if (enableSaveToPath && !saveOutputPath.empty()) {
				// 异步保存叠加图像到指定路径
				saveOverlayImageAsync(ho_Image, ho_FinalRegions, saveOutputPath);
			}
		}

		// 如果没有检测到缺陷
		if (!defectsDetected) {
			// 异步保存原始图像到临时路径或直接保存到指定路径
			HalconCpp::HObject emptyRegion;
			HalconCpp::GenEmptyObj(&emptyRegion);

			if (enableSaveToPath && !saveOutputPath.empty())
			{
				saveOverlayImageAsync(ho_Image, emptyRegion, saveOutputPath);
			}
		}

		return true;
	}
	catch (const HalconCpp::HException& e) {
		std::cerr << "HALCON Error #" << e.ErrorCode() << " in "
			<< e.ProcName().TextA() << ": " << e.ErrorMessage().TextA() << std::endl;
		return false;
	}
	catch (const std::exception& e) {
		std::cerr << "Standard exception: " << e.what() << std::endl;
		return false;
	}
}
