#include "HalconProcessor.h"
#include <random>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

HalconProcessor::HalconProcessor()
    : pendingAsyncOperations(0),
    completedAsyncOperations(0),
    failedAsyncOperations(0),
    stopAsyncThread(false)
{
    // 初始化临时目录
    tempDir = "./temp/";
    std::filesystem::create_directories(tempDir);

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

    if (asyncWorkerThread.joinable()) {
        asyncWorkerThread.join();
    }

    // 等待所有异步操作完成（最多等待5秒）
    auto start = std::chrono::steady_clock::now();
    while (pendingAsyncOperations > 0) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start);
        if (elapsed.count() >= 5) {
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
    if (saveCompleteCallback) {
        try {
            saveCompleteCallback(filePath, success, type, message);
        }
        catch (const std::exception& e) {
            std::cerr << "Error in save complete callback: " << e.what() << std::endl;
        }
    }
}

void HalconProcessor::waitForAsyncOperations()
{
    while (pendingAsyncOperations > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

HalconProcessor::AsyncStatus HalconProcessor::getAsyncStatus() const
{
    return AsyncStatus{
        pendingAsyncOperations.load(),
        completedAsyncOperations.load(),
        failedAsyncOperations.load()
    };
}

void HalconProcessor::cleanupTempFiles()
{
    try {
        std::cout << "Cleaning up temporary skeleton files..." << std::endl;
        size_t removedCount = 0;

        for (const auto& entry : std::filesystem::directory_iterator(tempDir)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();

                // 检查文件名是否以 "skeleton_" 开头并且是图片格式
                if (filename.find("skeleton_") == 0 &&
                    (entry.path().extension() == ".png" ||
                        entry.path().extension() == ".bmp" ||
                        entry.path().extension() == ".tiff" ||
                        entry.path().extension() == ".jpg" ||
                        entry.path().extension() == ".jpeg")) {

                    std::filesystem::remove(entry.path());
                    removedCount++;
                }
            }
        }

        std::cout << "Removed " << removedCount << " temporary skeleton files" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error during cleanup: " << e.what() << std::endl;
    }
}

std::string HalconProcessor::generateTempFilename(std::uint64_t group_id, int image_id) const
{
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();

    // 命名规则：skeleton_<timestamp>_g<group_id>_i<image_id>.png
    return tempDir + "skeleton_" + std::to_string(timestamp) +
        "_g" + std::to_string(group_id) +
        "_i" + std::to_string(image_id) + ".png";
}

std::string HalconProcessor::generateSavePathFilename(const std::string& savePath,
    const std::string& uuid) const
{
    // 构建完整路径: savePath/uuid/
    std::string directoryPath = savePath + "/" + uuid + "/";
    std::filesystem::create_directories(directoryPath);

    // 构建文件名: skeleton_ + uuid + .png
    return directoryPath + "skeleton_" + uuid + ".png";
}

void HalconProcessor::dev_update_off()
{
    return;
}

std::future<bool> HalconProcessor::saveOverlayImageAsync(const HalconCpp::HObject& ho_Image,
    const HalconCpp::HObject& ho_ScratchesAndDots,
    const std::string& outputPath)
{
    auto task = std::make_shared<AsyncSaveTask>();
    task->type = AsyncSaveTask::TaskType::TEMP_SAVE;

    // 关键步骤：复制HALCON对象到任务中，确保线程安全
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

std::future<bool> HalconProcessor::copyToSavePathAsync(const std::string& sourcePath,
    const std::string& targetPath)
{
    auto task = std::make_shared<AsyncSaveTask>();
    task->type = AsyncSaveTask::TaskType::PATH_SAVE;
    task->sourcePath = sourcePath;
    task->targetPath = targetPath;
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
    while (!stopAsyncThread) {
        std::shared_ptr<AsyncSaveTask> task;

        {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueCondition.wait(lock, [this]() {
                return !asyncSaveQueue.empty() || stopAsyncThread;
                });
        }

        if (stopAsyncThread) break;

        if (!asyncSaveQueue.empty()) {
            task = asyncSaveQueue.front();
            asyncSaveQueue.pop();

            bool success = true;
            std::string message;
            SaveType callbackType = SaveType::TEMP_SAVE;

            try {
                switch (task->type) {
                case AsyncSaveTask::TaskType::TEMP_SAVE:
                    success = performSaveOperation(task->image, task->scratchesAndDots, task->targetPath);
                    callbackType = SaveType::TEMP_SAVE;
                    break;
                case AsyncSaveTask::TaskType::PATH_SAVE:
                    success = performCopyOperation(task->sourcePath, task->targetPath);
                    callbackType = SaveType::PATH_SAVE;
                    break;
                default:
                    success = false;
                    message = "未知的异步任务类型";
                    break;
                }
            }
            catch (const HalconCpp::HException& e) {
                success = false;
                message = std::string("HALCON 异常: ") + e.ErrorMessage().TextA();
            }
            catch (const std::exception& e) {
                success = false;
                message = std::string("异常: ") + e.what();
            }

            // 设置promise值
            task->promise.set_value(success);

            // 调用回调函数通知保存完成
            invokeSaveCallback(task->targetPath, success, callbackType, message);

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
        // 将缺陷区域用红色叠加到原始彩色图像上
        HalconCpp::HObject ho_Red, ho_Green, ho_Blue;
        HalconCpp::HObject ho_OverlayImage;

        HalconCpp::Decompose3(image, &ho_Red, &ho_Green, &ho_Blue);

        // 在红色通道中绘制缺陷区域（设置为255）
        if (scratchesAndDots.CountObj() > 0) {
            HalconCpp::PaintRegion(scratchesAndDots, ho_Red, &ho_Red, 255, "fill");
            // 在绿色和蓝色通道中清除缺陷区域（设置为0）
            HalconCpp::PaintRegion(scratchesAndDots, ho_Green, &ho_Green, 0, "fill");
            HalconCpp::PaintRegion(scratchesAndDots, ho_Blue, &ho_Blue, 0, "fill");
        }

        // 重新组合三通道图像
        HalconCpp::Compose3(ho_Red, ho_Green, ho_Blue, &ho_OverlayImage);

        // 保存叠加图像
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

bool HalconProcessor::performCopyOperation(const std::string& sourcePath,
    const std::string& targetPath)
{
    try {
        // 确保目标目录存在
        std::filesystem::path targetFilePath(targetPath);
        std::filesystem::create_directories(targetFilePath.parent_path());

        // 复制文件
        std::filesystem::copy_file(sourcePath, targetPath,
            std::filesystem::copy_options::overwrite_existing);

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error copying file from " << sourcePath
            << " to " << targetPath << ": " << e.what() << std::endl;
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
        } else if (mat.type() == CV_8UC1) {
            // 单通道灰度
            HalconCpp::HObject hgray;
            HalconCpp::GenImage1(&hgray, "byte", mat.cols, mat.rows, (Hlong)mat.data);
            HalconCpp::CopyImage(hgray, &ho_Image);
            return true;
        } else {
            std::cerr << "gpuMatToHalconHObject: 不支持的图像类型: " << mat.type() << std::endl;
            return false;
        }
    } catch (const HalconCpp::HException& e) {
        std::cerr << "HALCON Error in gpuMatToHalconHObject: " << e.ErrorMessage().TextA() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception in gpuMatToHalconHObject: " << e.what() << std::endl;
        return false;
    }
}

bool HalconProcessor::processImage(const cv::cuda::GpuMat& inputImage,
    std::string& outputImagePath,
    std::vector<std::pair<double, double>>& centerPoints,
    bool enableSaveToPath,
    const std::string& uuid,
    const std::string& savePath,
    std::uint64_t group_id,
    int image_id)
{
    try {
        // 清空输出
        centerPoints.clear();
        outputImagePath.clear();

        // 验证参数
        if (enableSaveToPath) {
            if (uuid.empty()) {
                std::cerr << "Warning: UUID is empty when saveToPath is enabled" << std::endl;
            }
            if (savePath.empty()) {
                std::cerr << "Warning: Save path is empty when saveToPath is enabled" << std::endl;
                enableSaveToPath = false;
            }
        }

        // 根据调试开关决定是否生成临时输出路径
        std::string tempOutputPath;
        if (mEnableTempSave) {
            tempOutputPath = generateTempFilename(group_id, image_id);
        }

        // 生成保存路径（如果需要）
        std::string saveOutputPath;
        const bool doSaveToPath = enableSaveToPath && !uuid.empty() && !savePath.empty();
        if (doSaveToPath) {
            saveOutputPath = generateSavePathFilename(savePath, uuid);
        }

        // 设置输出路径（优先返回用户指定保存路径；否则返回临时路径；否则为空）
        if (doSaveToPath && !saveOutputPath.empty() && !mEnableTempSave) {
            outputImagePath = saveOutputPath;
        } else if (mEnableTempSave && !tempOutputPath.empty()) {
            outputImagePath = tempOutputPath;
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

        dev_update_off();
        HalconCpp::SetSystem("parallelize_operators", "true");
        HalconCpp::QueryAvailableComputeDevices(&hv_DeviceIdentifier);

        //===============================
        HalconCpp::MeanImage(ho_Image, &ho_ImageMean, 6, 6);
        //===============================

        //===============================
        HalconCpp::TextureLaws(ho_ImageMean, &ho_ImageVTexture, "le", 3, 5);
        HalconCpp::Decompose3(ho_ImageVTexture, &ho_R, &ho_G, &ho_B);
        HalconCpp::ConcatObj(ho_R, ho_G, &ho_RG);
        HalconCpp::MinImage(ho_RG, ho_B, &ho_RGoverlap);
        HalconCpp::Threshold(ho_RGoverlap, &ho_RegionOverlap, 64, 255);
        //===============================
        HalconCpp::TextureLaws(ho_ImageMean, &ho_ImageHTexture, "el", 3, 5);
        HalconCpp::Decompose3(ho_ImageHTexture, &ho_R1, &ho_G1, &ho_B1);
        HalconCpp::ConcatObj(ho_R1, ho_G1, &ho_RG1);
        HalconCpp::MinImage(ho_RG1, ho_RG, &ho_RGoverlap1);
        HalconCpp::Threshold(ho_RGoverlap1, &ho_RegionOverlap1, 50, 255);
        //===============================




        //===============================
        HalconCpp::DilationCircle(ho_RegionOverlap1, &ho_RegionDilated, 4);
        HalconCpp::ErosionCircle(ho_RegionDilated, &ho_RegionEroded, 2);
        HalconCpp::Connection(ho_RegionEroded, &ho_ConnectedRegions);
        HalconCpp::SelectShape(ho_ConnectedRegions, &ho_FinalDefects, "area", "and", 100, 999999);
        HalconCpp::ConcatObj(ho_ImageVTexture, ho_ImageHTexture, &ho_ObjectsConcat);
        //===============================
        HalconCpp::MultImage(ho_ImageHTexture, ho_ImageVTexture, &ho_MultipliedImage, 0.1, 0);
        HalconCpp::ScaleImage(ho_ImageVTexture, &ho_EnhancedImage, 3, 0);
        HalconCpp::Threshold(ho_EnhancedImage, &ho_BrightRegions, 128, 255);
        HalconCpp::Connection(ho_BrightRegions, &ho_ConnectedRegions1);
        HalconCpp::SelectShape(ho_ConnectedRegions1, &ho_FinalRegions, "area", "and", 100, 99999);
        HalconCpp::Connection(ho_FinalRegions, &ho_ConnectedRegions1);




        HalconCpp::CountObj(ho_FinalRegions, &hv_NumberScratchesAndDots);

        bool defectsDetected = false;


        if (0 != (int(hv_NumberScratchesAndDots>0)))
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

            if (mEnableTempSave && !tempOutputPath.empty()) {
                // 异步保存叠加图像到临时路径
                saveOverlayImageAsync(ho_Image, ho_FinalRegions, tempOutputPath);
                // 如果启用了保存到指定路径，启动异步复制
                if (doSaveToPath && !saveOutputPath.empty()) {
                    copyToSavePathAsync(tempOutputPath, saveOutputPath);
                }
            } else if (doSaveToPath && !saveOutputPath.empty()) {
                // 直接保存叠加图像到指定路径（不经过临时目录）
                saveOverlayImageAsync(ho_Image, ho_FinalRegions, saveOutputPath);
            }
        }


        // 如果没有检测到缺陷
        if (!defectsDetected) {
            // 异步保存原始图像到临时路径或直接保存到指定路径
            HalconCpp::HObject emptyRegion;
            HalconCpp::GenEmptyObj(&emptyRegion);

            if (mEnableTempSave && !tempOutputPath.empty()) {
                // 异步保存原始图像到临时路径
                saveOverlayImageAsync(ho_Image, emptyRegion, tempOutputPath);
                // 如果启用了保存到指定路径，启动异步复制
                if (doSaveToPath && !saveOutputPath.empty()) {
                    copyToSavePathAsync(tempOutputPath, saveOutputPath);
                }
            } else if (doSaveToPath && !saveOutputPath.empty()) {
                // 直接保存原始图像到指定路径（不经过临时目录）
                saveOverlayImageAsync(ho_Image, emptyRegion, saveOutputPath);
            }
        }

        std::cout << "Found " << centerPoints.size() << " center points" << std::endl;
        std::cout << "Output image: " << outputImagePath << std::endl;

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