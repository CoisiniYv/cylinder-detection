#pragma once

#include "HalconCpp.h"
#include "HDevThread.h"
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <string>
#include <vector>
#include <utility>
#include <future>
#include <mutex>
#include <atomic>
#include <queue>
#include <memory>
#include <functional>
#include <filesystem>
#include <cstdint>

class HalconProcessor
{
public:
    // 保存类型枚举
    enum class SaveType {
        TEMP_SAVE,      // 临时保存
        PATH_SAVE       // 指定路径保存
    };

    // 保存完成回调类型
    using SaveCompleteCallback = std::function<void(
        const std::string& filePath,     // 文件路径
        bool success,                    // 是否成功
        SaveType type,                   // 保存类型
        const std::string& message       // 附加信息（错误信息或成功信息）
        )>;

    HalconProcessor();
    ~HalconProcessor();

    // 处理图像并返回结果（输入为GPU图像）
    bool processImage(const cv::cuda::GpuMat& inputImage,
        std::string& outputImagePath,
        std::vector<std::pair<double, double>>& centerPoints,
        bool enableSaveToPath = false,
        const std::string& uuid = "",
        const std::string& savePath = "",
        std::uint64_t group_id = 0,
        int image_id = -1);

    // 设置保存完成回调函数
    void setSaveCompleteCallback(SaveCompleteCallback callback);

    // 清除回调函数
    void clearSaveCompleteCallback();

    // 手动清理临时文件
    void cleanupTempFiles();

    // 等待所有异步操作完成
    void waitForAsyncOperations();

    // 获取异步操作状态
    struct AsyncStatus {
        size_t pendingOperations;
        size_t completedOperations;
        size_t failedOperations;
    };

    AsyncStatus getAsyncStatus() const;

    // 调试开关：是否启用临时保存（默认关闭）
    void setEnableTempSave(bool enable) { mEnableTempSave = enable; }
    bool isTempSaveEnabled() const { return mEnableTempSave; }

private:
    std::string tempDir;

    // 回调函数相关
    SaveCompleteCallback saveCompleteCallback;
    std::mutex callbackMutex;

    // 异步操作相关成员
    std::atomic<size_t> pendingAsyncOperations;
    std::atomic<size_t> completedAsyncOperations;
    std::atomic<size_t> failedAsyncOperations;

    // 异步保存任务队列
    struct AsyncSaveTask {
        enum class TaskType {
            TEMP_SAVE,      // 保存到临时目录或直接保存到指定路径
            PATH_SAVE       // 从源路径复制到指定路径
        };

        TaskType type;
        HalconCpp::HObject image;
        HalconCpp::HObject scratchesAndDots;
        std::string sourcePath;
        std::string targetPath;
        std::promise<bool> promise;
    };

    std::queue<std::shared_ptr<AsyncSaveTask>> asyncSaveQueue;
    std::mutex queueMutex;
    std::condition_variable queueCondition;
    std::atomic<bool> stopAsyncThread;
    std::thread asyncWorkerThread;

    // 生成临时文件名（包含时间戳 + 组ID + 图片ID）
    std::string generateTempFilename(std::uint64_t group_id, int image_id) const;

    // 生成指定路径的文件名
    std::string generateSavePathFilename(const std::string& savePath,
        const std::string& uuid) const;

    // 开发模式设置
    void dev_update_off();

    // 异步保存叠加图像到临时路径或指定路径
    std::future<bool> saveOverlayImageAsync(const HalconCpp::HObject& ho_Image,
        const HalconCpp::HObject& ho_ScratchesAndDots,
        const std::string& outputPath);

    // 异步复制文件到指定路径
    std::future<bool> copyToSavePathAsync(const std::string& sourcePath,
        const std::string& targetPath);

    // 异步工作线程函数
    void asyncWorkerFunction();

    // 执行实际的保存操作
    bool performSaveOperation(const HalconCpp::HObject& image,
        const HalconCpp::HObject& scratchesAndDots,
        const std::string& outputPath);

    // 执行实际的复制操作
    bool performCopyOperation(const std::string& sourcePath,
        const std::string& targetPath);

    // 调用回调函数（线程安全）
    void invokeSaveCallback(const std::string& filePath,
        bool success,
        SaveType type,
        const std::string& message = "");

    // 将cv::cuda::GpuMat转换为Halcon图像对象（确保内部内存安全）
    static bool gpuMatToHalconHObject(const cv::cuda::GpuMat& gpuMat, HalconCpp::HObject& ho_Image);

    // 调试开关：是否启用临时保存
    bool mEnableTempSave = false;
};