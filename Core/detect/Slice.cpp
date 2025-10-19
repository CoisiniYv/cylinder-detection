#include "Slice.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>
#include <filesystem>

std::vector<SliceInfo> ImageSlicer::extractSlices(
    const cv::Mat& originalImage,
    const std::vector<std::pair<double, double>>& centerPoints,
    int distance,
    bool saveSlices,
    const std::string& savePath,
    int sliceWidth,
    int sliceHeight) {

    std::vector<SliceInfo> slices;
    // 预留容量，避免多次扩容
    slices.reserve(centerPoints.size() * 5);

    // 检查原图是否为空
    if (originalImage.empty()) {
        std::cerr << "错误: 原图像为空!" << std::endl;
        return slices;
    }

    // 检查中心点是否为空
    if (centerPoints.empty()) {
        std::cerr << "警告: 中心点向量为空!" << std::endl;
        return slices;
    }

    // 检查距离参数
    if (distance < 0) {
        std::cerr << "警告: 距离参数为负值，将使用绝对值!" << std::endl;
        distance = std::abs(distance);
    }

    // 检查保存路径
    if (saveSlices && savePath.empty()) {
        std::cerr << "警告: 启用了保存但保存路径为空，将不会保存切片!" << std::endl;
        saveSlices = false;
    }

    // 创建保存目录（如果需要）
    if (saveSlices && !savePath.empty()) {
        std::filesystem::create_directories(savePath);
    }

    // 提取每个中心点的5个切片
    for (size_t i = 0; i < centerPoints.size(); ++i) {
        const auto& center = centerPoints[i];
        double centerX = center.first;
        double centerY = center.second;

        // 中心点切片 (索引0)
        SliceInfo centerSlice = extractSingleSlice(originalImage, centerX, centerY,
            sliceWidth, sliceHeight);
        // 推入向量时使用移动语义，避免深拷贝
        slices.emplace_back(std::move(centerSlice));

        // 上方切片 (索引1)
        SliceInfo topSlice = extractSingleSlice(originalImage, centerX, centerY - distance,
            sliceWidth, sliceHeight);
        slices.emplace_back(std::move(topSlice));

        // 下方切片 (索引2)
        SliceInfo bottomSlice = extractSingleSlice(originalImage, centerX, centerY + distance,
            sliceWidth, sliceHeight);
        slices.emplace_back(std::move(bottomSlice));

        // 左方切片 (索引3)
        SliceInfo leftSlice = extractSingleSlice(originalImage, centerX - distance, centerY,
            sliceWidth, sliceHeight);
        slices.emplace_back(std::move(leftSlice));

        // 右方切片 (索引4)
        SliceInfo rightSlice = extractSingleSlice(originalImage, centerX + distance, centerY,
            sliceWidth, sliceHeight);
        slices.emplace_back(std::move(rightSlice));

        // 保存切片（如果需要）
        if (saveSlices) {
            saveSliceImage(centerSlice.slice, savePath, static_cast<int>(i), 0);
            saveSliceImage(topSlice.slice, savePath, static_cast<int>(i), 1);
            saveSliceImage(bottomSlice.slice, savePath, static_cast<int>(i), 2);
            saveSliceImage(leftSlice.slice, savePath, static_cast<int>(i), 3);
            saveSliceImage(rightSlice.slice, savePath, static_cast<int>(i), 4);
        }
    }

    std::cout << "成功提取 " << slices.size() << " 个切片 ("
        << centerPoints.size() << " 个中心点 × 5 个方向)" << std::endl;
    return slices;
}

SliceInfo ImageSlicer::extractSingleSlice(
    const cv::Mat& originalImage,
    double centerX,
    double centerY,
    int sliceWidth,
    int sliceHeight) {

    int imgWidth = originalImage.cols;
    int imgHeight = originalImage.rows;

    // 计算切片边界
    int startX = static_cast<int>(centerX - sliceWidth / 2.0);
    int startY = static_cast<int>(centerY - sliceHeight / 2.0);
    int endX = startX + sliceWidth;
    int endY = startY + sliceHeight;

    // 创建黑色背景的切片
    cv::Mat slice = cv::Mat::zeros(sliceHeight, sliceWidth, originalImage.type());

    // 计算原图中实际可用的区域
    int roiStartX = std::max(startX, 0);
    int roiStartY = std::max(startY, 0);
    int roiEndX = std::min(endX, imgWidth);
    int roiEndY = std::min(endY, imgHeight);

    // 计算切片中对应的区域
    int sliceStartX = roiStartX - startX;
    int sliceStartY = roiStartY - startY;
    int sliceEndX = sliceStartX + (roiEndX - roiStartX);
    int sliceEndY = sliceStartY + (roiEndY - roiStartY);

    // 检查是否有重叠区域
    if (roiStartX < roiEndX && roiStartY < roiEndY &&
        sliceStartX >= 0 && sliceStartY >= 0 &&
        sliceEndX <= sliceWidth && sliceEndY <= sliceHeight) {

        // 提取原图中的区域
        cv::Rect originalROI(roiStartX, roiStartY, roiEndX - roiStartX, roiEndY - roiStartY);
        cv::Mat originalRegion = originalImage(originalROI);

        // 复制到切片中对应的位置
        cv::Rect sliceROI(sliceStartX, sliceStartY, sliceEndX - sliceStartX, sliceEndY - sliceStartY);
        originalRegion.copyTo(slice(sliceROI));
    }

    return SliceInfo(slice, cv::Point(startX, startY));
}

void ImageSlicer::saveSliceImage(const cv::Mat& slice, const std::string& savePath,
    int centerIndex, int sliceIndex) {
    std::string filename = savePath + "/center_" + std::to_string(centerIndex) +
        "_slice_" + std::to_string(sliceIndex) + ".png";

    if (cv::imwrite(filename, slice)) {
        std::cout << "切片 " << centerIndex << "_" << sliceIndex << " 已保存到: " << filename << std::endl;
    }
    else {
        std::cerr << "错误: 无法保存切片 " << centerIndex << "_" << sliceIndex << " 到: " << filename << std::endl;
    }
}

std::vector<SliceInfo> ImageSlicer::extractSlicesGPU(
    const cv::cuda::GpuMat& originalImageGPU,
    const std::vector<std::pair<double, double>>& centerPoints,
    int distance,
    bool saveSlices,
    const std::string& savePath,
    int sliceWidth,
    int sliceHeight) {

    std::vector<SliceInfo> slices;
    slices.reserve(centerPoints.size() * 5);

    // 校验
    if (originalImageGPU.empty()) {
        std::cerr << "错误: 原图像(GPU)为空!" << std::endl;
        return slices;
    }
    if (centerPoints.empty()) {
        std::cerr << "警告: 中心点向量为空!" << std::endl;
        return slices;
    }
    if (distance < 0) distance = std::abs(distance);
    if (saveSlices && savePath.empty()) {
        std::cerr << "警告: 启用了保存但保存路径为空，将不会保存切片!" << std::endl;
        saveSlices = false;
    }
    if (saveSlices && !savePath.empty()) {
        std::filesystem::create_directories(savePath);
    }

    // GPU 提取每个中心点的5个切片
    for (size_t i = 0; i < centerPoints.size(); ++i) {
        const auto& center = centerPoints[i];
        double centerX = center.first;
        double centerY = center.second;

        // 中心点
        SliceInfo centerSlice = extractSingleSliceGPU(originalImageGPU, centerX, centerY, sliceWidth, sliceHeight);
        slices.emplace_back(std::move(centerSlice));

        // 上下左右
        slices.emplace_back(extractSingleSliceGPU(originalImageGPU, centerX, centerY - distance, sliceWidth, sliceHeight));
        slices.emplace_back(extractSingleSliceGPU(originalImageGPU, centerX, centerY + distance, sliceWidth, sliceHeight));
        slices.emplace_back(extractSingleSliceGPU(originalImageGPU, centerX - distance, centerY, sliceWidth, sliceHeight));
        slices.emplace_back(extractSingleSliceGPU(originalImageGPU, centerX + distance, centerY, sliceWidth, sliceHeight));

        // 保存切片（如果需要）
        if (saveSlices) {
            saveSliceImage(slices[slices.size() - 5].slice, savePath, static_cast<int>(i), 0);
            saveSliceImage(slices[slices.size() - 4].slice, savePath, static_cast<int>(i), 1);
            saveSliceImage(slices[slices.size() - 3].slice, savePath, static_cast<int>(i), 2);
            saveSliceImage(slices[slices.size() - 2].slice, savePath, static_cast<int>(i), 3);
            saveSliceImage(slices[slices.size() - 1].slice, savePath, static_cast<int>(i), 4);
        }
    }

    std::cout << "Succeed Extract " << slices.size() << " Slices(GPU)" << std::endl;
    return slices;
}

SliceInfo ImageSlicer::extractSingleSliceGPU(
    const cv::cuda::GpuMat& originalImageGPU,
    double centerX,
    double centerY,
    int sliceWidth,
    int sliceHeight) {

    int imgWidth = originalImageGPU.cols;
    int imgHeight = originalImageGPU.rows;

    // 计算切片边界
    int startX = static_cast<int>(centerX - sliceWidth / 2.0);
    int startY = static_cast<int>(centerY - sliceHeight / 2.0);
    int endX = startX + sliceWidth;
    int endY = startY + sliceHeight;

    // 在GPU上创建黑色背景的切片
    cv::cuda::GpuMat sliceGPU(sliceHeight, sliceWidth, originalImageGPU.type());
    sliceGPU.setTo(cv::Scalar::all(0));

    // 计算原图中实际可用的区域
    int roiStartX = std::max(startX, 0);
    int roiStartY = std::max(startY, 0);
    int roiEndX = std::min(endX, imgWidth);
    int roiEndY = std::min(endY, imgHeight);

    // 计算切片中对应的区域
    int sliceStartX = roiStartX - startX;
    int sliceStartY = roiStartY - startY;
    int roiW = roiEndX - roiStartX;
    int roiH = roiEndY - roiStartY;

    if (roiW > 0 && roiH > 0 &&
        sliceStartX >= 0 && sliceStartY >= 0 &&
        sliceStartX + roiW <= sliceWidth &&
        sliceStartY + roiH <= sliceHeight) {

        // GPU ROI: 从原图拷贝到目标切片的对应位置
        cv::Rect srcROI(roiStartX, roiStartY, roiW, roiH);
        cv::Rect dstROI(sliceStartX, sliceStartY, roiW, roiH);
        cv::cuda::GpuMat srcRegion(originalImageGPU, srcROI);
        cv::cuda::GpuMat dstRegion(sliceGPU, dstROI);
        srcRegion.copyTo(dstRegion);
    }

    // 下载到CPU以供后续推理使用
    cv::Mat slice;
    sliceGPU.download(slice);
    return SliceInfo(slice, cv::Point(startX, startY));
}