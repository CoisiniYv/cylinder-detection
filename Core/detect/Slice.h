#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <vector>
#include <string>
#include <utility>
#include "trtyolo_slice.hpp"

/**
 * @brief 切片信息类型，复用 trtyolo::SliceInfo 以避免重复结构体
 */
using SliceInfo = trtyolo::SliceInfo;

/**
 * @brief 图像切片工具类
 */
class ImageSlicer {
public:
    /**
     * @brief 从原图中提取多个切片（每个中心点切5个片）
     *
     * @param originalImage 原图像
     * @param centerPoints 切片中心点坐标向量 (x, y)
     * @param distance 距离参数（像素），用于上下左右偏移
     * @param saveSlices 是否保存切片图像
     * @param savePath 保存路径
     * @param sliceWidth 切片宽度
     * @param sliceHeight 切片高度
     * @return std::vector<SliceInfo> 切片信息向量
     */
    static std::vector<SliceInfo> extractSlices(
        const cv::Mat& originalImage,
        const std::vector<std::pair<double, double>>& centerPoints,
        int distance = 50,
        bool saveSlices = false,
        const std::string& savePath = "",
        int sliceWidth = 100,
        int sliceHeight = 100
    );

    /**
     * @brief 使用GPU在原图中提取多个切片（每个中心点切5个片），返回CPU Mat用于推理
     *
     * @param originalImageGPU 原图像（GPU）
     * @param centerPoints 切片中心点坐标向量 (x, y)
     * @param distance 距离参数（像素），用于上下左右偏移
     * @param saveSlices 是否保存切片图像
     * @param savePath 保存路径
     * @param sliceWidth 切片宽度
     * @param sliceHeight 切片高度
     * @return std::vector<SliceInfo> 切片信息向量（slice为CPU Mat）
     */
    static std::vector<SliceInfo> extractSlicesGPU(
        const cv::cuda::GpuMat& originalImageGPU,
        const std::vector<std::pair<double, double>>& centerPoints,
        int distance = 50,
        bool saveSlices = false,
        const std::string& savePath = "",
        int sliceWidth = 100,
        int sliceHeight = 100
    );

private:
    /**
     * @brief 提取单个切片
     *
     * @param originalImage 原图像
     * @param centerX 中心点x坐标
     * @param centerY 中心点y坐标
     * @param sliceWidth 切片宽度
     * @param sliceHeight 切片高度
     * @return SliceInfo 切片信息
     */
    static SliceInfo extractSingleSlice(
        const cv::Mat& originalImage,
        double centerX,
        double centerY,
        int sliceWidth,
        int sliceHeight
    );

    /**
     * @brief 使用GPU提取单个切片（在GPU上完成填充与复制），返回CPU Mat用于推理
     */
    static SliceInfo extractSingleSliceGPU(
        const cv::cuda::GpuMat& originalImageGPU,
        double centerX,
        double centerY,
        int sliceWidth,
        int sliceHeight
    );

    /**
     * @brief 保存切片图像
     *
     * @param slice 切片图像
     * @param savePath 保存路径
     * @param centerIndex 中心点索引
     * @param sliceIndex 切片索引（0-4）
     */
    static void saveSliceImage(const cv::Mat& slice, const std::string& savePath,
        int centerIndex, int sliceIndex);
};