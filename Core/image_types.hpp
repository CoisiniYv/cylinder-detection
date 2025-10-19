/*
 * image_types.hpp
 *  Created on: 2025年10月17日
 * 核心数据结构，包括四张图为一组的结构，单张图结果，四张图结果等
 * 每张图的存储是用的OpenCV的Mat，因为在采集线程的时候就直接完成解码并存入结构体中了，一个结构体包含012通道合成的光度立体图片，和345通道合成的RGB图片，采集线程还未实现。;
 */

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>
// 直接使用 OpenCV 的 Mat 作为图像存储类型
#include <opencv2/core.hpp>

namespace XL {

    // 四张图为一组，固定大小常量
    constexpr std::size_t kQuadImageCount = 4;

    // 单张图的基础元数据（与业务相关的补充信息）
    struct FrameMeta {
        // 采集时的时间戳（毫秒，统一使用系统时间）
        std::int64_t timestamp_ms = 0;
        // 设备/相机标识，可为空
        std::string device_id;
        // 所属组 ID（四张图为一组），用于下游分组与判定
        std::uint64_t group_id = 0;
        // 组内序号（0~3），不在组内时为 -1
        int index_in_group = -1;
        // 全局序号（可选，可由上层维护）
        std::uint64_t sequence_id = 0;
    };

    // 单张图（采集线程已完成解码、合成）：
    // - ps_image：由 0/1/2 通道合成的光度立体图片（BGR，CV_8UC3）
    // - rgb_image：由 3/4/5 通道合成的普通 RGB 图片（BGR，CV_8UC3）
    struct ImageFrame {
        cv::Mat ps_image;   // 光度立体图（012）
        cv::Mat rgb_image;  // 普通 RGB 图（345）

        FrameMeta meta{};   // 基础元数据

        inline bool has_ps() const noexcept { return !ps_image.empty(); }
        inline bool has_rgb() const noexcept { return !rgb_image.empty(); }
        inline bool valid() const noexcept { return has_ps() && has_rgb(); }

        // 若两张图尺寸一致，则返回统一的宽高；否则返回 -1 表示不一致或不存在
        inline int width() const noexcept {
            if (has_rgb()) return rgb_image.cols;
            if (has_ps())  return ps_image.cols;
            return -1;
        }
        inline int height() const noexcept {
            if (has_rgb()) return rgb_image.rows;
            if (has_ps())  return ps_image.rows;
            return -1;
        }

        inline void clear() {
            ps_image.release();
            rgb_image.release();
        }
    };

    // 四张图组：按顺序保存四张图，统一携带组级元数据
    struct QuadFrame {
        std::array<ImageFrame, kQuadImageCount> frames{};

        // 组级元数据
        std::uint64_t group_id = 0;        // 组唯一标识（可由上层生成）
        std::int64_t timestamp_ms = 0;     // 组采集时间（毫秒）

        inline ImageFrame& at(std::size_t i) { return frames.at(i); }
        inline const ImageFrame& at(std::size_t i) const { return frames.at(i); }

        // 组内有效性校验（检查 4 张图是否都已合成完成）
        inline bool valid() const noexcept {
            for (std::size_t i = 0; i < kQuadImageCount; ++i) {
                if (!frames[i].valid()) return false;
            }
            return true;
        }
    };

    // 基础检测框结构（像素坐标）
    struct BBox {
        int x = 0;
        int y = 0;
        int w = 0;
        int h = 0;

        inline int x2() const noexcept { return x + w; }
        inline int y2() const noexcept { return y + h; }
        inline bool empty() const noexcept { return w <= 0 || h <= 0; }
    };

    // 单项检测结果
    struct Detection {
        int label_id = -1;               // 类别 id
        std::string label;               // 类别名称
        float confidence = 0.f;          // 置信度 [0,1]
        BBox box{};                      // 检测框
    };

    // 单张图的分析结果
    struct SingleImageResult {
        // 对应的图像元信息（可拷贝或只保存必要字段）
        FrameMeta meta{};
        // 检测列表
        std::vector<Detection> detections{};

        // 输出资源（例如保存路径/缩略图等），按需使用
        std::string saved_path;          // 原图或可视化结果的保存路径
        std::string thumbnail_path;      // 缩略图保存路径
    };

    // 四张图组的分析结果（与 QuadFrame 对应）
    struct QuadFrameResult {
        QuadFrame source{};                                      // 原始四图组
        std::array<SingleImageResult, kQuadImageCount> results{}; // 四张图对应结果

        // 组级聚合信息（例如总目标数等）
        inline std::size_t totalDetections() const noexcept {
            std::size_t n = 0;
            for (const auto& r : results) n += r.detections.size();
            return n;
        }
    };

} // namespace XL