#include "sam.h"
#include <filesystem>
#include <algorithm>

using namespace std;

SamSegmenter::SamSegmenter() {}
SamSegmenter::~SamSegmenter() {}

bool SamSegmenter::init(const std::string& encoderEnginePath, const std::string& decoderEnginePath) {
    try {
        mSam = std::make_unique<SpeedSam>(encoderEnginePath, decoderEnginePath);
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "[SamSegmenter] 初始化失败: %s\n", e.what());
        mSam.reset();
        return false;
    } catch (...) {
        fprintf(stderr, "[SamSegmenter] 初始化失败: 未知异常\n");
        mSam.reset();
        return false;
    }
}

std::vector<cv::Mat> SamSegmenter::inferFromDetections(cv::Mat& image,
                                                       trtyolo::DetectRes& detections,
                                                       float area_threshold_px,
                                                       float diameter_threshold_px) {
    std::vector<cv::Mat> masks;
    if (!mSam) {
        fprintf(stderr, "[SamSegmenter] 尚未初始化 SpeedSam，无法推理。\n");
        return masks;
    }

    // 如果没有检测结果，直接返回空
    if (detections.num <= 0) {
        return masks;
    }

    // 对每个检测框，生成一个 mask，并按阈值过滤（面积 & 外接圆直径）
    masks.reserve(detections.num);

    // 收集通过过滤的检测结果
    std::vector<decltype(detections.boxes)::value_type> kept_boxes;
    std::vector<int> kept_classes;
    std::vector<float> kept_scores;
    kept_boxes.reserve(detections.num);
    kept_classes.reserve(detections.num);
    kept_scores.reserve(detections.num);

    const float bin_thresh = 0.5f; // 与可视化保持一致的二值化阈值

    for (size_t i = 0; i < static_cast<size_t>(detections.num); ++i) {
        const auto& box = detections.boxes[i];
        // 将框转换为 SAM 所需的两点提示：左上角与右下角
        std::vector<cv::Point> bboxPoints = {
            cv::Point(static_cast<int>(box.left), static_cast<int>(box.top)),
            cv::Point(static_cast<int>(box.right), static_cast<int>(box.bottom))
        };

        // 标签：2 -> 框左上角；3 -> 框右下角（参考项目 main.cpp 中用法）
        std::vector<float> labels = {2.0f, 3.0f};

        // 使用 SpeedSam 进行推理
        cv::Mat mask = mSam->predict(image, bboxPoints, labels);

        // 基于掩码进行过滤
        bool keep = true;
        if (!mask.empty()) {
            // 二值化后计算面积（像素个数）
            cv::Mat bin;
            cv::compare(mask, bin_thresh, bin, cv::CMP_GT); // 0/255, CV_8U
            int area_pixels = cv::countNonZero(bin);

            if (area_threshold_px > 0.0f && static_cast<float>(area_pixels) < area_threshold_px) {
                keep = false;
            }

            if (keep && diameter_threshold_px > 0.0f) {
                float diameter = calculateMaskCircleDiameter(mask, bin_thresh);
                if (diameter < diameter_threshold_px) {
                    keep = false;
                }
            }
        }

        if (keep) {
            masks.emplace_back(std::move(mask));
            kept_boxes.emplace_back(detections.boxes[i]);
            kept_classes.emplace_back(detections.classes[i]);
            kept_scores.emplace_back(detections.scores[i]);
        }
    }

    // 同步过滤 detections
    detections.boxes = std::move(kept_boxes);
    detections.classes = std::move(kept_classes);
    detections.scores = std::move(kept_scores);
    detections.num = static_cast<int>(masks.size());

    return masks;
}

cv::Mat SamSegmenter::combineBinaryMask(const std::vector<cv::Mat>& masks, float thresh) {
    if (masks.empty()) return cv::Mat();
    cv::Size sz = masks.front().size();
    cv::Mat combined(sz, CV_8UC1, cv::Scalar(0));

    for (const auto& m : masks) {
        if (m.empty()) continue;
        cv::Mat bin;
        // 将 float 掩码二值化为 0/255
        cv::compare(m, thresh, bin, cv::CMP_GT); // bin: CV_8U，0 或 255
        cv::bitwise_or(combined, bin, combined);
    }
    return combined;
}

float SamSegmenter::calculateMaskCircleDiameter(const cv::Mat& mask, float bin_thresh) {
    if (mask.empty()) return 0.0f;

    // 将 float 掩码二值化为 0/255 的单通道 8U 图像
    cv::Mat bin;
    cv::compare(mask, bin_thresh, bin, cv::CMP_GT);

    // findContours 会修改输入图像，拷贝一份
    cv::Mat bin_copy = bin.clone();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin_copy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    if (contours.empty()) return 0.0f;

    // 找到最大轮廓
    size_t largest_idx = 0;
    double max_area = -1.0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double a = cv::contourArea(contours[i]);
        if (a > max_area) {
            max_area = a;
            largest_idx = i;
        }
    }

    // 计算最小外接圆直径
    cv::Point2f center;
    float radius = 0.0f;
    cv::minEnclosingCircle(contours[largest_idx], center, radius);
    return radius * 2.0f;
}

cv::Mat SamSegmenter::visualize(const cv::cuda::GpuMat& gpuImage,
                                const trtyolo::DetectRes& detections,
                                const std::vector<cv::Mat>& masks,
                                bool enableSaveToPath,
                                const std::string& uuid,
                                const std::string& savePath) {
    // 将 GPU 图像下载到 CPU
    cv::Mat image;
    if (!gpuImage.empty()) {
        gpuImage.download(image);
    }
    if (image.empty()) {
        // 防御：如果传入的是空 GPU 图像，则返回空
        return cv::Mat();
    }

    // 组合所有掩码为一个二值掩码
    cv::Mat maskCombined = combineBinaryMask(masks, 0.5f);

    // 透明红色叠加
    cv::Mat result = image.clone();
    if (!maskCombined.empty()) {
        cv::Mat redLayer(image.size(), CV_8UC3, cv::Scalar(0, 0, 255)); // BGR: 红色
        cv::Mat blended;
        cv::addWeighted(image, 1.0, redLayer, 0.4, 0.0, blended); // 透明度 0.4

        // 仅在掩码区域覆盖
        blended.copyTo(result, maskCombined);
    }

    // 绘制检测框
    if (detections.num > 0) {
        for (size_t i = 0; i < static_cast<size_t>(detections.num); ++i) {
            const auto& box = detections.boxes[i];
            cv::rectangle(result,
                          cv::Point(static_cast<int>(box.left), static_cast<int>(box.top)),
                          cv::Point(static_cast<int>(box.right), static_cast<int>(box.bottom)),
                          cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }
    }

    // 保存到 savePath/uuid/seg_uuid.png
    if (enableSaveToPath && !uuid.empty() && !savePath.empty()) {
        try {
            std::filesystem::path out_dir = std::filesystem::path(savePath) / uuid;
            std::filesystem::create_directories(out_dir);
            std::filesystem::path out_file = out_dir / (std::string("seg_") + uuid + ".png");
            cv::imwrite(out_file.string(), result);
        } catch (const std::exception& e) {
            fprintf(stderr, "[SamSegmenter] 保存分割结果失败: %s\n", e.what());
        }
    }

    return result;
}