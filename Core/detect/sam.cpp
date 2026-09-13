#include "sam.h"

#include "../Utils/Log.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>

bool SamSegmenter::init(
    const std::string& encoder_engine_path,
    const std::string& decoder_engine_path) {
    try {
        mSam = std::make_unique<SpeedSam>(encoder_engine_path, decoder_engine_path);
        LOGI("SAM initialized: encoder=%s decoder=%s",
             encoder_engine_path.c_str(), decoder_engine_path.c_str());
        return true;
    }
    catch (const std::exception& e) {
        LOGE("SAM initialization failed: %s", e.what());
        mSam.reset();
        return false;
    }
}

std::vector<cv::Mat> SamSegmenter::inferFromDetections(
    cv::Mat& image,
    trtyolo::DetectRes& detections,
    float area_threshold_px,
    float diameter_threshold_px) {
    mLastAreasPx.clear();
    mLastDiametersPx.clear();

    std::vector<cv::Mat> masks;
    if (!mSam || image.empty() || detections.num <= 0) return masks;

    masks.reserve(static_cast<std::size_t>(detections.num));
    std::vector<decltype(detections.boxes)::value_type> kept_boxes;
    std::vector<int> kept_classes;
    std::vector<float> kept_scores;
    kept_boxes.reserve(static_cast<std::size_t>(detections.num));
    kept_classes.reserve(static_cast<std::size_t>(detections.num));
    kept_scores.reserve(static_cast<std::size_t>(detections.num));

    constexpr float kBinaryThreshold = 0.5f;
    for (int i = 0; i < detections.num; ++i) {
        const auto& box = detections.boxes[static_cast<std::size_t>(i)];
        const std::vector<cv::Point> points{
            cv::Point(static_cast<int>(box.left), static_cast<int>(box.top)),
            cv::Point(static_cast<int>(box.right), static_cast<int>(box.bottom))};
        const std::vector<float> labels{2.0f, 3.0f};

        cv::Mat mask = mSam->predict(image, points, labels);
        if (mask.empty()) continue;

        cv::Mat binary;
        cv::compare(mask, kBinaryThreshold, binary, cv::CMP_GT);
        const double area_pixels = static_cast<double>(cv::countNonZero(binary));
        const float diameter_pixels = calculateMaskCircleDiameter(mask, kBinaryThreshold);

        if (area_threshold_px > 0.0f && area_pixels < area_threshold_px) continue;
        if (diameter_threshold_px > 0.0f && diameter_pixels < diameter_threshold_px) continue;

        mLastAreasPx.push_back(area_pixels);
        mLastDiametersPx.push_back(diameter_pixels);
        masks.emplace_back(std::move(mask));
        kept_boxes.emplace_back(detections.boxes[static_cast<std::size_t>(i)]);
        kept_classes.emplace_back(detections.classes[static_cast<std::size_t>(i)]);
        kept_scores.emplace_back(detections.scores[static_cast<std::size_t>(i)]);
    }

    detections.boxes = std::move(kept_boxes);
    detections.classes = std::move(kept_classes);
    detections.scores = std::move(kept_scores);
    detections.num = static_cast<int>(masks.size());
    return masks;
}

cv::Mat SamSegmenter::combineBinaryMask(
    const std::vector<cv::Mat>& masks,
    float threshold) {
    if (masks.empty()) return {};

    const cv::Size expected_size = masks.front().size();
    cv::Mat combined(expected_size, CV_8UC1, cv::Scalar(0));
    for (const auto& mask : masks) {
        if (mask.empty()) continue;
        if (mask.size() != expected_size) {
            LOGE("SAM mask size mismatch; skipping mask");
            continue;
        }
        cv::Mat binary;
        cv::compare(mask, threshold, binary, cv::CMP_GT);
        cv::bitwise_or(combined, binary, combined);
    }
    return combined;
}

float SamSegmenter::calculateMaskCircleDiameter(const cv::Mat& mask, float threshold) {
    if (mask.empty()) return 0.0f;

    cv::Mat binary;
    cv::compare(mask, threshold, binary, cv::CMP_GT);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    if (contours.empty()) return 0.0f;

    const auto largest = std::max_element(
        contours.begin(), contours.end(), [](const auto& left, const auto& right) {
            return cv::contourArea(left) < cv::contourArea(right);
        });

    cv::Point2f center;
    float radius = 0.0f;
    cv::minEnclosingCircle(*largest, center, radius);
    return radius * 2.0f;
}

cv::Mat SamSegmenter::visualize(
    const cv::cuda::GpuMat& gpu_image,
    std::string& output_image_path,
    const trtyolo::DetectRes& detections,
    const std::vector<cv::Mat>& masks,
    bool enable_save_to_path,
    const std::string& save_path,
    std::uint64_t group_id,
    int index_in_group) {
    output_image_path.clear();
    if (gpu_image.empty()) return {};

    cv::Mat image;
    gpu_image.download(image);
    if (image.empty()) return {};

    cv::Mat result = image.clone();
    const cv::Mat combined_mask = combineBinaryMask(masks);
    if (!combined_mask.empty()) {
        cv::Mat red_layer(image.size(), CV_8UC3, cv::Scalar(0, 0, 255));
        cv::Mat blended;
        cv::addWeighted(image, 1.0, red_layer, 0.4, 0.0, blended);
        blended.copyTo(result, combined_mask);
    }

    const int box_count = std::min(
        detections.num,
        static_cast<int>(detections.boxes.size()));
    for (int i = 0; i < box_count; ++i) {
        const auto& box = detections.boxes[static_cast<std::size_t>(i)];
        cv::rectangle(
            result,
            cv::Point(static_cast<int>(box.left), static_cast<int>(box.top)),
            cv::Point(static_cast<int>(box.right), static_cast<int>(box.bottom)),
            cv::Scalar(0, 255, 0),
            2,
            cv::LINE_AA);
    }

    if (enable_save_to_path && !save_path.empty()) {
        const std::filesystem::path directory(save_path);
        std::filesystem::create_directories(directory);
        const std::filesystem::path output_file =
            directory /
            ("output_g" + std::to_string(group_id) +
             "_i" + std::to_string(index_in_group) + ".png");
        if (!cv::imwrite(output_file.string(), result)) {
            LOGE("failed to save SAM visualization: %s", output_file.string().c_str());
        }
        else {
            output_image_path = output_file.string();
        }
    }

    return result;
}
