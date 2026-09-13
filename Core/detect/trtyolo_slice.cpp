#include "trtyolo_slice.hpp"

#include "../Utils/Log.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trtyolo {

SliceDetector::SliceDetector(std::unique_ptr<DetectModel> model)
    : model_(std::move(model)) {
    if (!model_) throw std::invalid_argument("SliceDetector requires a model");
}

SliceDetector::SliceDetector(
    const std::string& trt_engine_file,
    const InferOption& infer_option)
    : model_(std::make_unique<DetectModel>(trt_engine_file, infer_option)) {
    LOGI("TensorRT detector initialized: %s", trt_engine_file.c_str());
}

SliceDetector::~SliceDetector() = default;

DetectRes SliceDetector::process_sliced_images(
    const std::vector<SliceInfo>& slice_infos,
    float nms_threshold,
    float conf_threshold,
    const std::vector<int>& allowed_class_indices) {
    if (slice_infos.empty()) return {};
    if (!model_) throw std::runtime_error("slice detector model is not initialized");
    if (nms_threshold <= 0.0f || conf_threshold <= 0.0f) {
        throw std::invalid_argument("slice detector thresholds must be positive");
    }

    const int model_batch_size = model_->batch();
    if (model_batch_size <= 0) {
        throw std::runtime_error("slice detector reported a non-positive batch size");
    }

    struct IndexedResult {
        std::size_t slice_index = 0;
        DetectRes result;
    };
    std::vector<IndexedResult> indexed_results;
    indexed_results.reserve(slice_infos.size());

    const std::size_t batch_size = static_cast<std::size_t>(model_batch_size);
    for (std::size_t begin = 0; begin < slice_infos.size(); begin += batch_size) {
        const std::size_t end = std::min(begin + batch_size, slice_infos.size());
        std::vector<Image> images;
        std::vector<std::size_t> source_indices;
        images.reserve(end - begin);
        source_indices.reserve(end - begin);

        for (std::size_t index = begin; index < end; ++index) {
            const cv::Mat& slice = slice_infos[index].slice;
            if (slice.empty()) continue;
            images.emplace_back(slice.data, slice.cols, slice.rows);
            source_indices.push_back(index);
        }
        if (images.empty()) continue;

        auto batch_results = model_->predict(images);
        const std::size_t paired_count = std::min(batch_results.size(), source_indices.size());
        if (batch_results.size() != source_indices.size()) {
            LOGE("slice detector batch result mismatch: requested=%zu returned=%zu",
                 source_indices.size(), batch_results.size());
        }
        for (std::size_t i = 0; i < paired_count; ++i) {
            indexed_results.push_back({source_indices[i], std::move(batch_results[i])});
        }
    }

    const bool filter_classes = !allowed_class_indices.empty();
    const std::unordered_set<int> allowed_classes(
        allowed_class_indices.begin(), allowed_class_indices.end());

    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::vector<float> confidences;

    for (const auto& indexed : indexed_results) {
        const auto& result = indexed.result;
        const auto& slice_info = slice_infos[indexed.slice_index];
        const int count = std::min({
            result.num,
            static_cast<int>(result.boxes.size()),
            static_cast<int>(result.classes.size()),
            static_cast<int>(result.scores.size())});

        for (int i = 0; i < count; ++i) {
            const float score = result.scores[static_cast<std::size_t>(i)];
            if (score < conf_threshold) continue;

            const int class_id = result.classes[static_cast<std::size_t>(i)];
            if (filter_classes && !allowed_classes.contains(class_id)) continue;

            const auto& source_box = result.boxes[static_cast<std::size_t>(i)];
            const int x1 = std::max(0, static_cast<int>(source_box.left) + slice_info.offset.x);
            const int y1 = std::max(0, static_cast<int>(source_box.top) + slice_info.offset.y);
            const int x2 = std::max(0, static_cast<int>(source_box.right) + slice_info.offset.x);
            const int y2 = std::max(0, static_cast<int>(source_box.bottom) + slice_info.offset.y);
            if (x2 <= x1 || y2 <= y1) continue;

            boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
            class_ids.push_back(class_id);
            confidences.push_back(score);
        }
    }

    if (boxes.empty()) return {};

    // NMS remains class-agnostic to preserve current model behavior. If classes
    // may overlap spatially, consider per-class NMS after validating metrics.
    std::vector<int> kept_indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, kept_indices);

    DetectRes final_result;
    final_result.num = static_cast<int>(kept_indices.size());
    final_result.boxes.reserve(kept_indices.size());
    final_result.classes.reserve(kept_indices.size());
    final_result.scores.reserve(kept_indices.size());

    for (const int index : kept_indices) {
        const std::size_t i = static_cast<std::size_t>(index);
        const cv::Rect& box = boxes[i];
        final_result.classes.push_back(class_ids[i]);
        final_result.scores.push_back(confidences[i]);
        final_result.boxes.emplace_back(
            static_cast<float>(box.x),
            static_cast<float>(box.y),
            static_cast<float>(box.x + box.width),
            static_cast<float>(box.y + box.height));
    }
    return final_result;
}

void SliceDetector::visualize_sliced_result(
    cv::Mat& image,
    const DetectRes& result,
    const std::vector<std::string>& labels) {
    if (image.empty() || result.num <= 0) return;

    const int count = std::min({
        result.num,
        static_cast<int>(result.boxes.size()),
        static_cast<int>(result.classes.size()),
        static_cast<int>(result.scores.size())});
    for (int i = 0; i < count; ++i) {
        const int class_id = result.classes[static_cast<std::size_t>(i)];
        if (class_id < 0 || class_id >= static_cast<int>(labels.size())) continue;

        const auto& box = result.boxes[static_cast<std::size_t>(i)];
        const std::string text =
            labels[static_cast<std::size_t>(class_id)] + " " +
            cv::format("%.3f", result.scores[static_cast<std::size_t>(i)]);

        int baseline = 0;
        const cv::Size label_size = cv::getTextSize(
            text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
        const int label_x = std::max(0, static_cast<int>(box.left));
        int label_y = static_cast<int>(box.top) - baseline;
        if (label_y < label_size.height) label_y = static_cast<int>(box.top) + label_size.height;

        cv::rectangle(
            image,
            cv::Point(static_cast<int>(box.left), static_cast<int>(box.top)),
            cv::Point(static_cast<int>(box.right), static_cast<int>(box.bottom)),
            cv::Scalar(251, 81, 163), 2, cv::LINE_AA);
        cv::rectangle(
            image,
            cv::Point(label_x, label_y - label_size.height),
            cv::Point(label_x + label_size.width, label_y + baseline),
            cv::Scalar(125, 40, 81), -1);
        cv::putText(
            image, text, cv::Point(label_x, label_y),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(253, 168, 208), 1, cv::LINE_AA);
    }
}

} // namespace trtyolo
