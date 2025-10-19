#include "trtyolo_slice.hpp"
#include <iostream>
#include <algorithm>
#include <unordered_set>

namespace trtyolo {

    // SliceDetector 实现

    SliceDetector::SliceDetector(std::unique_ptr<DetectModel> model)
        : model_(std::move(model)) {
    }

    SliceDetector::SliceDetector(const std::string& trt_engine_file, const InferOption& infer_option)
        : model_(std::make_unique<DetectModel>(trt_engine_file, infer_option)) {
    }

    SliceDetector::~SliceDetector() = default;

    DetectRes SliceDetector::process_sliced_images(
        const std::vector<SliceInfo>& slice_infos,
        float nms_threshold,
        float conf_threshold,
        const std::vector<int>& allowed_class_indices) {

        if (slice_infos.empty()) {
            return DetectRes();
        }

        if (!model_) {
            throw std::runtime_error("Model is not initialized");
        }

        const int batch_size = model_->batch();
        std::vector<DetectRes> all_results;
        all_results.reserve((slice_infos.size() + batch_size - 1) / batch_size * batch_size);

        // 批量处理切片图像
        for (size_t i = 0; i < slice_infos.size(); i += batch_size) {
            const size_t this_batch = std::min(static_cast<size_t>(batch_size), slice_infos.size() - i);
            std::vector<Image> img_batch;
            img_batch.reserve(this_batch);

            for (size_t j = 0; j < this_batch; ++j) {
                const cv::Mat& slice = slice_infos[i + j].slice;
                // 跳过空切片（安全防护）
                if (slice.empty()) {
                    continue;
                }
                img_batch.emplace_back(slice.data, slice.cols, slice.rows);
            }

            if (img_batch.empty()) {
                continue;
            }

            // 批量推理
            auto batch_results = model_->predict(img_batch);

            // 将结果添加到总结果中
            all_results.insert(all_results.end(), batch_results.begin(), batch_results.end());
        }

        // 收集所有检测框并转换到原图坐标（并按类别进行过滤）
        std::vector<cv::Rect> boxes;
        std::vector<int> class_ids;
        std::vector<float> confidences;

        // 准备类别过滤集合（为空表示不过滤）
        const bool use_class_filter = !allowed_class_indices.empty();
        std::unordered_set<int> allowed_classes;
        if (use_class_filter) {
            allowed_classes.insert(allowed_class_indices.begin(), allowed_class_indices.end());
        }

        for (size_t slice_idx = 0; slice_idx < all_results.size() && slice_idx < slice_infos.size(); ++slice_idx) {
            const auto& result = all_results[slice_idx];
            const auto& slice_info = slice_infos[slice_idx];

            for (size_t i = 0; i < result.num; ++i) {
                // 过滤低置信度检测
                if (result.scores[i] < conf_threshold) {
                    continue;
                }

                const auto& box = result.boxes[i];
                int cls = result.classes[i];
                float score = result.scores[i];

                // 过滤不在允许类别集合中的检测
                if (use_class_filter && allowed_classes.find(cls) == allowed_classes.end()) {
                    continue;
                }

                // 将框坐标转换到原图坐标
                int x1 = static_cast<int>(box.left) + slice_info.offset.x;
                int y1 = static_cast<int>(box.top) + slice_info.offset.y;
                int x2 = static_cast<int>(box.right) + slice_info.offset.x;
                int y2 = static_cast<int>(box.bottom) + slice_info.offset.y;

                // 确保坐标在合理范围内
                x1 = std::max(0, x1);
                y1 = std::max(0, y1);
                x2 = std::max(0, x2);
                y2 = std::max(0, y2);

                // 确保宽度和高度为正
                int width = x2 - x1;
                int height = y2 - y1;
                if (width <= 0 || height <= 0) {
                    continue;
                }

                cv::Rect transformed_box(x1, y1, width, height);

                boxes.push_back(transformed_box);
                class_ids.push_back(cls);
                confidences.push_back(score);
            }
        }

        // 如果没有检测到任何目标，返回空结果
        if (boxes.empty()) {
            return DetectRes();
        }

        // 应用NMS
        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_indices);

        // 构建最终结果
        DetectRes final_result;
        final_result.num = static_cast<int>(nms_indices.size());

        for (int idx : nms_indices) {
            final_result.classes.push_back(class_ids[idx]);
            final_result.scores.push_back(confidences[idx]);

            // 将cv::Rect转换回trtyolo::Box
            const cv::Rect& box = boxes[idx];
            final_result.boxes.emplace_back(
                static_cast<float>(box.x),
                static_cast<float>(box.y),
                static_cast<float>(box.x + box.width),
                static_cast<float>(box.y + box.height)
            );
        }

        return final_result;
    }

    void SliceDetector::visualize_sliced_result(
        cv::Mat& image,
        const DetectRes& result,
        const std::vector<std::string>& labels) {

        if (image.empty() || result.num == 0) {
            return;
        }

        for (size_t i = 0; i < result.num; ++i) {
            const auto& box = result.boxes[i];
            int cls = result.classes[i];
            float score = result.scores[i];

            // 确保类别索引在有效范围内
            if (cls < 0 || cls >= static_cast<int>(labels.size())) {
                continue;
            }

            const auto& label = labels[cls];
            std::string label_text = label + " " + cv::format("%.3f", score);

            // 绘制矩形和标签
            int base_line;
            cv::Size label_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &base_line);

            // 计算标签位置
            int label_x = static_cast<int>(box.left);
            int label_y = static_cast<int>(box.top) - base_line;

            // 确保标签在图像范围内
            if (label_y < 0) {
                label_y = static_cast<int>(box.top) + base_line;
            }

            // 绘制检测框
            cv::rectangle(image,
                cv::Point(static_cast<int>(box.left), static_cast<int>(box.top)),
                cv::Point(static_cast<int>(box.right), static_cast<int>(box.bottom)),
                cv::Scalar(251, 81, 163), 2, cv::LINE_AA);

            // 绘制标签背景
            cv::rectangle(image,
                cv::Point(label_x, label_y - label_size.height),
                cv::Point(label_x + label_size.width, label_y),
                cv::Scalar(125, 40, 81), -1);

            // 绘制标签文本
            cv::putText(image, label_text,
                cv::Point(label_x, label_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(253, 168, 208), 1);
        }
    }

    DetectModel* SliceDetector::get_model() const {
        return model_.get();
    }

} // namespace trtyolo