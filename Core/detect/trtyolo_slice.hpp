#pragma once

#include "trtyolo.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace trtyolo {

// One image slice and its top-left offset in the source image.
// Copies are intentionally deep because inference may outlive a temporary ROI.
struct TRTYOLOAPI SliceInfo {
    cv::Mat slice;
    cv::Point offset;

    SliceInfo() = default;
    SliceInfo(const cv::Mat& image, const cv::Point& source_offset)
        : slice(image), offset(source_offset) {}

    SliceInfo(SliceInfo&&) noexcept = default;
    SliceInfo& operator=(SliceInfo&&) noexcept = default;

    SliceInfo(const SliceInfo& other)
        : slice(other.slice.clone()), offset(other.offset) {}

    SliceInfo& operator=(const SliceInfo& other) {
        if (this != &other) {
            slice = other.slice.clone();
            offset = other.offset;
        }
        return *this;
    }
};

// Adapter around DetectModel that batches slices, maps detections back to source
// coordinates and applies final NMS/class filtering.
class TRTYOLOAPI SliceDetector {
public:
    SliceDetector() = default;
    explicit SliceDetector(std::unique_ptr<DetectModel> model);
    SliceDetector(const std::string& trt_engine_file, const InferOption& infer_option);
    ~SliceDetector();

    SliceDetector(const SliceDetector&) = delete;
    SliceDetector& operator=(const SliceDetector&) = delete;

    DetectRes process_sliced_images(
        const std::vector<SliceInfo>& slice_infos,
        float nms_threshold = 0.5f,
        float conf_threshold = 0.25f,
        const std::vector<int>& allowed_class_indices = {});

    static void visualize_sliced_result(
        cv::Mat& image,
        const DetectRes& result,
        const std::vector<std::string>& labels);

    DetectModel* get_model() const noexcept { return model_.get(); }

private:
    std::unique_ptr<DetectModel> model_;
};

} // namespace trtyolo
