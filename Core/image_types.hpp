#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace XL {

inline constexpr std::size_t kQuadImageCount = 4;

struct FrameMeta {
    std::string device_id;
    std::uint64_t group_id = 0;
    int index_in_group = -1;
};

struct ImageFrame {
    cv::Mat ps_image;   // photometric-stereo image, BGR CV_8UC3
    cv::Mat rgb_image;  // regular RGB source represented as BGR CV_8UC3
    FrameMeta meta;

    bool has_ps() const noexcept { return !ps_image.empty(); }
    bool has_rgb() const noexcept { return !rgb_image.empty(); }
    bool valid() const noexcept { return has_ps() && has_rgb(); }

    int width() const noexcept {
        if (has_rgb()) return rgb_image.cols;
        if (has_ps()) return ps_image.cols;
        return -1;
    }

    int height() const noexcept {
        if (has_rgb()) return rgb_image.rows;
        if (has_ps()) return ps_image.rows;
        return -1;
    }

    void clear() noexcept {
        ps_image.release();
        rgb_image.release();
    }
};

struct QuadFrame {
    std::array<ImageFrame, kQuadImageCount> frames;
    std::uint64_t group_id = 0;

    ImageFrame& at(std::size_t index) { return frames.at(index); }
    const ImageFrame& at(std::size_t index) const { return frames.at(index); }

    bool valid() const noexcept {
        for (const auto& frame : frames) {
            if (!frame.valid()) return false;
        }
        return true;
    }
};

struct BBox {
    std::int32_t x = 0;
    std::int32_t y = 0;
    std::int32_t w = 0;
    std::int32_t h = 0;

    int x2() const noexcept { return x + w; }
    int y2() const noexcept { return y + h; }
    bool empty() const noexcept { return w <= 0 || h <= 0; }
};

struct Detection {
    int label_id = -1;
    float confidence = 0.0f;
    BBox box;
    double length = 0.0; // mm
    double area = 0.0;   // mm^2
};

struct SingleImageResult {
    FrameMeta meta;
    std::vector<Detection> detections;

    std::string saved_path;
    std::string skeleton_path;
    std::string original_path;

    // Infrastructure/algorithm failures are distinct from a valid GOOD image
    // with zero detections. This prevents dropped frames from becoming a
    // mysterious GroupManager timeout.
    bool processing_ok = true;
    std::string error_message;
};

struct QuadFrameResult {
    QuadFrame source;
    std::array<SingleImageResult, kQuadImageCount> results;

    std::size_t totalDetections() const noexcept {
        std::size_t count = 0;
        for (const auto& result : results) count += result.detections.size();
        return count;
    }

    bool processingOk() const noexcept {
        for (const auto& result : results) {
            if (!result.processing_ok) return false;
        }
        return true;
    }
};

} // namespace XL
