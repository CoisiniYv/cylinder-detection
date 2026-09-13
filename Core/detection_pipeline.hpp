#pragma once

#include "detect_params.hpp"
#include "detection_context.hpp"
#include "image_types.hpp"

#include <memory>
#include <string>

namespace XL {

// Shared image-processing pipeline used by both multi-face and single-image
// workers. It owns device-bound model state and the HALCON processor, while
// callers remain responsible for thread/queue/request lifecycle.
class DetectionPipeline {
public:
    explicit DetectionPipeline(int pipeline_id = -1);
    ~DetectionPipeline();

    DetectionPipeline(const DetectionPipeline&) = delete;
    DetectionPipeline& operator=(const DetectionPipeline&) = delete;

    // Applies execution/model configuration. Reuses compatible model instances
    // and recreates only resources affected by GPU/model changes.
    bool configure(const DetectParams& params, std::string& error_message);

    // Multi-face path. Prefers the photometric-stereo frame when available and
    // writes output into <output_root>/<run_id>/<group_id>/.
    SingleImageResult processFrame(
        const ImageFrame& frame,
        const DetectionContext& context);

    // Single-image path. Reads the file and writes generated artifacts into
    // <output_root>/<run_id>/single/.
    SingleImageResult processFile(
        const std::string& input_file,
        const DetectionContext& context);

    bool configured() const noexcept;
    int gpuDevice() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace XL
