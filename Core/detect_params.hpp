#pragma once

#include <string>
#include <vector>

namespace XL {

// Runtime parameters for one detection task.
//
// This type is intentionally independent from the HTTP server so camera,
// scheduler and inference code do not depend on libevent/WinSock headers.
struct DetectParams {
    // Detection model.
    std::string model_name;
    std::string engine_file;

    // SAM models. Relative names are resolved against Config::modelDir by Server.
    std::string sam_encoder_name = "SAM/SAM_encoder.engine";
    std::string sam_decoder_name = "SAM/SAM_mask_decoder.engine";
    std::string sam_encoder_onnx_name = "SAM/SAM_encoder.onnx";
    std::string sam_decoder_onnx_name = "SAM/SAM_mask_decoder.onnx";
    std::string sam_encoder_engine_file;
    std::string sam_decoder_engine_file;
    std::string sam_encoder_onnx_file;
    std::string sam_decoder_onnx_file;

    // Single-image API input.
    std::string input_image_path;

    // ROI and QW masking.
    bool enable_four_side_crop = false;
    int crop_x = 0;
    int crop_y = 0;
    int crop_width = 0;
    int crop_height = 0;
    bool is_qw = false;
    int circle_x1 = 0;
    int circle_y1 = 0;
    int circle_x2 = 0;
    int circle_y2 = 0;
    int radius = 0;

    // Stripe removal and denoising.
    bool enable_fourier_transform = false;
    int filter_width = 10;
    double attenuation_factor = 0.0001;
    int target_angle = 90;
    int angle_tolerance = 10;
    bool enable_denoising = false;
    float denoise_h = 5.0f;
    float denoise_hColor = 8.0f;
    int denoise_search_window = 17;
    int denoise_template_window = 11;

    // Slicing and detection.
    int slice_width = 640;
    int slice_height = 640;
    int slice_distance = 40;
    float nms_threshold = 0.45f;
    float conf_threshold = 0.20f;
    std::vector<int> allowed_class_indices{0, 2};

    // SAM measurement filters.
    double pix_to_mm = 0.021;
    double min_area_mm2 = 0.0;
    double min_diameter_mm = 0.0;

    // Camera/task identity.
    int delay_ms = 0;
    int qw_index = 0;
    std::string device_id = "CAM-001";

    // Runtime execution placement.
    int gpu_device = 0;
};

} // namespace XL
