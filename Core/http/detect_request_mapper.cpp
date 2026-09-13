#include "detect_request_mapper.hpp"

#include "../Config.hpp"
#include "../detect_params.hpp"
#include "../image_types.hpp"
#include "request_params.hpp"

#include <filesystem>
#include <string>

namespace XL::http {
namespace {

std::string resolvePath(const std::string& base, const std::string& value) {
    if (value.empty()) return {};
    std::filesystem::path path(value);
    if (path.is_absolute()) return path.lexically_normal().string();
    if (base.empty()) return path.lexically_normal().string();
    return (std::filesystem::path(base) / path).lexically_normal().string();
}

void resolveModelPaths(DetectParams& params, const Config* config) {
    const std::string model_dir = config ? config->modelDir : std::string();
    params.engine_file = resolvePath(
        model_dir,
        params.engine_file.empty() ? params.model_name : params.engine_file);
    params.sam_encoder_engine_file = resolvePath(model_dir, params.sam_encoder_name);
    params.sam_decoder_engine_file = resolvePath(model_dir, params.sam_decoder_name);
    params.sam_encoder_onnx_file = resolvePath(model_dir, params.sam_encoder_onnx_name);
    params.sam_decoder_onnx_file = resolvePath(model_dir, params.sam_decoder_onnx_name);
}

bool validate(const DetectParams& params, std::string& error_message) {
    if (params.engine_file.empty()) {
        error_message = "model_name/trt_engine_file is required";
        return false;
    }
    if (params.nms_threshold <= 0.0f || params.nms_threshold > 1.0f ||
        params.conf_threshold <= 0.0f || params.conf_threshold > 1.0f) {
        error_message = "nms_threshold/conf_threshold must be in (0, 1]";
        return false;
    }
    if (params.slice_width <= 0 || params.slice_height <= 0 || params.slice_distance < 0) {
        error_message =
            "slice dimensions must be positive and slice_distance must be non-negative";
        return false;
    }
    if (params.qw_index < 0 || params.qw_index >= static_cast<int>(kQuadImageCount)) {
        error_message = "qw_index must be in [0, 3]";
        return false;
    }
    if (params.gpu_device < 0) {
        error_message = "gpu_device must be non-negative";
        return false;
    }
    if (params.pix_to_mm < 0.0 || params.min_area_mm2 < 0.0 || params.min_diameter_mm < 0.0) {
        error_message = "measurement thresholds must be non-negative";
        return false;
    }
    if (params.filter_width < 0 || params.angle_tolerance < 0 || params.angle_tolerance > 180) {
        error_message = "invalid stripe-removal parameters";
        return false;
    }
    if (params.attenuation_factor < 0.0 || params.attenuation_factor > 1.0) {
        error_message = "attenuation_factor must be in [0, 1]";
        return false;
    }
    if (params.enable_four_side_crop &&
        (params.crop_x < 0 || params.crop_y < 0 ||
         params.crop_width <= 0 || params.crop_height <= 0)) {
        error_message = "enabled ROI crop requires non-negative origin and positive size";
        return false;
    }
    if (params.is_qw &&
        (params.circle_x1 < 0 || params.circle_y1 < 0 ||
         params.circle_x2 < 0 || params.circle_y2 < 0 || params.radius <= 0)) {
        error_message = "enabled QW mask requires valid centers and positive radius";
        return false;
    }
    return true;
}

} // namespace

bool parseDetectParams(
    evhttp_request* request,
    const Config* config,
    DetectParams& params,
    std::string& error_message) {
    RequestParams values;
    if (!values.load(request, error_message)) return false;

    params.model_name = values.stringValue(
        "model_name",
        values.stringValue("yolo_model_name", params.model_name));
    params.engine_file = values.stringValue("trt_engine_file", params.engine_file);

    params.sam_encoder_name = values.stringValue("sam_encoder_name", params.sam_encoder_name);
    params.sam_decoder_name = values.stringValue("sam_decoder_name", params.sam_decoder_name);
    params.sam_encoder_onnx_name = values.stringValue(
        "sam_encoder_onnx_name",
        params.sam_encoder_onnx_name);
    params.sam_decoder_onnx_name = values.stringValue(
        "sam_decoder_onnx_name",
        params.sam_decoder_onnx_name);
    params.input_image_path = values.stringValue("input_image_path", params.input_image_path);

    params.enable_four_side_crop = values.boolValue(
        "enable_four_side_crop",
        params.enable_four_side_crop);
    params.crop_x = values.intValue("crop_x", params.crop_x);
    params.crop_y = values.intValue("crop_y", params.crop_y);
    params.crop_width = values.intValue("crop_width", params.crop_width);
    params.crop_height = values.intValue("crop_height", params.crop_height);
    params.is_qw = values.boolValue("is_qw", params.is_qw);
    params.circle_x1 = values.intValue("circle_x1", params.circle_x1);
    params.circle_y1 = values.intValue("circle_y1", params.circle_y1);
    params.circle_x2 = values.intValue("circle_x2", params.circle_x2);
    params.circle_y2 = values.intValue("circle_y2", params.circle_y2);
    params.radius = values.intValue("radius", params.radius);

    params.enable_fourier_transform = values.boolValue(
        "enable_fourier_transform",
        params.enable_fourier_transform);
    params.filter_width = values.intValue("filter_width", params.filter_width);
    params.attenuation_factor = values.doubleValue(
        "attenuation_factor",
        params.attenuation_factor);
    params.target_angle = values.intValue("target_angle", params.target_angle);
    params.angle_tolerance = values.intValue("angle_tolerance", params.angle_tolerance);
    params.enable_denoising = values.boolValue("enable_denoising", params.enable_denoising);
    params.denoise_h = values.floatValue("denoise_h", params.denoise_h);
    params.denoise_hColor = values.floatValue("denoise_hColor", params.denoise_hColor);
    params.denoise_search_window = values.intValue(
        "denoise_search_window",
        params.denoise_search_window);
    params.denoise_template_window = values.intValue(
        "denoise_template_window",
        params.denoise_template_window);

    params.slice_width = values.intValue("slice_width", params.slice_width);
    params.slice_height = values.intValue("slice_height", params.slice_height);
    params.slice_distance = values.intValue("slice_distance", params.slice_distance);
    params.nms_threshold = values.floatValue("nms_threshold", params.nms_threshold);
    params.conf_threshold = values.floatValue("conf_threshold", params.conf_threshold);
    params.allowed_class_indices = values.intListValue(
        "allowed_classes",
        params.allowed_class_indices);

    params.pix_to_mm = values.doubleValue("pix_to_mm", params.pix_to_mm);
    params.min_area_mm2 = values.doubleValue("min_area_mm2", params.min_area_mm2);
    params.min_diameter_mm = values.doubleValue(
        "min_diameter_mm",
        params.min_diameter_mm);

    params.delay_ms = values.intValue("delay_ms", params.delay_ms);
    params.qw_index = values.intValue("qw_index", params.qw_index);
    params.device_id = values.stringValue("device_id", params.device_id);
    params.gpu_device = values.intValue(
        "gpu_device",
        config ? config->gpuDevice : params.gpu_device);

    resolveModelPaths(params, config);
    if (params.device_id.empty()) params.device_id = "CAM-001";
    return validate(params, error_message);
}

} // namespace XL::http
