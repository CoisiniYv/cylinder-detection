#include "speedSam.h"

#include "config.h"

#include <algorithm>
#include <stdexcept>

SpeedSam::SpeedSam(const std::string& encoder_path, const std::string& decoder_path)
    : mFeatures(HIDDEN_DIM * FEATURE_WIDTH * FEATURE_HEIGHT),
      mMaskInput(HIDDEN_DIM * HIDDEN_DIM, 0.0f),
      mIouPrediction(NUM_LABELS),
      mLowResMasks(NUM_LABELS * HIDDEN_DIM * HIDDEN_DIM) {
    mImageEncoder = std::make_unique<EngineTRT>(
        encoder_path,
        std::vector<std::string>{"image"},
        std::vector<std::string>{"image_embeddings"},
        false,
        true);

    mMaskDecoder = std::make_unique<EngineTRT>(
        decoder_path,
        std::vector<std::string>{
            "image_embeddings", "point_coords", "point_labels", "mask_input", "has_mask_input"},
        std::vector<std::string>{"iou_predictions", "low_res_masks"},
        true,
        false);
}

cv::Mat SpeedSam::predict(
    cv::Mat& image,
    const std::vector<cv::Point>& points,
    const std::vector<float>& labels) {
    if (image.empty()) throw std::invalid_argument("SpeedSam input image is empty");
    if (points.empty()) return cv::Mat::zeros(image.rows, image.cols, CV_32FC1);
    if (labels.size() != points.size()) {
        throw std::invalid_argument("SpeedSam points/labels size mismatch");
    }

    cv::Mat resized_image = resizeImage(image, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);
    mImageEncoder->setInput(resized_image);
    if (!mImageEncoder->infer()) throw std::runtime_error("SAM encoder inference failed");
    mImageEncoder->getOutput(mFeatures.data());

    std::vector<float> point_data(points.size() * 2);
    prepareDecoderInput(points, point_data.data(), image.cols, image.rows);

    mMaskDecoder->setInput(
        mFeatures.data(),
        point_data.data(),
        labels.data(),
        mMaskInput.data(),
        &mHasMaskInput,
        static_cast<int>(points.size()));
    if (!mMaskDecoder->infer()) throw std::runtime_error("SAM decoder inference failed");
    mMaskDecoder->getOutput(mIouPrediction.data(), mLowResMasks.data());

    cv::Mat mask(HIDDEN_DIM, HIDDEN_DIM, CV_32FC1, mLowResMasks.data());
    upscaleMask(mask, image.cols, image.rows);
    return mask.clone();
}

void SpeedSam::prepareDecoderInput(
    const std::vector<cv::Point>& points,
    float* point_data,
    int image_width,
    int image_height) {
    if (!point_data || image_width <= 0 || image_height <= 0) {
        throw std::invalid_argument("invalid SAM decoder input");
    }

    const float scale = MODEL_INPUT_WIDTH /
        static_cast<float>(std::max(image_width, image_height));
    for (std::size_t i = 0; i < points.size(); ++i) {
        point_data[i * 2] = static_cast<float>(points[i].x) * scale;
        point_data[i * 2 + 1] = static_cast<float>(points[i].y) * scale;
    }

    std::fill(mMaskInput.begin(), mMaskInput.end(), 0.0f);
    mHasMaskInput = 0.0f;
}

cv::Mat SpeedSam::resizeImage(const cv::Mat& image, int input_width, int input_height) {
    if (image.empty() || input_width <= 0 || input_height <= 0) {
        throw std::invalid_argument("invalid SAM resize input");
    }

    const float aspect_ratio = static_cast<float>(image.cols) / static_cast<float>(image.rows);
    int resized_width = 0;
    int resized_height = 0;
    if (aspect_ratio >= 1.0f) {
        resized_width = input_width;
        resized_height = std::max(1, static_cast<int>(input_height / aspect_ratio));
    }
    else {
        resized_width = std::max(1, static_cast<int>(input_width * aspect_ratio));
        resized_height = input_height;
    }

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_width, resized_height), 0, 0, cv::INTER_LINEAR);
    cv::Mat output = cv::Mat::zeros(input_height, input_width, CV_8UC3);
    resized.copyTo(output(cv::Rect(0, 0, resized.cols, resized.rows)));
    return output;
}

void SpeedSam::upscaleMask(cv::Mat& mask, int target_width, int target_height, int size) {
    if (mask.empty() || target_width <= 0 || target_height <= 0 || size <= 0) {
        throw std::invalid_argument("invalid SAM mask upscale input");
    }

    int limit_x = 0;
    int limit_y = 0;
    if (target_width > target_height) {
        limit_x = size;
        limit_y = std::max(1, size * target_height / target_width);
    }
    else {
        limit_x = std::max(1, size * target_width / target_height);
        limit_y = size;
    }

    limit_x = std::min(limit_x, mask.cols);
    limit_y = std::min(limit_y, mask.rows);
    cv::Mat valid_region = mask(cv::Rect(0, 0, limit_x, limit_y));
    cv::resize(valid_region, mask, cv::Size(target_width, target_height));
}
