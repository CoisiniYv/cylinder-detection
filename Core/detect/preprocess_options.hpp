#pragma once

#include <string>

// Configuration for the shared GPU preprocessing stage.
// Grouping the options prevents call sites from depending on a long positional
// argument list where adjacent numeric values are easy to swap accidentally.
struct PreprocessOptions {
    struct Crop {
        bool enabled = false;
        int x = 0;
        int y = 0;
        int width = 0;
        int height = 0;
    } crop;

    struct QwMask {
        bool enabled = false;
        int x1 = 0;
        int y1 = 0;
        int x2 = 0;
        int y2 = 0;
        int radius = 0;
    } qw_mask;

    struct StripeRemoval {
        bool enabled = false;
        int filter_width = 10;
        double attenuation_factor = 0.0001;
        int target_angle = 90;
        int angle_tolerance = 10;
        bool denoise = false;
        float denoise_h = 5.0f;
        float denoise_h_color = 8.0f;
        int denoise_search_window = 17;
        int denoise_template_window = 11;
    } stripe;

    struct Save {
        std::string file_name = "cropped_image";
        std::string output_dir;
    } save;
};
