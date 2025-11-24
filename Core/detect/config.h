#pragma once

#define USE_FP16 ///< 设置为使用FP16（半精度浮点数）精度，或注释此行以使用FP32（单精度浮点数）精度

#define MAX_NUM_PROMPTS 1 ///< 单次处理的最大提示词数量

// 模型参数
#define MODEL_INPUT_WIDTH 1024.0f ///< 模型输入的像素宽度
#define MODEL_INPUT_HEIGHT 1024.0f ///< 模型输入的像素高度
#define HIDDEN_DIM 256 ///< 隐藏层维度
#define NUM_LABELS 4 ///< 输出标签数量
#define FEATURE_WIDTH 64 ///< 特征图宽度
#define FEATURE_HEIGHT 64 ///< 特征图高度
