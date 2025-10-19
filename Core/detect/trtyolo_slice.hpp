#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include "trtyolo.hpp"

namespace trtyolo {

    /**
     * @brief 切片信息结构体，封装切片图像和对应的偏移量
     */
    struct TRTYOLOAPI SliceInfo {
        cv::Mat slice;      // 切片图像
        cv::Point offset;   // 切片在原图中的偏移量 (x, y)

        /**
         * @brief 默认构造函数
         */
        SliceInfo() = default;

        /**
         * @brief 构造函数
         *
         * @param img 切片图像
         * @param off 偏移量
         */
        SliceInfo(const cv::Mat& img, const cv::Point& off)
            : slice(img), offset(off) {
        }

        /**
         * @brief 移动构造函数
         *
         * @param other 其他SliceInfo对象
         */
        SliceInfo(SliceInfo&& other) noexcept
            : slice(std::move(other.slice)), offset(other.offset) {
        }

        /**
         * @brief 拷贝构造函数
         *
         * @param other 其他SliceInfo对象
         */
        SliceInfo(const SliceInfo& other)
            : slice(other.slice.clone()), offset(other.offset) {
        }

        /**
         * @brief 拷贝赋值运算符
         *
         * @param other 其他SliceInfo对象
         * @return SliceInfo& 当前对象引用
         */
        SliceInfo& operator=(const SliceInfo& other) {
            if (this != &other) {
                slice = other.slice.clone();
                offset = other.offset;
            }
            return *this;
        }

        /**
         * @brief 移动赋值运算符
         *
         * @param other 其他SliceInfo对象
         * @return SliceInfo& 当前对象引用
         */
        SliceInfo& operator=(SliceInfo&& other) noexcept {
            if (this != &other) {
                slice = std::move(other.slice);
                offset = other.offset;
            }
            return *this;
        }
    };

    /**
     * @brief 切片检测工具类
     */
    class TRTYOLOAPI SliceDetector {
    public:
        /**
         * @brief 默认构造函数
         */
        SliceDetector() = default;

        /**
         * @brief 构造函数
         *
         * @param model 检测模型
         */
        explicit SliceDetector(std::unique_ptr<DetectModel> model);

        /**
         * @brief 构造函数
         *
         * @param trt_engine_file TensorRT引擎文件路径
         * @param infer_option 推理选项
         */
        SliceDetector(const std::string& trt_engine_file, const InferOption& infer_option);

        /**
         * @brief 析构函数
         */
        ~SliceDetector();

        /**
         * @brief 处理切片图像并进行检测，将结果转换到原图坐标并应用NMS
         *
         * @param slice_infos 切片信息向量
         * @param nms_threshold NMS阈值
         * @param conf_threshold 置信度阈值
         * @return DetectRes 原图上的检测结果
         */
        DetectRes process_sliced_images(
            const std::vector<SliceInfo>& slice_infos,
            float nms_threshold = 0.5f,
            float conf_threshold = 0.25f,
            const std::vector<int>& allowed_class_indices = std::vector<int>());

        /**
         * @brief 在图像上绘制切片检测结果
         *
         * @param image 原图像
         * @param result 检测结果
         * @param labels 标签向量
         */
        static void visualize_sliced_result(
            cv::Mat& image,
            const DetectRes& result,
            const std::vector<std::string>& labels);

        /**
         * @brief 获取模型指针
         *
         * @return DetectModel* 模型指针
         */
        DetectModel* get_model() const;

    private:
        std::unique_ptr<DetectModel> model_;  // 检测模型
    };

} // namespace trtyolo