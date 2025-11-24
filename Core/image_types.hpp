#pragma once
#include <opencv2/core.hpp>
#include <array>
#include <cstdint>
#include <string>
#include <vector>


namespace XL {

	// 四张图为一组，固定大小常量
	constexpr std::size_t kQuadImageCount = 4;

	// 单张图的基础元数据
	struct FrameMeta {
		// 设备/相机标识，可为空
		std::string device_id;
		// 所属组 ID（默认四张图为一组），用于下游分组与判定
		std::uint64_t group_id = 0;
		// 组内序号（0~3），不在组内时为 -1
		std::int8_t index_in_group = -1;
	};

	// 单张图：
	// - ps_image：由 0/1/2 通道合成的光度立体图片（BGR，CV_8UC3）
	// - rgb_image：由 3/4/5 通道合成的普通 RGB 图片（BGR，CV_8UC3）
	struct ImageFrame {
		cv::Mat ps_image;   // 光度立体图（012）
		cv::Mat rgb_image;  // 普通 RGB 图（345）

		FrameMeta meta;     // 基础元数据

		inline bool has_ps() const noexcept { return !ps_image.empty(); }
		inline bool has_rgb() const noexcept { return !rgb_image.empty(); }
		inline bool valid() const noexcept { return has_ps() && has_rgb(); }

		// 若两张图尺寸一致，则返回统一的宽高；否则返回 -1 表示不一致或不存在
		inline int width() const noexcept {
			if (has_rgb()) return rgb_image.cols;
			if (has_ps())  return ps_image.cols;
			return -1;
		}
		inline int height() const noexcept {
			if (has_rgb()) return rgb_image.rows;
			if (has_ps())  return ps_image.rows;
			return -1;
		}

		inline void clear() {
			ps_image.release();
			rgb_image.release();
		}
	};

	// 四张图组：按顺序保存四张图，统一携带组级元数据
	struct QuadFrame {
		std::array<ImageFrame, kQuadImageCount> frames;

		// 组级元数据
		std::uint64_t group_id = 0;        // 组唯一标识（可由上层生成）

		inline ImageFrame& at(std::size_t i) { return frames.at(i); }
		inline const ImageFrame& at(std::size_t i) const { return frames.at(i); }

		// 组内有效性校验（检查 4 张图是否都已合成完成）
		inline bool valid() const noexcept {
			for (std::size_t i = 0; i < kQuadImageCount; ++i) {
				if (!frames[i].valid()) return false;
			}
			return true;
		}
	};

	// 基础检测框结构（像素坐标）
	struct BBox {
		std::int32_t x = 0;
		std::int32_t y = 0;
		std::int32_t w = 0;
		std::int32_t h = 0;

		inline int x2() const noexcept { return x + w; }
		inline int y2() const noexcept { return y + h; }
		inline bool empty() const noexcept { return w <= 0 || h <= 0; }
	};

	// 单项检测结果
	struct Detection {
		std::int8_t label_id = -1;        // 类别 id
		float confidence = 0.0f;           // 置信度 [0,1]
		BBox box;                          // 检测框（像素）
		double length = 0.0;               // 缺陷长度（mm），默认0
		double area = 0.0;                 // 缺陷面积（mm^2），默认0
	};

	// 单张图的分析结果
	struct SingleImageResult {
		// 对应的图像元信息
		FrameMeta meta;
		// 检测列表
		std::vector<Detection> detections;

		// 保存路径
		std::string saved_path;          // 原图或可视化结果的保存路径
		std::string skeleton_path;       // 骨架图保存路径
		std::string original_path;         // 原始图像路径
	};

	// 四张图组的分析结果
	struct QuadFrameResult {
		QuadFrame source;                                      // 原始四图组
		std::array<SingleImageResult, kQuadImageCount> results; // 四张图对应结果

		// 组级聚合信息
		inline std::size_t totalDetections() const noexcept {
			std::size_t n = 0;
			for (const auto& r : results) n += r.detections.size();
			return n;
		}
	};

} // namespace XL