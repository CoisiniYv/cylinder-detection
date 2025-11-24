#pragma once

#include "Config.hpp"
#include "camera_thread.hpp"
#include <opencv2/imgcodecs.hpp>

#include <json/json.h>
#include <json/value.h>
#include <event2/event.h>
#include <event2/http.h>
#include <event2/buffer.h>
#include <event2/http_struct.h>
#include <event2/keyvalq_struct.h>
#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <WinSock2.h>
#include <WS2tcpip.h>
namespace XL {

	// 检测参数结构
	struct DetectParams {
		// 检测模型
		std::string model_name;           // 相对modelDir的model文件名(前端传入) 
		std::string engine_file;     // 服务器拼接后的完整路径

		// SAM模型
		std::string sam_encoder_name = "SAM\\SAM_encoder.engine"; // SAM模型名称
		std::string sam_decoder_name = "SAM\\SAM_mask_decoder.engine"; // SAM模型名称
		std::string sam_encoder_onnx_name = "SAM\\SAM_encoder.onnx"; // SAM_onnx模型名称
		std::string sam_decoder_onnx_name = "SAM\\SAM_mask_decoder.onnx"; // SAM_onnx模型名称
		std::string sam_encoder_engine_file; // 服务器拼接后的完整路径
		std::string sam_decoder_engine_file; // 服务器拼接后的完整路径
		std::string sam_encoder_onnx_file; // 服务器拼接后的完整路径
		std::string sam_decoder_onnx_file; // 服务器拼接后的完整路径

		std::string input_image_path;          // 待检测图像路径, 单张图片检测接口使用

		// ROI 裁剪与条纹线
		bool enable_four_side_crop = false; // 是否启用四边裁剪
		int crop_x = 0; // ROI 裁剪区域的左上角 x 坐标
		int crop_y = 0; // ROI 裁剪区域的左上角 y 坐标
		int crop_width = 0; // ROI 裁剪区域的宽度
		int crop_height = 0; // ROI 裁剪区域的高度
		bool is_qw = false; // 是否裁剪QW区域
		int circle_x1 = 0; // 第一个圆中心点对于原图的x位置
		int circle_y1 = 0; // 第一个圆中心点对于原图的y位置
		int circle_x2 = 0; // 第二个圆中心点对于原图的x位置
		int circle_y2 = 0; // 第二个圆中心点对于原图的y位置
		int radius = 0; // 圆裁剪区域的半径

		// 条纹去除与去噪参数
		bool enable_fourier_transform = false; // 是否启用傅里叶变换条纹去除
		int filter_width = 10;
		double attenuation_factor = 0.0001;
		int target_angle = 90;
		int angle_tolerance = 10;
		bool enable_denoising = false; // 是否启用去噪
		float denoise_h = 5.0f;
		float denoise_hColor = 8.0f;
		int denoise_search_window = 17;
		int denoise_template_window = 11;

		// 切片与检测阈值
		int slice_width = 640;
		int slice_height = 640;
		int slice_distance = 40;
		float nms_threshold = 0.45f;
		float conf_threshold = 0.20f;

		// SAM过滤参数
		double pix_to_mm = 0.021;           // 像素转毫米系数（mm/pixel）
		double min_area_mm2 = 0.0;          // 最小面积阈值（mm^2），默认0表示不限制
		double min_diameter_mm = 0.0;       // 最小圆直径阈值（mm），默认0表示不限制

		// 相机参数
		int delay_ms = 0;                    // 拍摄间隔（毫秒）
		int qw_index = 0;                  // QW相机索引（0-3）
		std::string device_id = "CAM-001";               // 设备/相机标识
	};

	// 服务器共享状态（后续 scheduler 将读取）
	struct ServerState {
		Config* config = nullptr;
		DetectParams last_params; // 最近一次 /api/control/add 的参数
		std::atomic<bool> has_task{ false };
		// 本次程序运行的唯一批次ID（由 main 生成并注册到 inspection_runs）
		std::string run_id;
	};

	class Server {
	public:
		explicit Server();
		~Server();

	public:
		// arg 传入 Config*（XL::Config），用于读取监听地址与端口
		void start(void* arg);

	private:
		std::thread mThread; // 若需要后台运行，可用线程承载 event loop
	};

} // namespace XL

// 接口回调（libevent 使用 C 风格回调，将 ServerState* 作为 arg 传入）
// 全局共享状态（供后续 scheduler 读取），Server 启动时将其填充
extern XL::ServerState g_server_state;
void api_health(struct evhttp_request* req, void* arg);//用于检测服务器是否正常运行，接口能否正常工作
void api_control_add(struct evhttp_request* req, void* arg);//用于添加一个检测任务，需要带好检测参数。
void api_control_cancel(struct evhttp_request* req, void* arg);//用于取消一个检测任务
void parse_get(struct evhttp_request* req, struct evkeyvalq* params);////用于解析get请求
void parse_post(struct evhttp_request* req, char* buff);//用于解析post请求

// 单张图片检测接口
void api_single_detect_add(struct evhttp_request* req, void* arg); // 提交单图检测任务并返回结果
void api_single_detect_cancel(struct evhttp_request* req, void* arg); // 释放单图检测线程资源

// 相机共享实例（供单图采集与组采集复用）
namespace XL { extern std::shared_ptr<CameraThread> g_camera_thread; }

// 单图采集接口（直接采集一张相机图并保存PNG）
void api_camera_single_capture(struct evhttp_request* req, void* arg);
// 相机状态查询接口
void api_camera_status(struct evhttp_request* req, void* arg);
// 相机释放接口
void api_camera_stop(struct evhttp_request* req, void* arg);
