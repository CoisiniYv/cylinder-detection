#pragma once
#include <WinSock2.h>
#include <WS2tcpip.h>
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
#include "Config.hpp"

namespace XL {

    // 检测参数结构（来源：detect_thread.cpp 中的表格）
    struct DetectParams {
        // 模型与输入
        std::string trt_engine_file;           // 模型引擎文件路径（默认从配置读取）
        bool enable_swap_rb = false;            // 是否启用 R/B 交换
        std::string input_image_path;          // 待检测图像路径（最终应来自队列，这里用于测试）

        // ROI 裁剪与条纹线
        int crop_x = 0;
        int crop_y = 0;
        int crop_width = 0;
        int crop_height = 0;
        int line_x1 = 0;
        int line_y1 = 0;
        int line_x2 = 0;
        int line_y2 = 0;
        int stripe_radius = 0;

        // 条纹去除与去噪参数
        int filter_width = 10;
        double attenuation_factor = 0.0001;
        int target_angle = 90;
        int angle_tolerance = 10;
        bool enable_denoising = true;
        float denoise_h = 5.0f;
        float denoise_hColor = 8.0f;
        int denoise_search_window = 17;
        int denoise_template_window = 11;

        // 切片与检测阈值
        bool save_slices = false;
        std::string slices_save_dir = "./slices";
        int slice_width = 640;
        int slice_height = 640;
        int slice_distance = 40;
        float nms_threshold = 0.45f;
        float conf_threshold = 0.20f;

        // 结果输出与标签
        std::string result_image_path = "./output/detection_result.png";
        std::vector<std::string> labels{}; // 可从配置或客户端传入

        // 相机参数
        int delay_ms = 0;                    // 拍摄间隔（毫秒）
        std::string device_id;               // 设备/相机标识（网页端配置并透传）
    };

    // 服务器共享状态（后续 scheduler 将读取）
    struct ServerState {
        Config* config = nullptr;
        DetectParams last_params; // 最近一次 /api/control/add 的参数
        std::atomic<bool> has_task{ false };
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
