#include <memory>
#include <vector>
#include <atomic>
#include "sqlite_helper.hpp"
#include "Server.hpp"           // DetectParams / g_server_state (optional)
#include "queue_manager.hpp"    // g_queue_manager
#include "camera_thread.hpp"
#include "detect_thread.hpp"
#include "group_manager.hpp"


namespace XL {

    // 调度器：负责整体多线程编排
    // - 当 Server 接收任务参数后，调用 Scheduler::start(params)
    // - 创建并启动：队列管理器、相机采集线程、4 个检测线程、组管理线程
    // - 在 stop() 时，优雅地停止所有线程并清理资源
    class Scheduler {
    public:
        Scheduler();
        ~Scheduler();

        // 启动整体流程；默认创建 4 个检测线程（对应四面）
        bool start(const DetectParams& params, int num_detect_threads = 4);
        // 从全局 server 状态启动（若有任务），便于与 Server 集成
        bool startFromServerState(int num_detect_threads = 4);

        void stop();
        void join();
        bool isRunning() const noexcept { return mRunning.load(std::memory_order_relaxed); }

        // 可选：设置组超时（毫秒），转发给 GroupManager
        void setGroupTimeoutMs(int ms) { mGroupTimeoutMs = ms; }

    private:
        void cleanup();

    private:
        std::atomic<bool> mRunning{ false };
        DetectParams mParams{};

        std::unique_ptr<CameraThread> mCamera;
        std::vector<std::unique_ptr<DetectThread>> mWorkers;
        std::unique_ptr<GroupManager> mGroupMgr;

        int mGroupTimeoutMs = 5000; // 默认 5 秒
    };

} // namespace XL