#pragma once

#include <thread>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <array>
#include <functional>
#include <bitset>
#include <chrono>

#include "image_types.hpp"
#include "queue_manager.hpp"
#include "camera_thread.hpp"


namespace XL {

    // 组管理器：聚合四面检测结果并在完成时通知相机允许下一组
    class GroupManager {
    public:
        GroupManager();
        ~GroupManager();

        void start();
        void stop();
        void join();
        bool isRunning() const noexcept { return mRunning.load(std::memory_order_relaxed); }

        // 设置相机线程引用，用于在组完成时调用 allow_next_group()
        void setCameraThread(CameraThread* cam) { mCamera = cam; }

        // 设置组超时（毫秒）。若一组超过此时间仍未收齐四张结果，则立即按当前结果判定，并用缺失部分的默认规则作为NG/GOOD。
        void setGroupTimeoutMs(int ms) { mGroupTimeoutMs = ms; }

        // 可选：组完成回调（下游可保存、上报或触发其它动作）
        void setOnGroupComplete(const std::function<void(const QuadFrameResult&)>& cb) { mOnComplete = cb; }

    private:
        void run();
        void handleResult(const SingleImageResult& r);
        bool finalizeGroup(std::uint64_t group_id, bool due_to_timeout);
        static bool isFaceNG(const SingleImageResult& r);

    private:
        struct GroupState {
            std::array<bool, kQuadImageCount> received{ false, false, false, false };
            std::array<SingleImageResult, kQuadImageCount> results{};
            std::chrono::steady_clock::time_point start_tp = std::chrono::steady_clock::now();
            std::uint64_t group_id = 0;
        };

        std::unordered_map<std::uint64_t, GroupState> mGroups; // group_id -> state
        std::mutex mMtx;

        std::atomic<bool> mRunning{ false };
        std::thread mThread;

        int mGroupTimeoutMs = 5000; // 默认 5 秒超时
        CameraThread* mCamera = nullptr; // 不拥有
        std::function<void(const QuadFrameResult&)> mOnComplete;
    };

} // namespace XL