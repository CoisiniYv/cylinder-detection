#pragma once

#include <concurrentqueue.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <chrono>
#include <cstddef>

#include "image_types.hpp"

namespace XL {

    // 全局数据管理队列：封装 moodycamel::ConcurrentQueue
    // 原始帧队列（camera_thread -> worker_thread），类型为 ImageFrame（每拍一张立即入队）
    // 检测结果队列（worker_thread -> group_manager），类型为 SingleImageResult（单张图检测结果）
    class QueueManager {
    public:
        using RawType = ImageFrame;
        using ResultType = SingleImageResult;

        QueueManager();
        ~QueueManager();

        // 控制：启动/停止（停止后阻塞等待立即返回）
        void start();
        void stop();
        bool stopped() const noexcept { return mStopped.load(std::memory_order_relaxed); }

        // 清空两个队列（用于重置或退出前清理）
        void clearRaw();
        void clearResult();

        // 原始帧队列：入队（相机采集线程调用）
        bool pushRaw(const RawType& item);
        bool pushRaw(RawType&& item);
        // 原始帧队列：非阻塞出队（工作线程调用）
        bool tryPopRaw(RawType& out);
        // 原始帧队列：阻塞出队（可设置超时，timeout_ms < 0 表示一直阻塞直到有数据或 stop）
        bool popRawBlocking(RawType& out, int timeout_ms = -1);

        // 检测结果队列：入队（工作线程调用）
        bool pushResult(const ResultType& item);
        bool pushResult(ResultType&& item);
        // 检测结果队列：非阻塞出队（组管理线程调用）
        bool tryPopResult(ResultType& out);
        // 检测结果队列：阻塞出队（可设置超时，timeout_ms < 0 表示一直阻塞直到有数据或 stop）
        bool popResultBlocking(ResultType& out, int timeout_ms = -1);

        // 近似队列长度（moodycamel 提供的 size_approx，非严格准确）
        std::size_t rawSizeApprox() const;
        std::size_t resultSizeApprox() const;

    private:
        // 队列
        moodycamel::ConcurrentQueue<RawType> mRawQueue;
        moodycamel::ConcurrentQueue<ResultType> mResultQueue;

        // 停止标志与条件变量实现阻塞等待唤醒
        std::atomic<bool> mStopped{ false };

        // 原始队列条件变量
        std::condition_variable mRawCv;
        mutable std::mutex mRawMtx;

        // 结果队列条件变量
        std::condition_variable mResultCv;
        mutable std::mutex mResultMtx;
    };

    // 全局实例（便于在各线程模块直接引用）
    extern QueueManager g_queue_manager;

} // namespace XL