#include "group_manager.hpp"
#include "Utils/Common.hpp"

namespace XL {

GroupManager::GroupManager() {}
GroupManager::~GroupManager() {
    stop();
    join();
}

void GroupManager::start() {
    if (mRunning.load(std::memory_order_relaxed)) return;
    mRunning.store(true, std::memory_order_relaxed);
    mThread = std::thread([this]() { this->run(); });
    LOGI("GroupManager Started");
}

void GroupManager::stop() {
    mRunning.store(false, std::memory_order_relaxed);
}

void GroupManager::join() {
    if (mThread.joinable()) mThread.join();
}

// 简单判定：若该面的检测列表非空，则视为NG；否则视为GOOD。
// 后续可按业务规则调整（例如按类别或阈值等）
bool GroupManager::isFaceNG(const SingleImageResult& r) {
    return !r.detections.empty();
}

void GroupManager::handleResult(const SingleImageResult& r) {
    const std::uint64_t gid = r.meta.group_id;
    const int idx = r.meta.index_in_group;
    if (gid == 0 || idx < 0 || idx >= static_cast<int>(kQuadImageCount)) {
        LOGE("GroupManager收到非法结果：group=%llu, idx=%d", gid, idx);
        return;
    }

    std::lock_guard<std::mutex> lk(mMtx);
    auto& st = mGroups[gid];
    if (st.group_id == 0) {
        st.group_id = gid;
        st.start_tp = std::chrono::steady_clock::now();
    }

    if (st.received[idx]) {
        LOGE("GroupManager 重复结果：group=%llu, idx=%d，忽略", gid, idx);
        return;
    }

    st.results[idx] = r;
    st.received[idx] = true;

    bool all_received = true;
    for (bool b : st.received) {
        if (!b) { all_received = false; break; }
    }

    if (all_received) {
        // 收齐四面，立即完成该组
        // 先释放锁再 finalize，避免死锁
        // 注意：finalizeGroup 会自行获取锁并移除该组
        // 为避免使用已释放的迭代器，这里只调用 finalizeGroup
    }
}

bool GroupManager::finalizeGroup(std::uint64_t group_id, bool due_to_timeout) {
    GroupState st;
    {
        std::lock_guard<std::mutex> lk(mMtx);
        auto it = mGroups.find(group_id);
        if (it == mGroups.end()) return false;
        st = it->second; // 拷贝一份在锁外使用
        mGroups.erase(it);
    }

    // 统计组 NG/GOOD：任一面为 NG，则整组 NG
    bool group_ng = false;
    for (int i = 0; i < static_cast<int>(kQuadImageCount); ++i) {
        if (st.received[i]) {
            if (isFaceNG(st.results[i])) {
                group_ng = true;
                break;
            }
        } else {
            // 超时缺失的面可按业务默认规则判定为 NG（此处按 NG）
            if (due_to_timeout) {
                group_ng = true;
                break;
            }
        }
    }

    // 构造组结果（聚合四个 SingleImageResult）
    QuadFrameResult qres;
    for (int i = 0; i < static_cast<int>(kQuadImageCount); ++i) {
        if (st.received[i]) {
            qres.results[i] = st.results[i];
        } else {
            // 缺失的结果：填充基本 meta 以便下游识别
            SingleImageResult miss;
            miss.meta.group_id = group_id;
            miss.meta.index_in_group = i;
            miss.meta.timestamp_ms = getCurTimestamp();
            // 保持 detections 为空（GOOD），或根据 due_to_timeout 逻辑设置占位检测表示 NG
            if (due_to_timeout) {
                Detection d;
                d.label_id = -1;
                d.label = "TIMEOUT";
                d.confidence = 1.0f;
                d.box = {0,0,0,0};
                miss.detections.push_back(d);
            }
            qres.results[i] = std::move(miss);
        }
    }

    LOGI("Group 完成：group=%llu, status=%s%s", group_id,
         group_ng ? "NG" : "GOOD",
         due_to_timeout ? " (TIMEOUT)" : "");

    // 通知相机允许下一组（若设置了 camera 引用）
    if (mCamera) {
        mCamera->allow_next_group();
    }

    // 回调（下游可保存/上报）
    if (mOnComplete) {
        mOnComplete(qres);
    }

    return true;
}

void GroupManager::run() {
    LOGI("GroupManager 线程开始");
    while (mRunning.load(std::memory_order_relaxed)) {
        SingleImageResult r;
        bool ok = g_queue_manager.popResultBlocking(r, 200);
        if (!ok) {
            if (g_queue_manager.stopped()) break; // 已停止
            // 检查超时的组
            std::vector<std::uint64_t> timeout_groups;
            {
                std::lock_guard<std::mutex> lk(mMtx);
                const auto now = std::chrono::steady_clock::now();
                for (auto& kv : mGroups) {
                    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - kv.second.start_tp).count();
                    if (mGroupTimeoutMs > 0 && elapsed_ms >= mGroupTimeoutMs) {
                        timeout_groups.push_back(kv.first);
                    }
                }
            }
            for (auto gid : timeout_groups) {
                finalizeGroup(gid, true);
            }
            continue;
        }

        // 正常收到一张结果
        handleResult(r);

        // 检查是否该组已收齐（避免在锁内调用 finalize），在 handleResult 内已判定
        // 这里直接在锁外再次判定并 finalize
        {
            std::uint64_t gid = r.meta.group_id;
            bool complete = false;
            {
                std::lock_guard<std::mutex> lk(mMtx);
                auto it = mGroups.find(gid);
                if (it != mGroups.end()) {
                    complete = true;
                    for (bool b : it->second.received) {
                        if (!b) { complete = false; break; }
                    }
                }
            }
            if (complete) {
                finalizeGroup(gid, false);
            }
        }
    }
    mRunning.store(false, std::memory_order_relaxed);
    LOGI("GroupManager 线程结束");
}

} // namespace XL