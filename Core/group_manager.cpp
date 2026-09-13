#include "group_manager.hpp"

#include "Utils/Log.hpp"
#include "camera_thread.hpp"
#include "queue_manager.hpp"

#include <exception>
#include <vector>

namespace XL {

GroupManager::GroupManager() = default;

GroupManager::~GroupManager() {
    stop();
    join();
}

void GroupManager::start() {
    if (mRunning.exchange(true, std::memory_order_relaxed)) return;
    mThread = std::thread(&GroupManager::run, this);
    LOGI("GroupManager started");
}

void GroupManager::stop() {
    mRunning.store(false, std::memory_order_relaxed);
}

void GroupManager::join() {
    if (mThread.joinable()) mThread.join();
}

bool GroupManager::isFaceNG(const SingleImageResult& result) {
    return !result.processing_ok || !result.detections.empty();
}

bool GroupManager::handleResult(const SingleImageResult& result) {
    const std::uint64_t group_id = result.meta.group_id;
    const int face_index = result.meta.index_in_group;
    if (group_id == 0 || face_index < 0 || face_index >= static_cast<int>(kQuadImageCount)) {
        LOGE("invalid worker result: group=%llu face=%d",
             static_cast<unsigned long long>(group_id), face_index);
        return false;
    }

    std::lock_guard<std::mutex> lock(mMutex);
    auto& group = mGroups[group_id];
    if (group.group_id == 0) {
        group.group_id = group_id;
        group.started_at = std::chrono::steady_clock::now();
    }

    if (group.received[face_index]) {
        LOGE("duplicate worker result ignored: group=%llu face=%d",
             static_cast<unsigned long long>(group_id), face_index);
        return false;
    }

    group.results[face_index] = result;
    group.received[face_index] = true;

    for (const bool received : group.received) {
        if (!received) return false;
    }
    return true;
}

bool GroupManager::finalizeGroup(std::uint64_t group_id, bool timed_out) {
    GroupState group;
    {
        std::lock_guard<std::mutex> lock(mMutex);
        const auto it = mGroups.find(group_id);
        if (it == mGroups.end()) return false;
        group = std::move(it->second);
        mGroups.erase(it);
    }

    QuadFrameResult result;
    std::string device_id;
    for (std::size_t i = 0; i < kQuadImageCount; ++i) {
        if (group.received[i]) {
            result.results[i] = std::move(group.results[i]);
            if (device_id.empty()) device_id = result.results[i].meta.device_id;
            continue;
        }

        SingleImageResult missing;
        missing.meta.group_id = group_id;
        missing.meta.index_in_group = static_cast<int>(i);
        missing.meta.device_id = device_id;
        missing.processing_ok = false;
        missing.error_message = timed_out ? "group result timed out" : "group result missing";
        result.results[i] = std::move(missing);
    }

    bool group_ng = false;
    for (const auto& face : result.results) {
        if (isFaceNG(face)) {
            group_ng = true;
            break;
        }
    }

    LOGI("group complete: group=%llu status=%s%s",
         static_cast<unsigned long long>(group_id),
         group_ng ? "NG" : "GOOD",
         timed_out ? " timeout" : "");

    // Persistence/reporting is an extension point. A downstream exception must
    // not terminate the aggregation thread or permanently stall camera flow.
    if (mOnComplete) {
        try {
            mOnComplete(result);
        }
        catch (const std::exception& e) {
            LOGE("group completion callback failed: %s", e.what());
        }
        catch (...) {
            LOGE("group completion callback failed with unknown exception");
        }
    }

    if (mCamera) mCamera->allow_next_group();
    return true;
}

void GroupManager::checkTimeouts() {
    if (mGroupTimeoutMs <= 0) return;

    std::vector<std::uint64_t> timed_out_groups;
    {
        std::lock_guard<std::mutex> lock(mMutex);
        const auto now = std::chrono::steady_clock::now();
        for (const auto& [group_id, group] : mGroups) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - group.started_at);
            if (elapsed.count() >= mGroupTimeoutMs) timed_out_groups.push_back(group_id);
        }
    }

    for (const auto group_id : timed_out_groups) {
        finalizeGroup(group_id, true);
    }
}

void GroupManager::run() {
    LOGI("GroupManager loop started");
    while (mRunning.load(std::memory_order_relaxed)) {
        SingleImageResult result;
        if (!g_queue_manager.popResultBlocking(result, 200)) {
            if (g_queue_manager.stopped()) break;
            checkTimeouts();
            continue;
        }

        const std::uint64_t group_id = result.meta.group_id;
        if (handleResult(result)) finalizeGroup(group_id, false);
        checkTimeouts();
    }

    mRunning.store(false, std::memory_order_relaxed);
    LOGI("GroupManager loop stopped");
}

} // namespace XL
