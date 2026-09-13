#include "inspection_repository.hpp"

#include "Utils/Log.hpp"
#include "sqlite_helper.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace XL {
namespace {

void upsertGroup(SQLiteHelper& db,
                 const std::string& run_id,
                 long long group_id,
                 const std::string& device_id,
                 const std::string& status) {
    db.prepare(
        "INSERT INTO inspection_groups(run_id,group_id,device_id,status) VALUES(?,?,?,?) "
        "ON CONFLICT(run_id,group_id) DO UPDATE SET "
        "device_id=excluded.device_id,status=excluded.status;");
    db.bind(1, run_id);
    db.bind(2, group_id);
    db.bind(3, device_id);
    db.bind(4, status);
    db.step();
}

long long upsertFace(SQLiteHelper& db,
                     const std::string& run_id,
                     long long group_id,
                     int face_index,
                     const SingleImageResult& result) {
    db.prepare(
        "INSERT INTO inspection_faces(run_id,group_id,face_index,saved_path,skeleton_path,original_path) "
        "VALUES(?,?,?,?,?,?) "
        "ON CONFLICT(run_id,group_id,face_index) DO UPDATE SET "
        "saved_path=excluded.saved_path,skeleton_path=excluded.skeleton_path,original_path=excluded.original_path;");
    db.bind(1, run_id);
    db.bind(2, group_id);
    db.bind(3, face_index);
    db.bind(4, result.saved_path);
    db.bind(5, result.skeleton_path);
    db.bind(6, result.original_path);
    db.step();

    db.prepare(
        "SELECT face_id FROM inspection_faces "
        "WHERE run_id=? AND group_id=? AND face_index=?;");
    db.bind(1, run_id);
    db.bind(2, group_id);
    db.bind(3, face_index);
    return db.stepInt64();
}

void replaceDefects(SQLiteHelper& db,
                    long long face_id,
                    const std::vector<Detection>& detections) {
    db.prepare("DELETE FROM defect_detections WHERE face_id=?;");
    db.bind(1, face_id);
    db.step();

    for (const auto& detection : detections) {
        db.prepare(
            "INSERT INTO defect_detections(face_id,label_id,confidence,bbox_x,bbox_y,bbox_w,bbox_h,length,area) "
            "VALUES(?,?,?,?,?,?,?,?,?);");
        db.bind(1, face_id);
        db.bind(2, detection.label_id);
        db.bind(3, static_cast<double>(detection.confidence));
        db.bind(4, static_cast<double>(detection.box.x));
        db.bind(5, static_cast<double>(detection.box.y));
        db.bind(6, static_cast<double>(detection.box.w));
        db.bind(7, static_cast<double>(detection.box.h));

        const double length = detection.length > 0.0
            ? detection.length
            : std::max(static_cast<double>(detection.box.w), static_cast<double>(detection.box.h));
        const double area = detection.area > 0.0
            ? detection.area
            : static_cast<double>(detection.box.w) * static_cast<double>(detection.box.h);
        db.bind(8, length);
        db.bind(9, area);
        db.step();
    }
}

} // namespace

InspectionRepository::InspectionRepository(
    std::string database_file,
    std::string output_root,
    std::string run_id,
    std::string device_id)
    : mDatabaseFile(std::move(database_file)),
      mOutputRoot(std::move(output_root)),
      mRunId(std::move(run_id)),
      mDeviceId(std::move(device_id)) {}

bool InspectionRepository::save(const QuadFrameResult& group_result) const {
    try {
        const auto group_it = std::find_if(
            group_result.results.begin(),
            group_result.results.end(),
            [](const SingleImageResult& result) { return result.meta.group_id != 0; });
        if (group_it == group_result.results.end()) {
            LOGE("cannot persist group without group_id");
            return false;
        }

        const std::uint64_t group_id_u64 = group_it->meta.group_id;
        const long long group_id = static_cast<long long>(group_id_u64);
        const bool is_ng = std::any_of(
            group_result.results.begin(),
            group_result.results.end(),
            [](const SingleImageResult& result) {
                return !result.processing_ok || !result.detections.empty();
            });
        const std::string status = is_ng ? "NG" : "GOOD";

        SQLiteHelper db(mDatabaseFile);
        db.begin();
        upsertGroup(db, mRunId, group_id, mDeviceId, status);

        const std::filesystem::path group_dir =
            std::filesystem::path(mOutputRoot) / mRunId / std::to_string(group_id);

        for (std::size_t i = 0; i < kQuadImageCount; ++i) {
            SingleImageResult result = group_result.results[i];
            if (result.original_path.empty()) {
                result.original_path =
                    (group_dir /
                     ("original_g" + std::to_string(group_id) + "_i" + std::to_string(i) + ".png"))
                        .string();
            }

            const long long face_id =
                upsertFace(db, mRunId, group_id, static_cast<int>(i), result);
            if (face_id <= 0) {
                db.rollback();
                LOGE("failed to persist face: run=%s group=%llu face=%zu",
                     mRunId.c_str(),
                     static_cast<unsigned long long>(group_id_u64),
                     i);
                return false;
            }
            replaceDefects(db, face_id, result.detections);
        }

        db.commit();
        LOGI("group persisted: run=%s group=%llu device=%s status=%s",
             mRunId.c_str(),
             static_cast<unsigned long long>(group_id_u64),
             mDeviceId.c_str(),
             status.c_str());
        return true;
    }
    catch (const std::exception& e) {
        LOGE("persist group failed: %s", e.what());
        return false;
    }
}

} // namespace XL
