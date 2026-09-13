#include "db_utils.hpp"

#include "Utils/Log.hpp"
#include "sqlite_helper.hpp"

#include <filesystem>

namespace XL {

bool ensure_db_initialized(const std::string& db_file) {
    try {
        const std::filesystem::path path(db_file);
        if (!path.parent_path().empty()) {
            std::filesystem::create_directories(path.parent_path());
        }

        SQLiteHelper db(db_file);
        db.begin();

        db.execute(R"(
CREATE TABLE IF NOT EXISTS inspection_runs (
    run_id       TEXT PRIMARY KEY,
    created_time DATETIME DEFAULT (datetime('now','localtime'))
);
)");

        db.execute(R"(
CREATE TABLE IF NOT EXISTS inspection_groups (
    group_pk     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT    NOT NULL,
    group_id     INTEGER NOT NULL,
    device_id    TEXT    NOT NULL,
    status       TEXT    NOT NULL CHECK (status IN ('NG','GOOD')),
    created_time DATETIME DEFAULT (datetime('now','localtime')),
    UNIQUE(run_id, group_id),
    FOREIGN KEY (run_id) REFERENCES inspection_runs(run_id) ON DELETE CASCADE
);
)");

        db.execute(R"(
CREATE TABLE IF NOT EXISTS inspection_faces (
    face_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT    NOT NULL,
    group_id      INTEGER NOT NULL,
    face_index    INTEGER NOT NULL CHECK (face_index BETWEEN 0 AND 3),
    saved_path    TEXT    NOT NULL,
    skeleton_path TEXT    NOT NULL,
    original_path TEXT    NOT NULL,
    created_time  DATETIME DEFAULT (datetime('now','localtime')),
    UNIQUE(run_id, group_id, face_index),
    FOREIGN KEY (run_id) REFERENCES inspection_runs(run_id) ON DELETE CASCADE
);
)");

        db.execute(R"(
CREATE TABLE IF NOT EXISTS defect_detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id      INTEGER NOT NULL,
    label_id     INTEGER NOT NULL,
    confidence   REAL    NOT NULL,
    bbox_x       REAL    NOT NULL,
    bbox_y       REAL    NOT NULL,
    bbox_w       REAL    NOT NULL,
    bbox_h       REAL    NOT NULL,
    length       REAL    NOT NULL,
    area         REAL    NOT NULL,
    created_time DATETIME DEFAULT (datetime('now','localtime')),
    FOREIGN KEY (face_id) REFERENCES inspection_faces(face_id) ON DELETE CASCADE
);
)");

        db.execute("CREATE INDEX IF NOT EXISTS idx_groups_device_id ON inspection_groups(device_id);");
        db.execute("CREATE INDEX IF NOT EXISTS idx_groups_status ON inspection_groups(status);");
        db.execute("CREATE INDEX IF NOT EXISTS idx_groups_created_time ON inspection_groups(created_time);");
        db.execute("CREATE INDEX IF NOT EXISTS idx_faces_run_group ON inspection_faces(run_id, group_id);");
        db.execute("CREATE INDEX IF NOT EXISTS idx_detections_face_id ON defect_detections(face_id);");
        db.execute("CREATE INDEX IF NOT EXISTS idx_detections_label_id ON defect_detections(label_id);");

        db.commit();
        return true;
    }
    catch (const std::exception& e) {
        LOGE("database initialization failed: %s", e.what());
        return false;
    }
}

bool register_run(const std::string& db_file, const std::string& run_id) {
    if (run_id.empty()) return false;

    try {
        SQLiteHelper db(db_file);
        db.prepare("INSERT OR IGNORE INTO inspection_runs(run_id) VALUES(?);");
        db.bind(1, run_id);
        db.step();
        return true;
    }
    catch (const std::exception& e) {
        LOGE("run registration failed: %s", e.what());
        return false;
    }
}

} // namespace XL
