#include "db_utils.hpp"
#include "sqlite_helper.hpp"
#include "Utils/Log.hpp"
#include <filesystem>

namespace XL {

static bool has_table(SQLiteHelper& db, const std::string& name) {
    std::string q = "SELECT name FROM sqlite_master WHERE type='table' AND name='" + name + "';";
    auto rows = db.query(q);
    return !rows.empty();
}

static bool table_has_column(SQLiteHelper& db, const std::string& table, const std::string& col) {
    std::string q = "PRAGMA table_info(" + table + ");";
    auto rows = db.query(q);
    for (const auto& r : rows) {
        if (r.size() >= 2 && r[1] == col) return true; // r[1] is column name
    }
    return false;
}

bool ensure_db_initialized(const std::string& dbfile) {
    try {
        namespace fs = std::filesystem;
        fs::path p(dbfile);
        if (!p.parent_path().empty() && !fs::exists(p.parent_path())) {
            fs::create_directories(p.parent_path());
        }

        SQLiteHelper db(dbfile); // sqlite3_open 会在文件不存在时创建新数据库文件
        db.execute("PRAGMA foreign_keys = OFF;"); // 迁移阶段先关闭外键
        db.begin();

        // 运行表（run 批次记录）
        const char* ddl_runs = R"(
CREATE TABLE IF NOT EXISTS inspection_runs (
    run_id       TEXT PRIMARY KEY,
    created_time DATETIME DEFAULT CURRENT_TIMESTAMP
);
)";
        db.execute(ddl_runs);

        // 目标结构（支持 run_id）
        const char* ddl_groups_new = R"(
CREATE TABLE IF NOT EXISTS inspection_groups_new (
    group_pk     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT    NOT NULL,
    group_id     INTEGER NOT NULL,
    device_id    TEXT    NOT NULL,
    status       TEXT    NOT NULL CHECK (status IN ('NG','GOOD')),
    created_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(run_id, group_id),
    FOREIGN KEY (run_id) REFERENCES inspection_runs(run_id) ON DELETE CASCADE
);
)";

        const char* ddl_faces_new = R"(
CREATE TABLE IF NOT EXISTS inspection_faces_new (
    face_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT    NOT NULL,
    group_id      INTEGER NOT NULL,
    face_index    INTEGER NOT NULL CHECK (face_index BETWEEN 0 AND 3),
    timestamp_ms  INTEGER NOT NULL,
    sequence_id   INTEGER NOT NULL,
    saved_path    TEXT    NOT NULL,
    skeleton_path TEXT    NOT NULL,
    created_time  DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(run_id, group_id, face_index),
    FOREIGN KEY (run_id) REFERENCES inspection_runs(run_id) ON DELETE CASCADE
);
)";

        const char* ddl_dets = R"(
CREATE TABLE IF NOT EXISTS defect_detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id      INTEGER NOT NULL,
    label_id     INTEGER NOT NULL,
    label        TEXT    NOT NULL,
    confidence   REAL    NOT NULL,
    bbox_x       REAL    NOT NULL,
    bbox_y       REAL    NOT NULL,
    bbox_w       REAL    NOT NULL,
    bbox_h       REAL    NOT NULL,
    length       REAL    NOT NULL,
    area         REAL    NOT NULL,
    created_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (face_id) REFERENCES inspection_faces(face_id) ON DELETE CASCADE
);
)";

        bool groups_exists = has_table(db, "inspection_groups");
        bool faces_exists  = has_table(db, "inspection_faces");
        bool dets_exists   = has_table(db, "defect_detections");

        bool groups_need_migration = groups_exists && !table_has_column(db, "inspection_groups", "run_id");
        bool faces_need_migration  = faces_exists  && !table_has_column(db, "inspection_faces",  "run_id");

        if (groups_need_migration || !groups_exists) {
            db.execute(ddl_groups_new);
        }
        if (faces_need_migration || !faces_exists) {
            db.execute(ddl_faces_new);
        }
        if (!dets_exists) {
            db.execute(ddl_dets);
        }

        // 若存在旧结构则迁移到新结构，并设置 legacy run
        if (groups_need_migration || faces_need_migration) {
            db.execute("INSERT OR IGNORE INTO inspection_runs(run_id) VALUES('legacy');");
        }

        if (groups_need_migration) {
            db.execute("INSERT INTO inspection_groups_new(run_id, group_id, device_id, status, created_time, updated_time) SELECT 'legacy', group_id, device_id, status, created_time, updated_time FROM inspection_groups;");
        }
        if (faces_need_migration) {
            db.execute("INSERT INTO inspection_faces_new(face_id, run_id, group_id, face_index, timestamp_ms, sequence_id, saved_path, skeleton_path, created_time) SELECT face_id, 'legacy', group_id, face_index, timestamp_ms, sequence_id, saved_path, skeleton_path, created_time FROM inspection_faces;");
        }

        // 删除旧表并重命名
        if (faces_need_migration) {
            db.execute("DROP TABLE inspection_faces;");
            db.execute("ALTER TABLE inspection_faces_new RENAME TO inspection_faces;");
            db.execute("CREATE INDEX IF NOT EXISTS idx_faces_run_group ON inspection_faces(run_id, group_id);");
            db.execute("CREATE INDEX IF NOT EXISTS idx_faces_timestamp_ms ON inspection_faces(timestamp_ms);");
            db.execute("CREATE INDEX IF NOT EXISTS idx_faces_sequence_id  ON inspection_faces(sequence_id);");
        } else if (!faces_exists) {
            // 新建表场景：直接重命名并建索引
            db.execute("ALTER TABLE inspection_faces_new RENAME TO inspection_faces;");
            db.execute("CREATE INDEX IF NOT EXISTS idx_faces_run_group ON inspection_faces(run_id, group_id);");
            db.execute("CREATE INDEX IF NOT EXISTS idx_faces_timestamp_ms ON inspection_faces(timestamp_ms);");
            db.execute("CREATE INDEX IF NOT EXISTS idx_faces_sequence_id  ON inspection_faces(sequence_id);");
        }

        if (groups_need_migration) {
            db.execute("DROP TABLE inspection_groups;");
            db.execute("ALTER TABLE inspection_groups_new RENAME TO inspection_groups;");
            db.execute("CREATE INDEX IF NOT EXISTS idx_groups_device_id      ON inspection_groups(device_id);");
            db.execute("CREATE INDEX IF NOT EXISTS idx_groups_status         ON inspection_groups(status);");
            db.execute("CREATE INDEX IF NOT EXISTS idx_groups_created_time   ON inspection_groups(created_time);");
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_groups_run_group ON inspection_groups(run_id, group_id);");
        } else if (!groups_exists) {
            db.execute("ALTER TABLE inspection_groups_new RENAME TO inspection_groups;");
            db.execute("CREATE INDEX IF NOT EXISTS idx_groups_device_id      ON inspection_groups(device_id);");
            db.execute("CREATE INDEX IF NOT EXISTS idx_groups_status         ON inspection_groups(status);");
            db.execute("CREATE INDEX IF NOT EXISTS idx_groups_created_time   ON inspection_groups(created_time);");
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_groups_run_group ON inspection_groups(run_id, group_id);");
        }

        db.commit();
        db.execute("PRAGMA foreign_keys = ON;");
        return true;
    } catch (const std::exception& e) {
        LOGE("初始化/迁移数据库失败: %s", e.what());
        return false;
    }
}

bool register_run(const std::string& dbfile, const std::string& run_id) {
    try {
        SQLiteHelper db(dbfile);
        db.prepare("INSERT OR IGNORE INTO inspection_runs(run_id) VALUES(?);");
        db.bind(1, run_id);
        db.step();
        return true;
    } catch (const std::exception& e) {
        LOGE("注册 run_id 失败: %s", e.what());
        return false;
    }
}

} // namespace XL