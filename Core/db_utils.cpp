#include "db_utils.hpp"
#include "sqlite_helper.hpp"
#include "Utils/Log.hpp"

#include <filesystem>

namespace XL {

	bool ensure_db_initialized(const std::string& dbfile) {
		try {
			namespace fs = std::filesystem;
			fs::path p(dbfile);
			if (!p.parent_path().empty() && !fs::exists(p.parent_path())) {
				fs::create_directories(p.parent_path());
			}

			SQLiteHelper db(dbfile);
			db.execute("PRAGMA foreign_keys = OFF;");
			db.begin();

			// 运行表
			const char* ddl_runs = R"(
CREATE TABLE IF NOT EXISTS inspection_runs (
    run_id       TEXT PRIMARY KEY,
    created_time DATETIME DEFAULT (datetime('now','localtime'))
);
)";
			db.execute(ddl_runs);

			// 目标结构表
			const char* ddl_groups = R"(
CREATE TABLE IF NOT EXISTS inspection_groups (
    group_pk     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT    NOT NULL,
    group_id     INTEGER NOT NULL,
    device_id    TEXT    NOT NULL,
    status       TEXT    NOT NULL CHECK (status IN ('NG','GOOD')),
    created_time DEFAULT (datetime('now','localtime')),
    UNIQUE(run_id, group_id),
    FOREIGN KEY (run_id) REFERENCES inspection_runs(run_id) ON DELETE CASCADE
);
)";

			const char* ddl_faces = R"(
CREATE TABLE IF NOT EXISTS inspection_faces (
    face_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT    NOT NULL,
    group_id      INTEGER NOT NULL,
    face_index    INTEGER NOT NULL CHECK (face_index BETWEEN 0 AND 3),
    saved_path    TEXT    NOT NULL,
    skeleton_path TEXT    NOT NULL,
    original_path TEXT    NOT NULL,
    created_time  DEFAULT (datetime('now','localtime')),
    UNIQUE(run_id, group_id, face_index),
    FOREIGN KEY (run_id) REFERENCES inspection_runs(run_id) ON DELETE CASCADE
);
)";

			const char* ddl_dets = R"(
CREATE TABLE IF NOT EXISTS defect_detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id      INTEGER NOT NULL,
    label_id     INTEGER NOT NULL CHECK (label_id BETWEEN -1 AND 6),
    confidence   REAL    NOT NULL,
    bbox_x       REAL    NOT NULL,
    bbox_y       REAL    NOT NULL,
    bbox_w       REAL    NOT NULL,
    bbox_h       REAL    NOT NULL,
    length       REAL    NOT NULL,
    area         REAL    NOT NULL,
    created_time DEFAULT (datetime('now','localtime')),
    FOREIGN KEY (face_id) REFERENCES inspection_faces(face_id) ON DELETE CASCADE
);
)";

			// 创建所有表
			db.execute(ddl_groups);
			db.execute(ddl_faces);
			db.execute(ddl_dets);

			// 索引
			// inspection_groups 表
			db.execute("CREATE INDEX IF NOT EXISTS idx_groups_device_id ON inspection_groups(device_id);");
			db.execute("CREATE INDEX IF NOT EXISTS idx_groups_status ON inspection_groups(status);");
			db.execute("CREATE INDEX IF NOT EXISTS idx_groups_created_time ON inspection_groups(created_time);");

			// inspection_faces 表
			db.execute("CREATE INDEX IF NOT EXISTS idx_faces_run_group ON inspection_faces(run_id, group_id);");

			// defect_detections 表
			db.execute("CREATE INDEX IF NOT EXISTS idx_detections_face_id ON defect_detections(face_id);");
			db.execute("CREATE INDEX IF NOT EXISTS idx_detections_label_id ON defect_detections(label_id);");

			db.commit();
			db.execute("PRAGMA foreign_keys = ON;");
			return true;
		}
		catch (const std::exception& e) {
			LOGE("初始化数据库失败: %s", e.what());
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
		}
		catch (const std::exception& e) {
			LOGE("注册 run_id 失败: %s", e.what());
			return false;
		}
	}

} // namespace XL