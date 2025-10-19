#include <sqlite3.h>
#include <iostream>
#include <string>

static const char* ddl_inspection_groups = R"(
CREATE TABLE IF NOT EXISTS inspection_groups (
    group_id     INTEGER PRIMARY KEY,          -- 对应 MySQL BIGINT UNSIGNED PK
    device_id    TEXT    NOT NULL,
    status       TEXT    NOT NULL
                 CHECK (status IN ('NG','GOOD')),
    created_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_time DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_device_id     ON inspection_groups(device_id);
CREATE INDEX IF NOT EXISTS idx_status        ON inspection_groups(status);
CREATE INDEX IF NOT EXISTS idx_g_created_time ON inspection_groups(created_time);
)";

static const char* ddl_inspection_faces = R"(
CREATE TABLE IF NOT EXISTS inspection_faces (
    face_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id      INTEGER NOT NULL,
    face_index    INTEGER NOT NULL CHECK (face_index BETWEEN 0 AND 3),
    timestamp_ms  INTEGER NOT NULL,
    sequence_id   INTEGER NOT NULL,
    saved_path    TEXT    NOT NULL,
    skeleton_path TEXT    NOT NULL,
    created_time  DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(group_id, face_index),
    FOREIGN KEY (group_id) REFERENCES inspection_groups(group_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_group_id     ON inspection_faces(group_id);
CREATE INDEX IF NOT EXISTS idx_timestamp_ms ON inspection_faces(timestamp_ms);
CREATE INDEX IF NOT EXISTS idx_sequence_id  ON inspection_faces(sequence_id);
)";

static const char* ddl_defect_detections = R"(
CREATE TABLE IF NOT EXISTS defect_detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id      INTEGER NOT NULL,
    label_id     INTEGER NOT NULL,
    label        TEXT    NOT NULL,
    confidence   REAL    NOT NULL,          -- 0.0-1.0
    bbox_x       REAL    NOT NULL,
    bbox_y       REAL    NOT NULL,
    bbox_w       REAL    NOT NULL,
    bbox_h       REAL    NOT NULL,
    length       REAL    NOT NULL,
    area         REAL    NOT NULL,
    created_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (face_id) REFERENCES inspection_faces(face_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_face_id     ON defect_detections(face_id);
CREATE INDEX IF NOT EXISTS idx_label_id    ON defect_detections(label_id);
CREATE INDEX IF NOT EXISTS idx_confidence  ON defect_detections(confidence);
)";

bool execute_sql(sqlite3* db, const std::string& sql) {
    char* err = nullptr;
    if (sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK) {
        std::cerr << "SQLite error: " << (err ? err : "unknown") << '\n';
        sqlite3_free(err);
        return false;
    }
    return true;
}

int main() {
    sqlite3* db = nullptr;
    if (sqlite3_open("my_inspection.db", &db) != SQLITE_OK) {
        std::cerr << "Can't open database\n";
        return 1;
    }

    bool ok = true;
    ok &= execute_sql(db, ddl_inspection_groups);
    ok &= execute_sql(db, ddl_inspection_faces);
    ok &= execute_sql(db, ddl_defect_detections);

    if (ok) std::cout << "All tables created successfully!\n";

    // 简单验证外键生效
    execute_sql(db, "PRAGMA foreign_keys = ON;");
    execute_sql(db,
        "INSERT INTO inspection_groups(group_id,device_id,status) "
        "VALUES (1,'D001','GOOD');");
    execute_sql(db,
        "INSERT INTO inspection_faces(group_id,face_index,timestamp_ms,sequence_id,saved_path,skeleton_path) "
        "VALUES (1,0,1680000000000,1001,'/a/0.jpg','/a/0_sk.jpg');");
    execute_sql(db,
        "INSERT INTO defect_detections(face_id,label_id,label,confidence,bbox_x,bbox_y,bbox_w,bbox_h,length,area) "
        "VALUES (1,2,'scratch',0.9234,10.5,20.0,30.0,40.0,50.0,600.0);");

    std::cout << "Foreign key & cascade test passed.\n";
    sqlite3_close(db);
    std::cout << "数据库已写入 my_inspection.db\n";
    return ok ? 0 : 2;
}



inspection_groups（任务级）
group_id     INTEGER PK           -- 任务号（JSON.group_id）
device_id    TEXT  NOT NULL       -- 设备标识（JSON.device_id）
status       TEXT  CHECK(..)      -- 整任务结论 NG|GOOD
created_time DATETIME DEFAULT CURRENT_TIMESTAMP
updated_time DATETIME DEFAULT CURRENT_TIMESTAMP
索引：device_id | status | created_time
C++ 映射（示例）
struct Group{
    int64_t  group_id;
    std::string device_id;
    std::string status;   // "NG" or "GOOD"
    int64_t created_time; // unix epoch seconds (-1 用默认)
};
**************


**************
② inspection_faces（面级）
face_id       INTEGER PK AUTOINCREMENT  -- 面记录序号
group_id      INTEGER FK → groups(group_id) ON DELETE CASCADE
face_index    INTEGER  CHECK(0-3)        -- 数组下标
timestamp_ms  INTEGER                   -- 曝光时间戳（JSON.meta.timestamp_ms）
sequence_id   INTEGER                   -- 图像序列号（JSON.meta.sequence_id）
saved_path    TEXT                      -- 原图绝对路径
skeleton_path TEXT                      -- 骨架图绝对路径
created_time  DATETIME DEFAULT CURRENT_TIMESTAMP
复合唯一：(group_id, face_index)
索引：group_id | timestamp_ms | sequence_id
C++ 映射
struct Face{
    int64_t face_id;        // 插入后由 SQLite 返回
    int64_t group_id;
    int     face_index;     // 0..3
    int64_t timestamp_ms;
    int64_t sequence_id;
    std::string saved_path;
    std::string skeleton_path;
};
**************


③ defect_detections（缺陷框级）
detection_id INTEGER PK AUTOINCREMENT
face_id      INTEGER FK → faces(face_id) ON DELETE CASCADE
label_id     INTEGER                   -- 缺陷类别编号
label        TEXT                      -- 缺陷英文名/中文名
confidence   REAL       [0.0-1.0]      -- 置信度
bbox_x       REAL                        -- 左上角 X (px)
bbox_y       REAL                        -- 左上角 Y (px)
bbox_w       REAL                        -- 宽度 (px)
bbox_h       REAL                        -- 高度 (px)
length       REAL                        -- 缺陷长度 (px)
area         REAL                        -- 缺陷面积 (px²)
created_time DATETIME DEFAULT CURRENT_TIMESTAMP
索引：face_id | label_id | confidence
C++ 映射
struct Detection{
    int64_t detection_id;   // 插入后返回
    int64_t face_id;
    int     label_id;
    std::string label;
    double confidence;
    double bbox_x, bbox_y, bbox_w, bbox_h;
    double length, area;
};