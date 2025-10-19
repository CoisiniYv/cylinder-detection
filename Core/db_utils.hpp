#pragma once
#include <string>

namespace XL {
    // 初始化本地 SQLite 数据库：
    // - 若文件不存在则创建
    // - 打开数据库，启用外键约束
    // - 幂等创建/迁移业务表到支持 run_id 的结构
    // 返回 true 表示初始化成功（或已是已初始化状态），false 表示失败
    bool ensure_db_initialized(const std::string& dbfile);

    // 确保指定 run_id 在 inspection_runs 表中已注册（INSERT OR IGNORE）。
    // 返回 true 表示成功或已存在，false 表示失败。
    bool register_run(const std::string& dbfile, const std::string& run_id);
}