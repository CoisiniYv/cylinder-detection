#pragma once
#include <sqlite3.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <initializer_list>

class SQLiteHelper {
public:
    explicit SQLiteHelper(const std::string& file);
    ~SQLiteHelper();

    void execute(const std::string& sql);               // 无结果语句
    void begin();
    void commit();
    void rollback();

    // 参数化 SQL 接口：支持 ? 占位符
    void prepare(const std::string& sql);
    void bind(int idx, int val);
    void bind(int idx, long long val);
    void bind(int idx, double val);
    void bind(int idx, const std::string& val);
    void bind(int idx, const void* blob, int n);
    void step();                    // 执行并自动 finalize
    void reset();                   // 可复用 prepared 语句

    // 查询：返回 vector<vector<std::string>>
    using Row = std::vector<std::string>;
    std::vector<Row> query(const std::string& sql);

private:
    sqlite3* db_ = nullptr;
    sqlite3_stmt* stmt_ = nullptr;
    char* err_ = nullptr;

    void check(int rc, const char* msg);
};