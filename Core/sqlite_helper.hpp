#pragma once

#include <sqlite3.h>

#include <stdexcept>
#include <string>
#include <vector>

// Small RAII wrapper around one SQLite connection and one prepared statement.
// Every connection enables foreign keys and a busy timeout in the constructor.
class SQLiteHelper {
public:
    explicit SQLiteHelper(const std::string& file);
    ~SQLiteHelper();

    SQLiteHelper(const SQLiteHelper&) = delete;
    SQLiteHelper& operator=(const SQLiteHelper&) = delete;

    void execute(const std::string& sql);
    void begin();
    void commit();
    void rollback();

    void prepare(const std::string& sql);
    void bind(int index, int value);
    void bind(int index, long long value);
    void bind(int index, double value);
    void bind(int index, const std::string& value);
    void bind(int index, const void* blob, int size);
    void step();
    void reset();

    using Row = std::vector<std::string>;
    std::vector<Row> query(const std::string& sql);

private:
    void check(int rc, const char* operation);

private:
    sqlite3* db_ = nullptr;
    sqlite3_stmt* stmt_ = nullptr;
    char* err_ = nullptr;
};
