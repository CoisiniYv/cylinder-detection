#pragma once

#include <sqlite3.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Small RAII wrapper around one SQLite connection and one active prepared
// statement. The wrapper intentionally exposes only the operations required by
// the inspection repository so SQL ownership remains centralized.
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

    // Execute a statement that is expected to produce no row result.
    void step();

    // Execute a statement with RETURNING and read the first column as int64.
    long long stepInt64();

    using Row = std::vector<std::string>;
    std::vector<Row> query(const std::string& sql);

private:
    void check(int rc, const char* operation) const;
    void finalizeStatement() noexcept;
    void closeConnection() noexcept;

private:
    sqlite3* db_ = nullptr;
    sqlite3_stmt* stmt_ = nullptr;
};
