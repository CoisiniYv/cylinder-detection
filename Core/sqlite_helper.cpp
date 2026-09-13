#include "sqlite_helper.hpp"

#include <utility>

SQLiteHelper::SQLiteHelper(const std::string& file) {
    const int rc = sqlite3_open(file.c_str(), &db_);
    if (rc != SQLITE_OK) {
        const std::string message = db_ ? sqlite3_errmsg(db_) : "unknown sqlite open error";
        if (db_) {
            sqlite3_close(db_);
            db_ = nullptr;
        }
        throw std::runtime_error("open: " + message);
    }

    execute("PRAGMA foreign_keys = ON;");
    sqlite3_busy_timeout(db_, 5000);
}

SQLiteHelper::~SQLiteHelper() {
    if (stmt_) sqlite3_finalize(stmt_);
    if (err_) sqlite3_free(err_);
    if (db_) sqlite3_close(db_);
}

void SQLiteHelper::check(int rc, const char* operation) {
    if (rc == SQLITE_OK || rc == SQLITE_DONE || rc == SQLITE_ROW) return;
    const char* detail = err_ ? err_ : (db_ ? sqlite3_errmsg(db_) : "unknown sqlite error");
    throw std::runtime_error(std::string(operation) + ": " + detail);
}

void SQLiteHelper::execute(const std::string& sql) {
    if (err_) {
        sqlite3_free(err_);
        err_ = nullptr;
    }

    const int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err_);
    if (rc == SQLITE_OK) return;

    const std::string detail = err_ ? err_ : sqlite3_errmsg(db_);
    if (err_) {
        sqlite3_free(err_);
        err_ = nullptr;
    }
    throw std::runtime_error("exec: " + detail);
}

void SQLiteHelper::begin() { execute("BEGIN TRANSACTION;"); }
void SQLiteHelper::commit() { execute("COMMIT;"); }
void SQLiteHelper::rollback() { execute("ROLLBACK;"); }

void SQLiteHelper::prepare(const std::string& sql) {
    if (stmt_) {
        sqlite3_finalize(stmt_);
        stmt_ = nullptr;
    }
    check(sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt_, nullptr), "prepare");
}

void SQLiteHelper::bind(int index, int value) {
    check(sqlite3_bind_int(stmt_, index, value), "bind int");
}

void SQLiteHelper::bind(int index, long long value) {
    check(sqlite3_bind_int64(stmt_, index, static_cast<sqlite3_int64>(value)), "bind int64");
}

void SQLiteHelper::bind(int index, double value) {
    check(sqlite3_bind_double(stmt_, index, value), "bind double");
}

void SQLiteHelper::bind(int index, const std::string& value) {
    check(
        sqlite3_bind_text(
            stmt_, index, value.c_str(), static_cast<int>(value.size()), SQLITE_TRANSIENT),
        "bind text");
}

void SQLiteHelper::bind(int index, const void* blob, int size) {
    check(sqlite3_bind_blob(stmt_, index, blob, size, SQLITE_TRANSIENT), "bind blob");
}

void SQLiteHelper::step() {
    if (!stmt_) throw std::runtime_error("step: no prepared statement");

    const int rc = sqlite3_step(stmt_);
    if (rc != SQLITE_DONE && rc != SQLITE_ROW) {
        const std::string detail = sqlite3_errmsg(db_);
        sqlite3_finalize(stmt_);
        stmt_ = nullptr;
        throw std::runtime_error("step: " + detail);
    }

    sqlite3_finalize(stmt_);
    stmt_ = nullptr;
}

void SQLiteHelper::reset() {
    if (!stmt_) throw std::runtime_error("reset: no prepared statement");
    check(sqlite3_reset(stmt_), "reset");
}

std::vector<SQLiteHelper::Row> SQLiteHelper::query(const std::string& sql) {
    std::vector<Row> result;
    sqlite3_stmt* statement = nullptr;
    check(sqlite3_prepare_v2(db_, sql.c_str(), -1, &statement, nullptr), "query prepare");

    int rc = SQLITE_OK;
    while ((rc = sqlite3_step(statement)) == SQLITE_ROW) {
        Row row;
        const int column_count = sqlite3_column_count(statement);
        row.reserve(static_cast<std::size_t>(column_count));
        for (int column = 0; column < column_count; ++column) {
            const unsigned char* text = sqlite3_column_text(statement, column);
            row.emplace_back(text ? reinterpret_cast<const char*>(text) : "");
        }
        result.emplace_back(std::move(row));
    }

    if (rc != SQLITE_DONE) {
        const std::string detail = sqlite3_errmsg(db_);
        sqlite3_finalize(statement);
        throw std::runtime_error("query step: " + detail);
    }

    sqlite3_finalize(statement);
    return result;
}
