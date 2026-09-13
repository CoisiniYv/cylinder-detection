#include "sqlite_helper.hpp"

#include <cstring>

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

    // SQLite connection-local settings must be applied to every connection.
    execute("PRAGMA foreign_keys = ON;");
    sqlite3_busy_timeout(db_, 5000);
}

SQLiteHelper::~SQLiteHelper() {
    if (stmt_) sqlite3_finalize(stmt_);
    if (db_) sqlite3_close(db_);
}

void SQLiteHelper::check(int rc, const char* msg) {
    if (rc == SQLITE_OK || rc == SQLITE_DONE || rc == SQLITE_ROW) return;
    const char* detail = err_ ? err_ : (db_ ? sqlite3_errmsg(db_) : "unknown sqlite error");
    throw std::runtime_error(std::string(msg) + ": " + detail);
}

void SQLiteHelper::execute(const std::string& sql) {
    if (err_) {
        sqlite3_free(err_);
        err_ = nullptr;
    }
    const int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err_);
    if (rc != SQLITE_OK) {
        const std::string detail = err_ ? err_ : sqlite3_errmsg(db_);
        if (err_) {
            sqlite3_free(err_);
            err_ = nullptr;
        }
        throw std::runtime_error("exec: " + detail);
    }
}

void SQLiteHelper::begin() { execute("BEGIN TRANSACTION;"); }
void SQLiteHelper::commit() { execute("COMMIT;"); }
void SQLiteHelper::rollback() { execute("ROLLBACK;"); }

void SQLiteHelper::prepare(const std::string& sql) {
    if (stmt_) {
        sqlite3_finalize(stmt_);
        stmt_ = nullptr;
    }
    const int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt_, nullptr);
    check(rc, "prepare");
}

void SQLiteHelper::bind(int idx, int val) {
    check(sqlite3_bind_int(stmt_, idx, val), "bind int");
}

void SQLiteHelper::bind(int idx, long long val) {
    check(sqlite3_bind_int64(stmt_, idx, static_cast<sqlite3_int64>(val)), "bind int64");
}

void SQLiteHelper::bind(int idx, double val) {
    check(sqlite3_bind_double(stmt_, idx, val), "bind double");
}

void SQLiteHelper::bind(int idx, const std::string& val) {
    check(sqlite3_bind_text(stmt_, idx, val.c_str(), static_cast<int>(val.size()), SQLITE_TRANSIENT), "bind text");
}

void SQLiteHelper::bind(int idx, const void* blob, int n) {
    check(sqlite3_bind_blob(stmt_, idx, blob, n, SQLITE_TRANSIENT), "bind blob");
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
    sqlite3_stmt* st = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &st, nullptr);
    check(rc, "query prepare");

    while ((rc = sqlite3_step(st)) == SQLITE_ROW) {
        Row row;
        const int cols = sqlite3_column_count(st);
        row.reserve(cols);
        for (int i = 0; i < cols; ++i) {
            const unsigned char* txt = sqlite3_column_text(st, i);
            row.emplace_back(txt ? reinterpret_cast<const char*>(txt) : "");
        }
        result.emplace_back(std::move(row));
    }

    if (rc != SQLITE_DONE) {
        const std::string detail = sqlite3_errmsg(db_);
        sqlite3_finalize(st);
        throw std::runtime_error("query step: " + detail);
    }

    sqlite3_finalize(st);
    return result;
}
