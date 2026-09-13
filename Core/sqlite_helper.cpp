#include "sqlite_helper.hpp"

#include <utility>

SQLiteHelper::SQLiteHelper(const std::string& file) {
    const int open_rc = sqlite3_open(file.c_str(), &db_);
    if (open_rc != SQLITE_OK) {
        const std::string message = db_ ? sqlite3_errmsg(db_) : "unknown sqlite open error";
        closeConnection();
        throw std::runtime_error("open: " + message);
    }

    try {
        check(sqlite3_busy_timeout(db_, 5000), "busy timeout");
        execute("PRAGMA foreign_keys = ON;");
    }
    catch (...) {
        closeConnection();
        throw;
    }
}

SQLiteHelper::~SQLiteHelper() {
    finalizeStatement();
    closeConnection();
}

void SQLiteHelper::check(int rc, const char* operation) const {
    if (rc == SQLITE_OK || rc == SQLITE_DONE || rc == SQLITE_ROW) return;
    const char* detail = db_ ? sqlite3_errmsg(db_) : "unknown sqlite error";
    throw std::runtime_error(std::string(operation) + ": " + detail);
}

void SQLiteHelper::finalizeStatement() noexcept {
    if (!stmt_) return;
    sqlite3_finalize(stmt_);
    stmt_ = nullptr;
}

void SQLiteHelper::closeConnection() noexcept {
    if (!db_) return;
    sqlite3_close(db_);
    db_ = nullptr;
}

void SQLiteHelper::execute(const std::string& sql) {
    char* error = nullptr;
    const int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &error);
    if (rc == SQLITE_OK) return;

    const std::string detail = error ? error : (db_ ? sqlite3_errmsg(db_) : "unknown sqlite error");
    if (error) sqlite3_free(error);
    throw std::runtime_error("exec: " + detail);
}

void SQLiteHelper::begin() { execute("BEGIN TRANSACTION;"); }
void SQLiteHelper::commit() { execute("COMMIT;"); }
void SQLiteHelper::rollback() { execute("ROLLBACK;"); }

void SQLiteHelper::prepare(const std::string& sql) {
    finalizeStatement();
    check(sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt_, nullptr), "prepare");
}

void SQLiteHelper::bind(int index, int value) {
    if (!stmt_) throw std::runtime_error("bind int: no prepared statement");
    check(sqlite3_bind_int(stmt_, index, value), "bind int");
}

void SQLiteHelper::bind(int index, long long value) {
    if (!stmt_) throw std::runtime_error("bind int64: no prepared statement");
    check(sqlite3_bind_int64(stmt_, index, static_cast<sqlite3_int64>(value)), "bind int64");
}

void SQLiteHelper::bind(int index, double value) {
    if (!stmt_) throw std::runtime_error("bind double: no prepared statement");
    check(sqlite3_bind_double(stmt_, index, value), "bind double");
}

void SQLiteHelper::bind(int index, const std::string& value) {
    if (!stmt_) throw std::runtime_error("bind text: no prepared statement");
    check(
        sqlite3_bind_text(
            stmt_, index, value.c_str(), static_cast<int>(value.size()), SQLITE_TRANSIENT),
        "bind text");
}

void SQLiteHelper::bind(int index, const void* blob, int size) {
    if (!stmt_) throw std::runtime_error("bind blob: no prepared statement");
    check(sqlite3_bind_blob(stmt_, index, blob, size, SQLITE_TRANSIENT), "bind blob");
}

void SQLiteHelper::step() {
    if (!stmt_) throw std::runtime_error("step: no prepared statement");

    const int rc = sqlite3_step(stmt_);
    if (rc != SQLITE_DONE) {
        const std::string detail = db_ ? sqlite3_errmsg(db_) : "unknown sqlite error";
        finalizeStatement();
        throw std::runtime_error("step: expected SQLITE_DONE: " + detail);
    }
    finalizeStatement();
}

long long SQLiteHelper::stepInt64() {
    if (!stmt_) throw std::runtime_error("stepInt64: no prepared statement");

    const int rc = sqlite3_step(stmt_);
    if (rc != SQLITE_ROW) {
        const std::string detail = db_ ? sqlite3_errmsg(db_) : "unknown sqlite error";
        finalizeStatement();
        throw std::runtime_error("stepInt64: expected SQLITE_ROW: " + detail);
    }

    const long long value = static_cast<long long>(sqlite3_column_int64(stmt_, 0));
    const int tail_rc = sqlite3_step(stmt_);
    if (tail_rc != SQLITE_DONE) {
        const std::string detail = db_ ? sqlite3_errmsg(db_) : "unknown sqlite error";
        finalizeStatement();
        throw std::runtime_error("stepInt64: statement did not finish: " + detail);
    }

    finalizeStatement();
    return value;
}

std::vector<SQLiteHelper::Row> SQLiteHelper::query(const std::string& sql) {
    std::vector<Row> result;
    sqlite3_stmt* statement = nullptr;
    const int prepare_rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &statement, nullptr);
    if (prepare_rc != SQLITE_OK) {
        if (statement) sqlite3_finalize(statement);
        check(prepare_rc, "query prepare");
    }

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
        const std::string detail = db_ ? sqlite3_errmsg(db_) : "unknown sqlite error";
        sqlite3_finalize(statement);
        throw std::runtime_error("query step: " + detail);
    }

    sqlite3_finalize(statement);
    return result;
}
