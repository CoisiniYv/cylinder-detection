#include "sqlite_helper.hpp"

#include <cstring>

SQLiteHelper::SQLiteHelper(const std::string& file) {
	int rc = sqlite3_open(file.c_str(), &db_);
	check(rc, "open");
}
SQLiteHelper::~SQLiteHelper() {
	if (stmt_) sqlite3_finalize(stmt_);
	if (db_)   sqlite3_close(db_);
}

void SQLiteHelper::check(int rc, const char* msg) {
	if (rc != SQLITE_OK && rc != SQLITE_DONE)
		throw std::runtime_error(msg + std::string(": ") +
			(err_ ? err_ : sqlite3_errmsg(db_)));
}

void SQLiteHelper::execute(const std::string& sql) {
	int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err_);
	check(rc, "exec");
	if (err_) { sqlite3_free(err_); err_ = nullptr; }
}

void SQLiteHelper::begin() { execute("BEGIN TRANSACTION;"); }
void SQLiteHelper::commit() { execute("COMMIT;"); }
void SQLiteHelper::rollback() { execute("ROLLBACK;"); }

/* ---------- prepared statement ---------- */
void SQLiteHelper::prepare(const std::string& sql) {
	if (stmt_) sqlite3_finalize(stmt_);
	int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt_, nullptr);
	check(rc, "prepare");
}
void SQLiteHelper::bind(int idx, int val) {
	sqlite3_bind_int(stmt_, idx, val);
}
void SQLiteHelper::bind(int idx, long long val) {
	sqlite3_bind_int64(stmt_, idx, static_cast<sqlite3_int64>(val));
}
void SQLiteHelper::bind(int idx, double val) {
	sqlite3_bind_double(stmt_, idx, val);
}
void SQLiteHelper::bind(int idx, const std::string& val) {
	sqlite3_bind_text(stmt_, idx, val.c_str(), val.size(), SQLITE_TRANSIENT);
}
void SQLiteHelper::bind(int idx, const void* blob, int n) {
	sqlite3_bind_blob(stmt_, idx, blob, n, SQLITE_TRANSIENT);
}
void SQLiteHelper::step() {
	int rc = sqlite3_step(stmt_);
	check(rc, "step");
	sqlite3_finalize(stmt_);
	stmt_ = nullptr;
}
void SQLiteHelper::reset() { sqlite3_reset(stmt_); }

/* ---------- 查询 ---------- */
std::vector<SQLiteHelper::Row>
SQLiteHelper::query(const std::string& sql) {
	std::vector<Row> result;
	sqlite3_stmt* st;
	int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &st, nullptr);
	check(rc, "query prepare");
	while ((rc = sqlite3_step(st)) == SQLITE_ROW) {
		Row row;
		int cols = sqlite3_column_count(st);
		for (int i = 0; i < cols; ++i) {
			const unsigned char* txt = sqlite3_column_text(st, i);
			row.emplace_back(txt ? reinterpret_cast<const char*>(txt) : "");
		}
		result.emplace_back(std::move(row));
	}
	sqlite3_finalize(st);
	return result;
}