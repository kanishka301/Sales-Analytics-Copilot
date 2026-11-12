"""
Safety layers:
  1. validate_sql: one statement, must start with SELECT/WITH, forbidden keywords blocked
  2. database opened READ-ONLY with external file access disabled (src/db.py), so even a missed
     check cannot modify data or read other files on disk
  3. row limit and a timeout (query is interrupted)
"""
import datetime
import decimal
import re
import threading

from src.config import SQL_ROW_LIMIT, SQL_TIMEOUT_S


class SqlRejected(ValueError):
    pass


_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|pragma|install|load|"
    r"export|import|call|truncate|grant|vacuum|checkpoint)\b|\bread_[a-z_]+|\bglob\b|https?:|\bfile:",
    re.I,
)


def _clean(q: str) -> str:
    q = re.sub(r"--[^\n]*", "", q)
    q = re.sub(r"/\*.*?\*/", "", q, flags=re.S)
    return re.sub(r"'(?:[^']|'')*'", "''", q)  # blank out string literals


def validate_sql(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise SqlRejected("Empty query.")
    q = query.strip()
    while q.endswith(";"):
        q = q[:-1].rstrip()
    if re.search(r"\b(from|join)\s*\(?\s*'", re.sub(r"--[^\n]*", "", q), re.I):
        raise SqlRejected("Query tables by name (sales_daily, sales_weekly), not by file path.")
    cleaned = _clean(q)
    if ";" in cleaned:
        raise SqlRejected("Only one statement is allowed.")
    if not re.match(r"^\(?\s*(select|with)\b", cleaned, re.I):
        raise SqlRejected("Only SELECT queries are allowed.")
    m = _FORBIDDEN.search(cleaned)
    if m:
        raise SqlRejected(f"Forbidden keyword or function: {m.group(0)!r}")
    return q


def _jsonable(v):
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return None if v != v else round(v, 4)
    if isinstance(v, decimal.Decimal):
        return round(float(v), 4)
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.isoformat()
    return str(v)


def run_sql(query: str, limit: int = SQL_ROW_LIMIT, cursor=None) -> dict:
    q = validate_sql(query)
    if cursor is None:
        from src.db import get_cursor
        cursor = get_cursor()
    timer = threading.Timer(SQL_TIMEOUT_S, cursor.interrupt)
    timer.start()
    try:
        res = cursor.execute(q)
        cols = [d[0] for d in res.description]
        rows = res.fetchmany(limit + 1)
    finally:
        timer.cancel()
    truncated = len(rows) > limit
    rows = [[_jsonable(v) for v in r] for r in rows[:limit]]
    return {"columns": cols, "rows": rows, "row_count": len(rows), "truncated": truncated}
