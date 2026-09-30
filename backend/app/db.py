"""SQLite via stdlib. One connection per call (thread-safe enough for our queue)."""
import json
import sqlite3
from datetime import datetime, timezone

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    name TEXT,
    source_type TEXT,
    source_url TEXT,
    sha256 TEXT,
    status TEXT,
    progress REAL DEFAULT 0,
    error TEXT,
    created_at TEXT,
    finished_at TEXT,
    video_path TEXT,
    meta TEXT,
    scenes TEXT,
    frames TEXT,
    estimate TEXT,
    report_md TEXT,
    transcript TEXT,
    ocr TEXT,
    analysis TEXT,
    multi TEXT,
    provider TEXT,
    model TEXT,
    mode TEXT
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_sha ON jobs(sha256);
"""

# migrasi kolom untuk DB lama (idempotent)
MIGRATIONS = [
    "ALTER TABLE jobs ADD COLUMN transcript TEXT",
    "ALTER TABLE jobs ADD COLUMN ocr TEXT",
    "ALTER TABLE jobs ADD COLUMN analysis TEXT",
    "ALTER TABLE jobs ADD COLUMN multi TEXT",
    """CREATE TABLE IF NOT EXISTS recreations(
        id TEXT PRIMARY KEY, original_id TEXT, generated_id TEXT,
        parent_id TEXT, created_at TEXT, scores TEXT,
        improved_prompt TEXT, report_md TEXT)""",
]


def _conn():
    c = sqlite3.connect(str(config.DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def init():
    with _conn() as c:
        c.executescript(SCHEMA)
        for sql in MIGRATIONS:
            try:
                c.execute(sql)
            except sqlite3.OperationalError:
                pass  # kolom sudah ada


def now():
    return datetime.now(timezone.utc).isoformat()


def create_job(job: dict):
    with _conn() as c:
        c.execute(
            """INSERT INTO jobs (id,name,source_type,source_url,sha256,status,progress,
               error,created_at,finished_at,video_path,meta,scenes,frames,estimate,
               report_md,transcript,ocr,analysis,multi,provider,model,mode)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (job["id"], job.get("name"), job.get("source_type"), job.get("source_url"),
             job.get("sha256"), job.get("status", "queued"), job.get("progress", 0),
             job.get("error"), job.get("created_at", now()), job.get("finished_at"),
             job.get("video_path"), _j(job.get("meta")), _j(job.get("scenes")),
             _j(job.get("frames")), _j(job.get("estimate")), job.get("report_md"),
             _j(job.get("transcript")), _j(job.get("ocr")), _j(job.get("analysis")), _j(job.get("multi")),
             job.get("provider"), job.get("model"), job.get("mode")),
        )


def _j(v):
    return json.dumps(v) if v is not None else None


def update_job(job_id: str, **fields):
    fields = {k: (_j(v) if k in ("meta", "scenes", "frames", "estimate",
                                 "transcript", "ocr", "analysis", "multi") else v)
              for k, v in fields.items()}
    sets = ", ".join(f"{k}=?" for k in fields)
    with _conn() as c:
        c.execute(f"UPDATE jobs SET {sets} WHERE id=?", (*fields.values(), job_id))


def create_recreation(rec: dict):
    with _conn() as c:
        c.execute(
            "INSERT INTO recreations (id,original_id,generated_id,parent_id,created_at,"
            "scores,improved_prompt,report_md) VALUES (?,?,?,?,?,?,?,?)",
            (rec["id"], rec["original_id"], rec["generated_id"], rec.get("parent_id"),
             rec["created_at"], json.dumps(rec.get("scores") or {}, ensure_ascii=False),
             rec.get("improved_prompt"), rec.get("report_md")))


def list_recreations(original_id: str):
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM recreations WHERE original_id=? ORDER BY created_at",
            (original_id,)).fetchall()
    return rows


def get_job(job_id: str):
    with _conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return _row(row)


def _row(row):
    if not row:
        return None
    d = dict(row)
    for k in ("meta", "scenes", "frames", "estimate", "transcript", "ocr", "analysis", "multi"):
        d[k] = json.loads(d[k]) if d[k] else None
    return d


def list_jobs():
    with _conn() as c:
        rows = c.execute(
            "SELECT id,name,source_type,status,progress,error,created_at,finished_at,mode "
            "FROM jobs ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def delete_job(job_id: str):
    with _conn() as c:
        c.execute("DELETE FROM jobs WHERE id=?", (job_id,))


def find_cached(sha256: str):
    """Latest completed job with the same video hash."""
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM jobs WHERE sha256=? AND status='completed' "
            "ORDER BY finished_at DESC LIMIT 1", (sha256,)).fetchone()
    return _row(row)


def reset_stale():
    with _conn() as c:
        c.execute("UPDATE jobs SET status='failed', error='worker restart saat job berjalan' "
                  "WHERE status NOT IN ('completed','failed')")


def get_setting(key: str):
    with _conn() as c:
        row = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def set_setting(key: str, value: str):
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO settings (key,value,updated_at) VALUES (?,?,?)",
                  (key, value, now()))
