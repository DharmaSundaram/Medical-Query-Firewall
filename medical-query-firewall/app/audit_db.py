# app/audit_db.py
import sqlite3
from pathlib import Path
import json
from typing import Dict, Any, List

# DB file stored at project_root/data/audit.db
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "audit.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        session_id TEXT,
        raw_text TEXT,
        masked_text TEXT,
        pii TEXT,
        decision TEXT,
        classifier_json TEXT,
        matched_rules TEXT,
        block_hits TEXT,
        warn_hits TEXT,
        reviewer_decision TEXT
    )
    """)
    conn.commit()
    conn.close()

def insert_audit(audit: Dict[str, Any]) -> int:
    """Insert an audit record. Returns the new row id."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO audits (
        timestamp, session_id, raw_text, masked_text, pii, decision,
        classifier_json, matched_rules, block_hits, warn_hits, reviewer_decision
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        audit.get("timestamp"),
        audit.get("session_id"),
        audit.get("raw_text"),
        audit.get("masked_text"),
        json.dumps(audit.get("pii") or []),
        audit.get("decision"),
        json.dumps(audit.get("classifier") or {}),
        json.dumps(audit.get("matched_rules") or []),
        json.dumps(audit.get("block_hits") or []),
        json.dumps(audit.get("warn_hits") or []),
        None
    ))
    conn.commit()
    rowid = cur.lastrowid
    conn.close()
    return rowid

def fetch_audits(limit: int = 200) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM audits ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        result.append(d)
    return result

def fetch_audit_by_id(aid: int) -> Dict[str, Any]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM audits WHERE id=?", (aid,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def set_reviewer_decision(aid: int, decision: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE audits SET reviewer_decision=? WHERE id=?", (decision, aid))
    conn.commit()
    conn.close()
