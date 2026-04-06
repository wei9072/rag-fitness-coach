import sqlite3
import os
from contextlib import contextmanager
from typing import Generator

# 設定 DB 存放路徑 (放在 backend/data/ 資料夾內)
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "fitai_memory.db")

def init_db():
    """初始化 SQLite 資料庫與 Schema (Users, Sessions, Chat_History)"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Users Table (擴充版：含帳號密碼)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE,
                password_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Sessions Table (一個 User 可以有多個對話階段)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
            )
        """)
        
        # Chat History Table (記錄具體對話內容)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
            )
        """)
        
        # Training Logs Table (結構化訓練紀錄，供聚合查詢 Text-to-SQL)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                date TEXT NOT NULL,
                exercise TEXT NOT NULL,
                weight_kg REAL,
                reps INTEGER NOT NULL,
                sets INTEGER NOT NULL DEFAULT 1,
                note TEXT,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tl_user_date ON training_logs(user_id, date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tl_user_exercise ON training_logs(user_id, exercise)")

        # 嘗試 ALTER TABLE 補齊 username / password_hash，如果舊資料庫沒有這些欄位
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN username TEXT UNIQUE")
        except sqlite3.OperationalError:
            pass  # 欄位已存在

        try:
            cursor.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
        except sqlite3.OperationalError:
            pass  # 欄位已存在
        
        conn.commit()
    print(f"📦 [DB] SQLite Database Initialized at {DB_PATH}")

@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """提供可管理連線生命週期的 Context Manager"""
    conn = sqlite3.connect(DB_PATH)
    # 支援透過 column name 存取
    conn.row_factory = sqlite3.Row
    try:
        # 強制開啟外鍵約束 (SQLite 預設關閉)
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
    finally:
        conn.close()

# ════════════════════════════════════════
# Training Logs 工具函式
# ════════════════════════════════════════

def insert_training_logs(records: list[dict]) -> int:
    """批次寫入結構化訓練紀錄，回傳插入筆數。"""
    if not records:
        return 0
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.executemany(
            """INSERT INTO training_logs
               (user_id, date, exercise, weight_kg, reps, sets, note, source)
               VALUES (:user_id, :date, :exercise, :weight_kg, :reps, :sets, :note, :source)""",
            records
        )
        conn.commit()
        return cursor.rowcount


def clear_training_logs_by_source(source: str, user_id: str = "default_user"):
    """清除指定來源檔案的訓練紀錄（冪等重建用）。"""
    with get_db_connection() as conn:
        conn.execute(
            "DELETE FROM training_logs WHERE source = ? AND user_id = ?",
            (source, user_id)
        )
        conn.commit()


def execute_safe_select(sql: str, params: dict | None = None) -> list[dict]:
    """以唯讀模式執行 SELECT 查詢。非 SELECT 語句會被 SQLite 層級拒絕。"""
    normalized = sql.strip().upper()
    if not normalized.startswith("SELECT"):
        raise ValueError("只允許 SELECT 查詢")

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql, params or {})
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


# 在啟動時自動初始化資料表
init_db()
