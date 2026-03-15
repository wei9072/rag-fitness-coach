import sqlite3
import os
from contextlib import contextmanager
from typing import Generator

# 設定 DB 存放路徑 (預設放在 data/ 資料夾內)
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "fitai_memory.db")

def init_db():
    """初始化 SQLite 資料庫與 Schema (Users, Sessions, Chat_History)"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Users Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
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

# 在啟動時自動初始化資料表
init_db()
