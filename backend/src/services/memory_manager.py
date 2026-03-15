import uuid
from typing import TypedDict, List
from src.db.database import get_db_connection

class ChatMessage(TypedDict):
    role: str
    content: str

class MemoryManager:
    """負責處理長期記憶與 SQLite RDBMS 互動的 Service Layer"""
    
    def create_or_get_user(self, user_id: str) -> None:
        """確保使用者存在於 Users 表中"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO users (user_id) VALUES (?)",
                (user_id,)
            )
            conn.commit()

    def create_session(self, user_id: str) -> str:
        """為使用者建立一個全新的對話 Session (UUID)"""
        self.create_or_get_user(user_id)
        session_id = str(uuid.uuid4())
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (session_id, user_id) VALUES (?, ?)",
                (session_id, user_id)
            )
            conn.commit()
        return session_id

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        將單句對話存入 Chat_History 表。
        role 必須是 'user', 'assistant', 'system'
        """
        if role not in ('user', 'assistant', 'system'):
            raise ValueError(f"不允許的對話角色: {role}")
            
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # 檢查 Session 存不存在
            cursor.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,))
            if not cursor.fetchone():
                raise ValueError(f"Session {session_id} 不存在")
                
            cursor.execute(
                "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content.strip())
            )
            conn.commit()

    def get_recent_messages(self, session_id: str, limit: int = 5) -> List[ChatMessage]:
        """
        滑動視窗 (Sliding Window Context):
        抓取該 Session 中最新 N"組" (問+答) 對話紀錄。
        為了保證對話完整性，其實是抓出最新 2*N 筆紀錄。
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # 依照時間反著排序取出最新的 limit*2 筆留言，然後再正向排序
            cursor.execute(
                """
                SELECT role, content 
                FROM (
                    SELECT role, content, id
                    FROM chat_history 
                    WHERE session_id = ? 
                    ORDER BY id DESC 
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (session_id, limit * 2)  # 一次提問包含一問一答，所以要 * 2
            )
            rows = cursor.fetchall()
            messages: List[ChatMessage] = [{"role": row["role"], "content": row["content"]} for row in rows]
            return messages
            
    def format_history_for_prompt(self, messages: List[ChatMessage]) -> str:
        """
        將從 DB 取出來的 List[Dict] 轉成純文字，方便塞入 LLM 的 Prompt 中
        """
        if not messages:
            return "無。"
            
        formatted_str = ""
        for msg in messages:
            role_label = "使用者 (User)" if msg["role"] == "user" else "AI 教練 (Assistant)"
            formatted_str += f"{role_label}: {msg['content']}\n"
            
        return formatted_str.strip()

# Singleton 實體供其他模組呼叫
memory_manager = MemoryManager()
