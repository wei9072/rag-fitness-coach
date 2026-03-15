"""
認證服務 (Auth Service)
負責密碼雜湊、JWT 簽發與驗證。
"""
import uuid
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from jose import jwt, JWTError
from src.config.settings import JWT_SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRE_MINUTES
from src.db.database import get_db_connection


# ════════════════════════════════════════
# 密碼工具 (使用 hashlib 替代 passlib，減少依賴)
# ════════════════════════════════════════

def hash_password(plain: str) -> str:
    """SHA-256 + salt 雜湊密碼"""
    salted = f"fitai_salt_{plain}_2026"
    return sha256(salted.encode()).hexdigest()


def verify_password(plain: str, hashed: str) -> bool:
    """驗證明文密碼是否與雜湊匹配"""
    return hash_password(plain) == hashed


# ════════════════════════════════════════
# JWT 工具
# ════════════════════════════════════════

def create_access_token(user_id: str, username: str) -> str:
    """簽發 JWT Token"""
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "username": username,
        "exp": expire
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict | None:
    """解碼 JWT Token，失敗回傳 None"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


# ════════════════════════════════════════
# 使用者 CRUD
# ════════════════════════════════════════

def register_user(username: str, password: str) -> dict:
    """註冊新使用者，回傳 {user_id, username}"""
    user_id = str(uuid.uuid4())
    pw_hash = hash_password(password)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # 檢查是否已存在
        cursor.execute("SELECT 1 FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            raise ValueError("此帳號已被註冊")
        
        cursor.execute(
            "INSERT INTO users (user_id, username, password_hash) VALUES (?, ?, ?)",
            (user_id, username, pw_hash)
        )
        conn.commit()
    
    return {"user_id": user_id, "username": username}


def authenticate_user(username: str, password: str) -> dict | None:
    """驗證帳密，成功回傳使用者資訊"""
    pw_hash = hash_password(password)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_id, username FROM users WHERE username = ? AND password_hash = ?",
            (username, pw_hash)
        )
        row = cursor.fetchone()
        if row:
            return {"user_id": row["user_id"], "username": row["username"]}
    return None
