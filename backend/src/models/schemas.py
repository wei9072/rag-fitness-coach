from typing import Literal, Optional
from pydantic import BaseModel, Field

# ── Auth 請求與回應 ──────────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str

class UserInfo(BaseModel):
    user_id: str
    username: str

# ── FastAPI Chat 請求與回應 ──────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None  # 用於追蹤對話上下文 (Memory)
    api_key: str | None = None
    llm_provider: str = "groq"
    model_name: str | None = None
    is_paid: bool = False
    user_profile: str | None = None

class ChatResponse(BaseModel):
    answer: str
    rewritten_query: str = ""
    sources: list[str] = []
    session_id: str  # 回傳給前端，前端後續請求需帶上此 ID

# ── 內部通訊用模型（不直接暴露給前端）────────────────────────────
class RouteDecision(BaseModel):
    """決定如何檢索健身紀錄的路由結果。"""
    intent: Literal["semantic", "temporal", "all"] = Field(
        ..., description="檢索意圖：'semantic' (語意搜尋), 'temporal' (時間排序), 'all' (全量統計)"
    )
    intent_category: Literal["QA_INTENT", "PLANNING_INTENT", "AGGREGATE_INTENT"] = Field(
        ..., description="意圖大類：查問歷史紀錄時選 'QA_INTENT'，要求規劃未來菜單/安排計畫時選 'PLANNING_INTENT'，詢問數值統計（最重、次數、平均、PR）時選 'AGGREGATE_INTENT'"
    )
    n_count: int = Field(
        5, description="如果是 'temporal' 意圖，需要獲取的紀錄筆數"
    )
    refined_query: str = Field(
        ..., description="檢索關鍵詞"
    )


class SqlGeneration(BaseModel):
    """LLM 結構化輸出：Text-to-SQL 生成結果"""
    sql: str = Field(description="生成的 SQLite SELECT 語句，user_id 使用 :user_id 參數佔位符")
    explanation: str = Field(description="簡短說明這條 SQL 查詢的目的")


# ── Upload 請求與回應 ──────────────────────────────────────────
class UploadRequest(BaseModel):
    content: str  # 原始 TXT 訓練紀錄內容
    source_name: str | None = None

class UploadResponse(BaseModel):
    chunks_added: int
    sql_records: int
    message: str
