from typing import Literal
from pydantic import BaseModel, Field

# ── FastAPI 請求與回應 ──────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    api_key: str | None = None
    llm_provider: str = "groq"
    model_name: str | None = None
    is_paid: bool = False
    strategy: str = "B"
    user_profile: str | None = None

class ChatResponse(BaseModel):
    answer: str
    rewritten_query: str
    sources: list[str]

# ── 內部通訊用模型（不直接暴露給前端）────────────────────────────
class RouteDecision(BaseModel):
    """決定如何檢索健身紀錄的路由結果。"""
    intent: Literal["semantic", "temporal", "all"] = Field(
        ..., description="檢索意圖：'semantic' (語意搜尋), 'temporal' (時間排序), 'all' (全量統計)"
    )
    intent_category: Literal["QA_INTENT", "PLANNING_INTENT"] = Field(
        ..., description="意圖大類：查問歷史紀錄時選 'QA_INTENT'，要求規劃未來菜單/安排計畫時選 'PLANNING_INTENT'"
    )
    n_count: int = Field(
        5, description="如果是 'temporal' 意圖，需要獲取的紀錄筆數"
    )
    refined_query: str = Field(
        ..., description="檢索關鍵詞"
    )
