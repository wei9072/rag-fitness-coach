"""
api.py — FastAPI 伺服器
提供單純的 Web API 介面，RAG 與 LangChain 核心邏輯均封裝於 src/rag.py 中。
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.rag import generate_rag_response, get_index_size

# ── FastAPI App ──────────────────────────────────────────
app = FastAPI(title="RAG 健身教練 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ──────────────────────────────────────
class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    rewritten_query: str
    sources: list[str]


# ── API 端點 ─────────────────────────────────────────────
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """處理前端對話請求。核心 RAG 邏輯在 generate_rag_response() 執行。"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="問題不可為空")

    try:
        # 呼叫封裝好的 RAG 執行函數
        result = generate_rag_response(req.question)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "index_size": get_index_size()}
