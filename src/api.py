"""
api.py — FastAPI 伺服器（LangChain 架構）
Query Rewriting → FAISS 檢索（含時間意圖偵測）→ Groq LLM 生成。
"""

import os
import pathlib
import re

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage

# ── 環境變數 ──────────────────────────────────────────────
load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY or GROQ_API_KEY == "your_key_here":
    raise RuntimeError("請在 .env 中設定有效的 GROQ_API_KEY")

# ── 路徑與常數 ────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).resolve().parent.parent
INDEX_DIR  = BASE_DIR / "data" / "faiss_index"

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
LLM_MODEL  = "llama-3.3-70b-versatile"
TOP_K      = 10

SYSTEM_PROMPT = (
    "你是一位擁有 10 年經驗的專業健身教練，根據使用者的訓練紀錄並結合自身專業知識來回答問題。\n\n"
    "## 回覆原則\n"
    "1. **分析能力**：主動分析使用者的訓練趨勢與進步幅度，例如重量變化、組數提升等。\n"
    "2. **訓練建議**：根據紀錄給出下次訓練的具體建議，包括重量調整、組數安排、動作改進方向。\n"
    "3. **格式規範**：回覆請使用條列式或表格呈現重點，讓資訊清晰易讀。\n"
    "4. **安全提醒**：若發現紀錄中有潛在風險（如重量驟增、疑似不適備註），請主動提醒注意姿勢與傷痛預防。\n\n"
    "## 限制\n"
    "- 若紀錄中無相關資訊，請誠實回答不知道。\n"
    "- 回覆時請使用繁體中文。\n"
    "### 使用者設定\n"
    "姓名：陳韋成\n"
    "年齡：25\n"
    "性別：男\n"
    "身高：175cm\n"
    "體重：70kg\n"
    "目標：增肌\n"
    "訓練頻率：一週三次\n"
    "訓練經驗：一年\n"
    "訓練偏好：自由重量\n"
    "飲食偏好：高蛋白、低碳水\n"
    "飲食限制：無\n"
    "過敏：無\n"
    "傷病史：無\n"
    "其他備註：消化系統較弱，容易脹氣，建議多攝取益生菌\n"
    "### 回話風格\n"
    "親切、專業、有耐心、會主動關心使用者狀況"
)

# ── 啟動時載入 LangChain 元件 ─────────────────────────────
print("⏳ 載入 Embedding 模型...")
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True},
)
print("✅ Embedding 模型載入完成")

print("⏳ 載入 FAISS 向量庫...")
if not INDEX_DIR.exists():
    raise FileNotFoundError(
        f"找不到 FAISS 索引：{INDEX_DIR}，請先執行 python src/indexer.py"
    )
vectorstore = FAISS.load_local(
    str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
)
print(f"✅ FAISS 載入完成（{vectorstore.index.ntotal} 筆向量）")

# LangChain Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

# LangChain LLM
llm = ChatGroq(
    model=LLM_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.3,
    max_tokens=1024,
)

# 用於 Query Rewriting 的輕量 LLM 呼叫
rewrite_llm = ChatGroq(
    model=LLM_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.0,
    max_tokens=50,
)


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


# ── Query Rewriting ──────────────────────────────────────
_REWRITE_SYSTEM = (
    "你是一個搜尋查詢改寫助手。"
    "將使用者的問題改寫成簡短、精準的中文搜尋關鍵詞（不超過 20 字），"
    "只保留對健身紀錄搜尋最有用的核心詞彙，去除問候、修飾語等廢話。"
    "只輸出改寫後的關鍵詞，不要加任何解釋。"
)


def rewrite_query(question: str) -> str:
    """用 LangChain ChatGroq 改寫問題為精準搜尋關鍵詞。"""
    try:
        resp = rewrite_llm.invoke([
            SystemMessage(content=_REWRITE_SYSTEM),
            HumanMessage(content=question),
        ])
        rewritten = resp.content.strip()
        print(f"  🔄 Query Rewrite: 「{question}」→「{rewritten}」")
        return rewritten
    except Exception as e:
        print(f"  ⚠️ Query Rewrite 失敗：{e}")
        return question


# ── 時間意圖偵測 ──────────────────────────────────────────
_TEMPORAL_KEYWORDS = re.compile(
    r"最近|最新|上次|上一次|近期|最後|前(\d+)筆|最近(\d+)筆|近(\d+)次|前(\d+)次"
)
_NUMBER_RE = re.compile(r"(\d+)")

# ── 全量意圖偵測（統計、總覽類問題）────────────────────────
_ALL_KEYWORDS = re.compile(
    r"幾筆|幾次|多少筆|多少次|總共|全部|所有|統計|總結|總覽|整體|彙整|彙總"
)


def detect_temporal_intent(question: str) -> tuple[bool, int]:
    """偵測問題是否包含時間排序意圖，回傳 (is_temporal, n)。"""
    if not _TEMPORAL_KEYWORDS.search(question):
        return False, 0
    nums = _NUMBER_RE.findall(question)
    n = int(nums[0]) if nums else 5
    return True, n


def detect_all_intent(question: str) -> bool:
    """偵測問題是否需要回傳全部紀錄（統計、總覽類問題）。"""
    return bool(_ALL_KEYWORDS.search(question))


def retrieve_all() -> list[str]:
    """回傳向量庫中所有文件，按日期降序排列。"""
    all_docs = list(vectorstore.docstore._dict.values())
    dated = [
        (d.metadata.get("date") or "", d.page_content)
        for d in all_docs
    ]
    dated.sort(key=lambda x: x[0] if x[0] else "0000-00-00", reverse=True)
    print(f"  📊 全量檢索：共 {len(dated)} 筆紀錄")
    return [text for _, text in dated]


def retrieve_by_date(n: int) -> list[str]:
    """從向量庫取出所有文件，按日期降序排列後取前 n 筆。"""
    all_docs = list(vectorstore.docstore._dict.values())
    dated = [
        (d.metadata.get("date") or "", d.page_content)
        for d in all_docs
    ]
    dated.sort(key=lambda x: x[0] if x[0] else "0000-00-00", reverse=True)
    print(f"  📅 時間意圖檢索：取最新 {n} 筆（共 {len(dated)} 筆）")
    return [text for _, text in dated[:n]]


# ── 智慧檢索 ─────────────────────────────────────────────
def smart_retrieve(query: str, original_question: str) -> list[str]:
    """
    智慧檢索：
    - 偵測到全量意圖（幾筆/總共/全部）→ 回傳所有紀錄
    - 偵測到時間意圖（最近/最新/前N筆）→ 按日期排序
    - 否則 → LangChain Retriever 語意搜尋
    """
    # 全量意圖優先
    if detect_all_intent(original_question):
        return retrieve_all()

    is_temporal, n = detect_temporal_intent(original_question)
    if is_temporal:
        return retrieve_by_date(n)

    # LangChain FAISS Retriever
    docs = retriever.invoke(query)
    return [d.page_content for d in docs]


# ── API 端點 ─────────────────────────────────────────────
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """問題 → Query Rewriting → 智慧檢索 → LLM 生成回覆。"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="問題不可為空")

    # 1. Query Rewriting
    rewritten = rewrite_query(req.question)

    # 2. 智慧檢索
    relevant = smart_retrieve(rewritten, original_question=req.question)

    if not relevant:
        return ChatResponse(
            answer="找不到相關紀錄。",
            rewritten_query=rewritten,
            sources=[],
        )

    # 3. 組合 Prompt 並呼叫 LLM
    context_text = "\n\n".join(
        f"【紀錄 {i+1}】{c}" for i, c in enumerate(relevant)
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"以下是使用者的健身紀錄：\n\n{context_text}\n\n"
            f"使用者的問題：{req.question}"
        )),
    ]

    try:
        resp = llm.invoke(messages)
        answer = resp.content
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq API 錯誤：{e}")

    return ChatResponse(answer=answer, rewritten_query=rewritten, sources=relevant)


@app.get("/health")
async def health():
    return {"status": "ok", "index_size": vectorstore.index.ntotal}
