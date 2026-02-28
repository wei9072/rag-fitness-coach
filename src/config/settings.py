import os
import pathlib
from dotenv import load_dotenv

# ── 系統路徑 ──────────────────────────────────────────────
BASE_DIR  = pathlib.Path(__file__).resolve().parent.parent.parent
INDEX_DIR = BASE_DIR / "data" / "faiss_index"

# ── 載入環境變數 ──────────────────────────────────────────
load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY or GROQ_API_KEY == "your_key_here":
    raise RuntimeError("請在 .env 中設定有效的 GROQ_API_KEY")

# ── 模型常數 ──────────────────────────────────────────────
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
LLM_MODEL  = "llama-3.3-70b-versatile"
TOP_K      = 10
