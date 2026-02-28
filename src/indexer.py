"""
indexer.py — 使用 LangChain 讀取 data/ 資料夾，
向量化後存入 FAISS 本地索引。
"""

import json
import pathlib
import re

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ── 路徑與設定 ───────────────────────────────────────────
from src.config.settings import BASE_DIR, INDEX_DIR, MODEL_NAME

DATA_DIR  = BASE_DIR / "data"

# ── TXT 切塊設定 ─────────────────────────────────────────
CHUNK_SIZE    = 300
CHUNK_OVERLAP = 50


# ─────────────────────────────────────────────────────────
# 日期解析
# ─────────────────────────────────────────────────────────
_DATE_ISO_RE       = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_DATE_MMDD_RE      = re.compile(r"^\s*(\d{4})\s", re.MULTILINE)
_DATE_MMDD_SOLO_RE = re.compile(r"^\s*(\d{4})\s*$", re.MULTILINE)


def _parse_date(text: str, default_year: str = "2026") -> str | None:
    """從文字中萃取日期，回傳 YYYY-MM-DD 或 None。"""
    m = _DATE_ISO_RE.search(text)
    if m:
        return m.group(1)
    m = _DATE_MMDD_SOLO_RE.search(text)
    if not m:
        m = _DATE_MMDD_RE.search(text)
    if m:
        mmdd = m.group(1)
        month, day = mmdd[:2], mmdd[2:]
        if 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
            return f"{default_year}-{month}-{day}"
    return None


# ─────────────────────────────────────────────────────────
# 檔案載入 → LangChain Document
# ─────────────────────────────────────────────────────────
def load_json_documents(path: pathlib.Path) -> list[Document]:
    """讀取 JSON 陣列，每筆轉為 LangChain Document。"""
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    docs = []
    for record in records:
        if "sets" in record:
            sets_desc = "；".join(
                f"第{s['set']}組 {s['reps']}下 {s['weight_kg']}公斤"
                for s in record["sets"]
            )
            text = (
                f"日期：{record['date']}｜動作：{record['exercise']}｜"
                f"組數細節：{sets_desc}｜備註：{record['notes']}"
            )
            date = record.get("date")
        else:
            text = " | ".join(f"{k}：{v}" for k, v in record.items())
            date = _parse_date(text)

        docs.append(Document(
            page_content=text,
            metadata={"date": date, "source": path.name},
        ))
    return docs


def load_txt_documents(path: pathlib.Path) -> list[Document]:
    """讀取 TXT 檔，依段落切塊為 LangChain Document。"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    docs = []

    for para in paragraphs:
        date = _parse_date(para)
        meta = {"date": date, "source": path.name}

        if len(para) <= CHUNK_SIZE:
            docs.append(Document(page_content=para, metadata=meta))
        else:
            start = 0
            while start < len(para):
                chunk = para[start : start + CHUNK_SIZE]
                docs.append(Document(page_content=chunk, metadata=meta))
                start += CHUNK_SIZE - CHUNK_OVERLAP

    return docs


# ─────────────────────────────────────────────────────────
# 統一掃描
# ─────────────────────────────────────────────────────────
LOADERS = {".json": load_json_documents, ".txt": load_txt_documents}


def collect_documents() -> list[Document]:
    """掃描 data/ 下所有支援的檔案，回傳 Document 列表。"""
    all_docs: list[Document] = []

    for ext, loader in LOADERS.items():
        for path in sorted(DATA_DIR.glob(f"*{ext}")):
            docs = loader(path)
            print(f"  📄 {path.name}  →  {len(docs)} 個區塊")
            all_docs.extend(docs)

    if not all_docs:
        raise FileNotFoundError(f"在 {DATA_DIR} 中找不到任何 .json 或 .txt 檔案")

    with_date = sum(1 for d in all_docs if d.metadata.get("date"))
    print(f"\n✅ 合計 {len(all_docs)} 個區塊，其中 {with_date} 個有日期 Metadata")
    return all_docs


# ─────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────
def build_index():
    """讀取 → 向量化 → 建立 FAISS 索引 → 儲存。"""
    print("📂 掃描 data/ 資料夾...")
    docs = collect_documents()

    for i, d in enumerate(docs):
        date_str = d.metadata.get("date") or "無日期"
        print(f"  [{i}] ({date_str}) {d.page_content[:50]}...")

    print(f"\n⏳ 載入 Embedding 模型 {MODEL_NAME} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )
    print("✅ 模型載入完成")

    # 使用 LangChain FAISS 建立向量索引
    vectorstore = FAISS.from_documents(docs, embeddings)
    print(f"✅ FAISS 索引包含 {vectorstore.index.ntotal} 筆向量")

    # 儲存
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))

    print(f"\n🎉 索引已儲存至 {INDEX_DIR}")


if __name__ == "__main__":
    build_index()
