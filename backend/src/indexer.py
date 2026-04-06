"""
indexer.py — 讀取 data/ 資料夾的訓練資料：
1. 結構化解析後存入 SQLite training_logs（供 Text-to-SQL 聚合查詢）
2. 語意切塊後向量化存入 Qdrant 本地索引（供語意檢索）
"""

import json
import pathlib
import re

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
import time
import torch

# ── Embedding 模型與切塊器 ─────────────────────────────────
from src.config.settings import BASE_DIR, INDEX_DIR, MODEL_NAME, LLM_MODEL, GROQ_API_KEY

# 本地開發時資料在根目錄 data/，Docker 建置時會 COPY 到 backend/data/
_ROOT_DATA_DIR = BASE_DIR.parent / "data"
_BACKEND_DATA_DIR = BASE_DIR / "data"
DATA_DIR = _ROOT_DATA_DIR if _ROOT_DATA_DIR.exists() and any(_ROOT_DATA_DIR.glob("*.txt")) else _BACKEND_DATA_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⏳ 載入 SemanticChunker 使用的 Embedding 模型 ({device.upper()})...")
_chunking_embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)
semantic_chunker = SemanticChunker(
    _chunking_embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=60
)
print("✅ SemanticChunker 初始化完成")


# ═════════════════════════════════════════════════════════════
# 日期解析
# ═════════════════════════════════════════════════════════════
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


# ═════════════════════════════════════════════════════════════
# TXT 結構化解析器（供 SQLite training_logs）
# ═════════════════════════════════════════════════════════════
_SESSION_HEADER_RE = re.compile(r"^\s*(\d{4})\s+\d{4}[練]?\s*$")
_REPS_SETS_RE = re.compile(r"(\d+)\*(\d+)")
_WEIGHT_KG_RE = re.compile(r"(\d+(?:\.\d+)?)\s*kg")


def parse_txt_to_structured_records(
    path_or_content: pathlib.Path | str,
    user_id: str = "default_user",
    default_year: str = "2026",
    source: str = "",
) -> list[dict]:
    """解析訓練 TXT 為結構化訓練紀錄 list[dict]，供寫入 SQLite。
    可接受檔案路徑或原始文字內容。"""
    if isinstance(path_or_content, pathlib.Path):
        content = path_or_content.read_text(encoding="utf-8")
        source = source or path_or_content.name
    else:
        content = path_or_content
    records: list[dict] = []
    current_date = None

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue

        # 日期標頭偵測：MMDD HHmm練
        header_match = _SESSION_HEADER_RE.match(line)
        if header_match:
            mmdd = header_match.group(1)
            month, day = mmdd[:2], mmdd[2:]
            if 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
                current_date = f"{default_year}-{month}-{day}"
            continue

        if not current_date:
            continue

        # 跳過沒有 reps*sets 的行（如「跑步 4公里」）
        if not _REPS_SETS_RE.search(line):
            continue

        parsed = _parse_exercise_line(line, current_date, user_id, source)
        records.extend(parsed)

    return records


def _parse_exercise_line(
    line: str, date: str, user_id: str, source: str
) -> list[dict]:
    """解析單行運動紀錄，回傳一或多筆結構化 dict。"""

    # ── Step 1: 切割動作名稱 vs 組數資料 ──
    # 先移除括號備註（如「（空槓11.3）」），避免干擾 marker 偵測
    line_clean = re.sub(r"[（(][^）)]*[）)]", "", line).strip()

    # 找到第一個可能是組數資料起點的位置
    first_marker = re.search(
        r"(空槓|自重|(?:單邊\s*)?\d+(?:\.\d+)?\s*kg\s+\d+\*\d+|\d+\*\d+)",
        line_clean
    )
    if not first_marker:
        return []

    # 對應回原始 line 的位置：用 clean line 的 exercise name 長度來定位
    marker_start = first_marker.start()
    before_marker = line_clean[:marker_start].rstrip()

    # 嘗試把 weight 前綴也納入 set data
    weight_prefix = re.search(
        r"(?:空槓(?:\s*\d+(?:\.\d+)?)?\s*|自重\s*|(?:單邊\s*)?\d+(?:\.\d+)?\s*kg\s*)$",
        before_marker
    )
    if weight_prefix:
        split_pos = weight_prefix.start()
    else:
        split_pos = marker_start

    exercise_name = line_clean[:split_pos].strip()
    # 清理尾部「單邊」
    exercise_name = re.sub(r"\s*單邊\s*$", "", exercise_name).strip()

    if not exercise_name:
        return []

    # ── Step 2: 解析各組資料（使用清理後的 line）──
    sets_str = line_clean[split_pos:].strip()
    groups = sets_str.split("、")

    results: list[dict] = []
    last_weight: float | None = None

    for group in groups:
        group = group.strip()
        rs_match = _REPS_SETS_RE.search(group)
        if not rs_match:
            continue

        reps = int(rs_match.group(1))
        sets_count = int(rs_match.group(2))
        before = group[: rs_match.start()].strip()

        weight: float | None = None
        note: str | None = None

        if "空槓" in before:
            weight = 20.0
            note = "空槓"
        elif "自重" in before:
            weight = None
            note = "自重"
        else:
            wm = _WEIGHT_KG_RE.search(before)
            if wm:
                weight = float(wm.group(1))
                if "單邊" in before:
                    note = "單邊"
            else:
                # 沒有重量標示 → 繼承上一組重量（carry-over）
                weight = last_weight

        if weight is not None:
            last_weight = weight

        results.append({
            "user_id": user_id,
            "date": date,
            "exercise": exercise_name,
            "weight_kg": weight,
            "reps": reps,
            "sets": sets_count,
            "note": note,
            "source": source,
        })

    return results


def parse_json_to_structured_records(
    path: pathlib.Path,
    user_id: str = "default_user",
) -> list[dict]:
    """解析 JSON 訓練紀錄為結構化 list[dict]。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records: list[dict] = []
    for entry in data:
        if "sets" not in entry or "exercise" not in entry:
            continue
        for s in entry["sets"]:
            records.append({
                "user_id": user_id,
                "date": entry.get("date", ""),
                "exercise": entry["exercise"],
                "weight_kg": s.get("weight_kg"),
                "reps": s.get("reps", 0),
                "sets": 1,
                "note": entry.get("notes"),
                "source": path.name,
            })
    return records


# ═════════════════════════════════════════════════════════════
# 檔案載入 → LangChain Document（供 Qdrant 向量索引）
# ═════════════════════════════════════════════════════════════

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
            metadata={"date": date, "source": path.name, "user_id": "default_user", "data_type": "personal"},
        ))
    return docs


def load_txt_documents(path: pathlib.Path) -> list[Document]:
    """讀取 TXT 檔，使用 SemanticChunker 進行語意切塊。"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    docs = semantic_chunker.create_documents([content])

    for doc in docs:
        date = _parse_date(doc.page_content)
        doc.metadata = {"date": date, "source": path.name, "user_id": "default_user", "data_type": "personal"}

    return docs


def load_pdf_documents(path: pathlib.Path) -> list[Document]:
    """讀取 PDF 檔，使用 SemanticChunker 進行語意切塊。標記為共享知識庫。"""
    try:
        loader = PyPDFLoader(str(path))
        pages = loader.load()
    except Exception as e:
        print(f"  ⚠️ 無法解析 PDF {path.name}: {e}")
        return []

    content = "\n\n".join([page.page_content for page in pages]).strip()

    if not content:
        return []

    docs = semantic_chunker.create_documents([content])

    for doc in docs:
        date = _parse_date(doc.page_content)
        doc.metadata = {"date": date, "source": path.name, "user_id": "__shared__", "data_type": "knowledge_base"}

    return docs


# ═════════════════════════════════════════════════════════════
# 統一掃描（Qdrant 用）
# ═════════════════════════════════════════════════════════════
LOADERS = {
    ".json": load_json_documents,
    ".txt": load_txt_documents,
    ".pdf": load_pdf_documents
}


def collect_documents() -> list[Document]:
    """掃描 data/ 下所有支援的檔案，回傳 Document 列表。"""
    all_docs: list[Document] = []

    for ext, loader in LOADERS.items():
        for path in sorted(DATA_DIR.glob(f"*{ext}")):
            docs = loader(path)
            print(f"  📄 {path.name}  →  {len(docs)} 個區塊")
            all_docs.extend(docs)

    if not all_docs:
        print(f"  ⚠️ 在 {DATA_DIR} 中找不到任何支援的檔案 (.txt, .json, .pdf)")
        return []

    with_date = sum(1 for d in all_docs if d.metadata.get("date"))
    print(f"\n✅ 合計 {len(all_docs)} 個區塊，其中 {with_date} 個有日期 Metadata")
    return all_docs


# ═════════════════════════════════════════════════════════════
# 結構化資料寫入 SQLite
# ═════════════════════════════════════════════════════════════

def _build_structured_training_logs():
    """掃描 data/ 中的 TXT/JSON，解析結構化欄位寫入 SQLite training_logs。"""
    from src.db.database import insert_training_logs, clear_training_logs_by_source

    total = 0

    for txt_path in sorted(DATA_DIR.glob("*.txt")):
        clear_training_logs_by_source(txt_path.name)
        records = parse_txt_to_structured_records(txt_path)
        if records:
            count = insert_training_logs(records)
            print(f"  📊 {txt_path.name} → {count} 筆結構化紀錄")
            total += count

    for json_path in sorted(DATA_DIR.glob("*.json")):
        clear_training_logs_by_source(json_path.name)
        records = parse_json_to_structured_records(json_path)
        if records:
            count = insert_training_logs(records)
            print(f"  📊 {json_path.name} → {count} 筆結構化紀錄")
            total += count

    print(f"✅ 共寫入 {total} 筆結構化訓練紀錄至 SQLite training_logs")


# ═════════════════════════════════════════════════════════════
# 增量 Ingestion（使用者上傳用）
# ═════════════════════════════════════════════════════════════

def ingest_user_training_text(
    content: str,
    user_id: str,
    source_name: str = "user_upload",
) -> dict:
    """
    增量寫入使用者訓練資料（不重建整個 Qdrant collection）。
    1. 結構化解析 → SQLite training_logs
    2. 語意切塊 → Qdrant add_documents()
    回傳 {"chunks_added": int, "sql_records": int}
    """
    from src.db.database import insert_training_logs
    from src.services.vector_service import vector_service

    # Step 1: 結構化寫入 SQLite
    records = parse_txt_to_structured_records(
        content, user_id=user_id, source=source_name
    )
    sql_count = 0
    if records:
        sql_count = insert_training_logs(records)

    # Step 2: 語意切塊 → Qdrant 增量寫入
    if not content.strip():
        return {"chunks_added": 0, "sql_records": sql_count}

    docs = semantic_chunker.create_documents([content])
    for doc in docs:
        date = _parse_date(doc.page_content)
        doc.metadata = {
            "date": date,
            "source": source_name,
            "user_id": user_id,
            "data_type": "personal",
        }

    vector_service.vectorstore.add_documents(docs)
    print(f"  📥 增量寫入: {len(docs)} chunks + {sql_count} SQL records (user: {user_id})")

    return {"chunks_added": len(docs), "sql_records": sql_count}


def delete_user_data(user_id: str) -> None:
    """刪除使用者在 Qdrant + SQLite 的所有個人資料。"""
    from qdrant_client.http import models as qmodels
    from src.services.vector_service import vector_service
    from src.db.database import get_db_connection

    # Qdrant: 刪除該使用者的所有向量
    vector_service.client.delete(
        collection_name=vector_service.collection_name,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(must=[
                qmodels.FieldCondition(
                    key="metadata.user_id",
                    match=qmodels.MatchValue(value=user_id)
                )
            ])
        )
    )

    # SQLite: 刪除該使用者的訓練紀錄
    with get_db_connection() as conn:
        conn.execute("DELETE FROM training_logs WHERE user_id = ?", (user_id,))
        conn.commit()

    print(f"  🗑️ 已刪除使用者 {user_id} 的所有個人資料")


# ═════════════════════════════════════════════════════════════
# 主流程
# ═════════════════════════════════════════════════════════════

def build_index():
    """讀取 → SemanticChunker(語意切塊) → Qdrant 索引 + 結構化解析 → SQLite。"""
    QDRANT_DIR = BASE_DIR / "data" / "qdrant_storage"

    index_exists = QDRANT_DIR.exists() and any(QDRANT_DIR.iterdir())

    print("📂 掃描 data/ 資料夾...")
    docs = collect_documents()

    if not docs:
        if index_exists:
            print(f"✨ 偵測到現有的向量資料庫於 {QDRANT_DIR}，且無新檔案需索引，跳過建立流程。")
        else:
            print("❌ 錯誤：找不到來源檔案且無現有索引，無法提供服務。")
        # 即使沒有新文件，也嘗試建立結構化資料
        print("\n⏳ 解析結構化訓練數據寫入 SQLite...")
        _build_structured_training_logs()
        return

    for i, d in enumerate(docs):
        date_str = d.metadata.get("date") or "無日期"
        if i < 5 or i >= len(docs) - 1:
            print(f"  [{i}] ({date_str}) {d.page_content[:50]}...")

    print(f"\n⏳ 準備建立 Qdrant 本地索引...")

    from langchain_qdrant import QdrantVectorStore

    vectorstore = QdrantVectorStore.from_documents(
        docs,
        _chunking_embeddings,
        path=str(QDRANT_DIR),
        collection_name="fitness_logs",
        force_recreate=True
    )

    print(f"\n🎉 Qdrant 索引已儲存至 {QDRANT_DIR}")

    # 同步建立結構化訓練紀錄
    print("\n⏳ 解析結構化訓練數據寫入 SQLite...")
    _build_structured_training_logs()


if __name__ == "__main__":
    build_index()
