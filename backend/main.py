import os
import sys
import uvicorn

# 確保 backend/ 為 Python 模組根目錄
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🚀 啟動 RAG 健身建議系統（後端）...")

    index_path = os.path.join("data", "qdrant_storage")
    qdrant_exists = os.path.exists(index_path)

    # 檢查 SQLite training_logs 是否有資料
    has_training_logs = False
    try:
        from src.db.database import execute_safe_select
        rows = execute_safe_select("SELECT COUNT(*) as cnt FROM training_logs", {})
        has_training_logs = rows[0]["cnt"] > 0
    except Exception:
        pass

    if not qdrant_exists or not has_training_logs:
        reason = []
        if not qdrant_exists:
            reason.append("Qdrant 向量庫")
        if not has_training_logs:
            reason.append("SQLite training_logs")
        print(f"⚠️ 未偵測到 {' 和 '.join(reason)}，開始自動建立索引...")
        from src.indexer import build_index
        build_index()

    port = int(os.environ.get("PORT", 7860))
    print(f"🌐 啟動 FastAPI 後端 API (http://0.0.0.0:{port}) ...")
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=port, log_level="info")

if __name__ == "__main__":
    main()
