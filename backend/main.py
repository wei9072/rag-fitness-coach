import os
import sys
import uvicorn

# 確保 backend/ 為 Python 模組根目錄
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🚀 啟動 RAG 健身建議系統（後端）...")

    index_path = os.path.join("data", "qdrant_storage")
    if not os.path.exists(index_path):
        print("⚠️ 未偵測到 Qdrant 向量庫 (data/qdrant_storage)，開始自動建立索引...")
        from src.indexer import build_index
        build_index()

    print("🌐 啟動 FastAPI 後端 API (http://localhost:8000) ...")
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()
