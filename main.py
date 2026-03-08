import os
import uvicorn

def main():
    print("🚀 啟動 RAG 健身建議系統...")

    index_path = os.path.join("data", "faiss_index", "index.faiss")
    if not os.path.exists(index_path):
        print("⚠️ 未偵測到 FAISS 向量庫 (data/faiss_index)，開始自動建立索引...")
        from src.indexer import build_index
        build_index()

    print("🌐 啟動 FastAPI + React 前端 (http://localhost:7860) ...")
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=7860, log_level="info")

if __name__ == "__main__":
    main()
