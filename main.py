import subprocess
import sys
import time
import os

def main():
    print("🚀 啟動 RAG 健身建議系統...")
    
    index_path = os.path.join("data", "faiss_index", "index.faiss")
    if not os.path.exists(index_path):
        print("⚠️ 未偵測到 FAISS 向量庫 (data/faiss_index)，開始自動建立索引...")
        from src.indexer import build_index
        build_index()
    
    # 啟動 FastAPI 後端
    print("啟動 API 伺服器 (FastAPI)...")
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    # 等待一下讓 API 先啟動
    time.sleep(3)
    
    # 啟動 Streamlit 前端
    print("啟動前端介面 (Streamlit)...")
    app_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "src/app.py", "--server.port", "7860", "--server.address", "0.0.0.0"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    try:
        # 保持主程式運行
        api_process.wait()
        app_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 關閉系統中...")
        api_process.terminate()
        app_process.terminate()
        api_process.wait()
        app_process.wait()
        print("✅ 系統已關閉")

if __name__ == "__main__":
    main()
