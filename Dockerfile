# 使用相對輕量的 Python 3.11 作為基底
FROM python:3.11-slim

# 設定容器工作目錄
WORKDIR /app

# 安裝系統層級的編譯工具 (如果 FAISS 需要)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 複製套件清單並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# [最佳化] 在建置映像檔時預先下載 HuggingFace 模型，避免每次啟動容器都要等幾分鐘下載 1GB 模型
RUN python -c "from langchain_community.embeddings import HuggingFaceBgeEmbeddings; HuggingFaceBgeEmbeddings(model_name='BAAI/bge-small-zh-v1.5')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base', max_length=512)"

# 複製所有專案原始碼與資料壓縮進映像檔
COPY . .

# 暴露 FastAPI (8000) 與 Streamlit (7860) 開放的連接埠
EXPOSE 8000 7860

# 預設啟動腳本 (會同時喚起 FastAPI 與 Streamlit)
CMD ["python", "main.py"]
