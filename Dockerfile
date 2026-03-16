# 階段一：構建前端 (Node.js)
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install

COPY frontend/ .
RUN npm run build


# 階段二：構建後端與最終映像檔 (Python)
FROM python:3.11-slim

# 設定容器工作目錄為 backend 所在的層級
WORKDIR /app

# 強制 Python 立刻輸出 Log
ENV PYTHONUNBUFFERED=1

COPY backend/requirements.txt backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# 預先下載常用模型 (利用 Docker Layer Cache)
RUN python -c "from langchain_community.embeddings import HuggingFaceBgeEmbeddings; HuggingFaceBgeEmbeddings(model_name='BAAI/bge-small-zh-v1.5', encode_kwargs={'normalize_embeddings': True})"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base', max_length=512)"

# 複製後端程式碼與資料 (確保包含了 backend/src, backend/data 等等)
COPY backend/ backend/

# 將前端打包好的產物，複製到後端能讀到的 static 資料夾
COPY --from=frontend-builder /app/frontend/dist backend/static

# [預設動作] 在建置期間建立 Qdrant 索引 (若原本已經存在則視情況不重建，這裡預設先執行)
# 如果希望每次啟動時才建立，可以把這行拿掉，由 main.py 自動偵測執行
RUN cd backend && python -m src.indexer

# 暴露 FastAPI 連接埠（同時 Serve React SPA 前端）
EXPOSE 7860

# 因為從 /app 作為 WORKDIR，我們需要切換環境變數或確保直接執行 backend/main.py (注意 main.py 有自己的 uvicorn 啟動路徑)
ENV PYTHONPATH=/app/backend
WORKDIR /app/backend

# 把 backend 內部的 port 開放在 7860 以符合 HF Space 需求
CMD ["python", "main.py"]
