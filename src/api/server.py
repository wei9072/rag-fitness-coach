from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.api.endpoints import api_router
import os

def create_app() -> FastAPI:
    """工廠模式建立 FastAPI 實例"""
    app = FastAPI(title="個人化 AI 健身建議系統 (模組化架構)", version="2.0.0")
    
    # 掛載 CORS，允許跨域
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 註冊所有的 API Routes
    app.include_router(api_router, prefix="/api")

    # Serve React Frontend
    static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
    if os.path.exists(static_dir):
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app

# Server 啟動進入點
app = create_app()
