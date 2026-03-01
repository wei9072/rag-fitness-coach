from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.endpoints import api_router

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

    return app

# Server 啟動進入點
app = create_app()
