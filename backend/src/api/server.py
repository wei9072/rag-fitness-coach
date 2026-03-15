from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.endpoints import api_router
from src.api.auth import auth_router

def create_app() -> FastAPI:
    """工廠模式建立 FastAPI 實例"""
    app = FastAPI(title="個人化 AI 健身建議系統 (前後端分離架構)", version="3.0.0")
    
    # 掛載 CORS — 明確允許前端 Vite 開發伺服器
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",   # Vite dev server
            "http://localhost:3000",   # 備用
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 註冊所有的 API Routes
    app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])
    app.include_router(api_router, prefix="/api", tags=["Chat"])

    return app

# Server 啟動進入點
app = create_app()
