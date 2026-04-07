from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.endpoints import api_router
from src.api.auth import auth_router

def create_app() -> FastAPI:
    """工廠模式建立 FastAPI 實例"""
    app = FastAPI(title="個人化 AI 健身建議系統 (前後端分離架構)", version="8.0.0")
    
    # 掛載 CORS — 明確允許前端 Vite 開發伺服器
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 註冊所有的 API Routes
    app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])
    app.include_router(api_router, prefix="/api", tags=["Chat"])

    # SPA (Single Page Application) 靜態檔案服務
    # 如果 frontend 已經 build 好並放在 backend/static 目錄下
    import os
    from fastapi.staticfiles import StaticFiles
    from starlette.responses import FileResponse
    
    # 這裡的寫法假設會在容器的 /app/static 或本機的 backend/static 底下放打包後的前端
    static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static")
    
    # 先判斷 static_dir 是否存在，避免在本機純開發後端時報錯
    if os.path.isdir(static_dir):
        app.mount("/assets", StaticFiles(directory=os.path.join(static_dir, "assets")), name="assets")
        
        # 捕捉所有非 /api 開頭的 Path，回傳 index.html 交給 React Router 處理
        @app.get("/{full_path:path}")
        async def catch_all(full_path: str):
            if full_path.startswith("api"):
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="API route not found")
            return FileResponse(os.path.join(static_dir, "index.html"))

    return app

# Server 啟動進入點
app = create_app()
