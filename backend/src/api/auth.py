"""
認證路由 (Auth Router)
提供 /api/auth/register, /api/auth/login, /api/auth/me
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from src.models.schemas import RegisterRequest, LoginRequest, TokenResponse, UserInfo
from src.services.auth_service import (
    register_user, authenticate_user,
    create_access_token, decode_access_token
)

auth_router = APIRouter()
security = HTTPBearer()


@auth_router.post("/register", response_model=TokenResponse)
async def register(req: RegisterRequest):
    """註冊新帳號，成功後直接回傳 JWT Token"""
    if len(req.username) < 2:
        raise HTTPException(status_code=400, detail="帳號至少需要 2 個字元")
    if len(req.password) < 4:
        raise HTTPException(status_code=400, detail="密碼至少需要 4 個字元")
    
    try:
        user = register_user(req.username, req.password)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    
    token = create_access_token(user["user_id"], user["username"])
    return TokenResponse(
        access_token=token,
        user_id=user["user_id"],
        username=user["username"]
    )


@auth_router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    """登入取得 JWT Token"""
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="帳號或密碼錯誤")
    
    token = create_access_token(user["user_id"], user["username"])
    return TokenResponse(
        access_token=token,
        user_id=user["user_id"],
        username=user["username"]
    )


@auth_router.get("/me", response_model=UserInfo)
async def get_me(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """取得目前登入使用者的資訊"""
    payload = decode_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Token 過期或無效")
    
    return UserInfo(
        user_id=payload["sub"],
        username=payload["username"]
    )
