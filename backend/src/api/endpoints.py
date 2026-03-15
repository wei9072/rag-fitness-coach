"""
Chat 端點 (Endpoints)
支援雙模式認證：帶 JWT Token = 個人化，不帶 Token = 匿名模式
"""
import uuid
from fastapi import APIRouter, HTTPException, Request
from src.models.schemas import ChatRequest, ChatResponse
from src.services.intent_router import intent_router
from src.services.agent_workflow import agent_workflow
from src.services.memory_manager import memory_manager
from src.services.auth_service import decode_access_token

api_router = APIRouter()


def _extract_user_from_request(request: Request) -> tuple[str, bool]:
    """
    從 HTTP Request Header 解析使用者身分。
    回傳 (user_id, is_authenticated)
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1]
        payload = decode_access_token(token)
        if payload:
            return payload["sub"], True
    
    # 匿名模式：產生臨時 guest ID
    return f"guest_{uuid.uuid4().hex[:8]}", False


@api_router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, request: Request):
    """
    RAG 的調度器 (Orchestrator)
    接收使用者的請求，指派各種服務完成工作。
    支援：已登入使用者（帶 Token）或匿名使用者。
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="問題不可為空")

    try:
        # Step 1: 解析使用者身分
        user_id, is_authenticated = _extract_user_from_request(request)
        session_id = req.session_id
        
        # 只有已登入使用者才建立持久 Session
        if is_authenticated:
            if not session_id:
                session_id = memory_manager.create_session(user_id)
            
            # 提取最近對話紀錄 (預設抓 5 輪前文)
            recent_msgs = memory_manager.get_recent_messages(session_id)
            chat_history_str = memory_manager.format_history_for_prompt(recent_msgs)
        else:
            # 匿名模式：建立臨時 session，不持久化
            if not session_id:
                session_id = f"anon_{uuid.uuid4().hex[:12]}"
            chat_history_str = ""

        # Step 2: 由 Factory 取回相應的檢索策略實體 (Strategy) 與改寫後的關鍵字
        strategy, refined_query, intent_category = intent_router.route_query(
            question=req.question,
            chat_history=chat_history_str,
            api_key=req.api_key,
            is_paid=req.is_paid,
            llm_provider=req.llm_provider,
            model_name=req.model_name
        )

        # Step 3: 交由 Agentic Workflow 執行「檢索 -> 評估 -> (如果 NO 則改寫重搜) -> 生成」的閉環
        final_answer, relevant_docs = agent_workflow.run_agentic_rag(
            strategy=strategy,
            original_query=req.question,
            refined_query=refined_query,
            chat_history=chat_history_str,
            user_id=user_id,
            api_key=req.api_key,
            is_paid=req.is_paid,
            llm_provider=req.llm_provider,
            model_name=req.model_name,
            user_profile=req.user_profile,
            intent_category=intent_category
        )

        # Step 4: 只有已登入使用者才儲存對話進入長期記憶系統
        if is_authenticated:
            memory_manager.add_message(session_id, "user", req.question)
            memory_manager.add_message(session_id, "assistant", final_answer)

        # 返回結果
        return ChatResponse(
            answer=final_answer,
            rewritten_query=refined_query,
            sources=relevant_docs,
            session_id=session_id
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"系統異常：{str(e)}")


@api_router.get("/health")
async def health_check():
    """確認系統健康狀態"""
    from src.services.vector_service import vector_service
    return {"status": "ok", "index_size": vector_service.count()}
