from fastapi import APIRouter, HTTPException
from src.models.schemas import ChatRequest, ChatResponse
from src.services.intent_router import intent_router
from src.services.agent_workflow import agent_workflow

from src.services.memory_manager import memory_manager

api_router = APIRouter()

@api_router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    RAG 的調度器 (Orchestrator)
    接收使用者的請求，指派各種服務完成工作。
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="問題不可為空")

    try:
        # Step 1: 管理對話 Session (Memory)
        user_id = req.user_id
        session_id = req.session_id
        
        if not session_id:
            # 如果前端沒傳，開一個新階段
            session_id = memory_manager.create_session(user_id)
            
        # 提取最近對話紀錄 (預設抓 5 輪前文)
        recent_msgs = memory_manager.get_recent_messages(session_id)
        chat_history_str = memory_manager.format_history_for_prompt(recent_msgs)

        # Step 2: 由 Factory 取回相應的檢索策略實體 (Strategy) 與改寫後的關鍵字
        # 加上歷史紀錄賦予 Intent Router 結合上下文解析代名詞的能力
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

        # Step 4: 儲存對話進入長期記憶系統
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
        
        # 將系統錯誤轉換為給使用者的對話回覆
        error_msg = str(e)
        if "API Key" in error_msg or "驗證" in error_msg:
            answer = f"⚠️ 發生錯誤：{error_msg}\n\n👉 **解決方法：** 請確認您已經在設定中填寫了對應模型的 API Key。"
        else:
            answer = f"⚠️ 系統發生異常：{error_msg}\n\n👉 **解決方法：** 請檢查系統設定是否有誤，或稍後再試。"

        # 嘗試取得 session_id，若尚未建立則使用請求中提供的
        current_session = locals().get("session_id", req.session_id)
        
        # 為了讓前端能正常顯示對話並提示使用者，回傳 200 OK 和錯誤訊息
        return ChatResponse(
            answer=answer,
            session_id=current_session
        )


@api_router.get("/health")
async def health_check():
    """確認 FAISS 健康狀態"""
    from src.services.vector_service import vector_service  # 延遲載入避免循環依賴
    return {"status": "ok", "index_size": vector_service.count()}
