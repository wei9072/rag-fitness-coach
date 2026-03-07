from fastapi import APIRouter, HTTPException
from src.models.schemas import ChatRequest, ChatResponse
from src.services.intent_router import intent_router
from src.services.agent_workflow import agent_workflow

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
        # Step 1: 由 Factory 取回相應的檢索策略實體 (Strategy) 與改寫後的關鍵字
        strategy, refined_query, intent_category = intent_router.route_query(
            req.question,
            api_key=req.api_key,
            is_paid=req.is_paid,
            llm_provider=req.llm_provider,
            model_name=req.model_name
        )

        # Step 2 & 3: 交由 Agentic Workflow 執行「檢索 -> 評估 -> (如果 NO 則改寫重搜) -> 生成」的閉環
        limit_k = 1 if req.strategy == "A" else None
        
        final_answer, relevant_docs = agent_workflow.run_agentic_rag(
            strategy=strategy,
            original_query=req.question,
            refined_query=refined_query,
            api_key=req.api_key,
            is_paid=req.is_paid,
            llm_provider=req.llm_provider,
            model_name=req.model_name,
            limit_k=limit_k,
            user_profile=req.user_profile,
            retrieval_strategy_name=req.strategy,
            intent_category=intent_category
        )

        # 返回結果
        return ChatResponse(
            answer=final_answer,
            rewritten_query=refined_query, # 紀錄第一次 Intent Router 的查詢，後續 Agent 改寫都在後台印出
            sources=relevant_docs
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        # 若任何底層拋出異常，一律由 API 層攔截並標上HTTP 500
        raise HTTPException(status_code=502, detail=f"系統異常：{str(e)}")


@api_router.get("/health")
async def health_check():
    """確認 FAISS 健康狀態"""
    from src.services.vector_service import vector_service  # 延遲載入避免循環依賴
    return {"status": "ok", "index_size": vector_service.count()}
