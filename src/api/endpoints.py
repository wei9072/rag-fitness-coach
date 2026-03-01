from fastapi import APIRouter, HTTPException
from src.models.schemas import ChatRequest, ChatResponse
from src.services.intent_router import intent_router
from src.services.llm_service import llm_service

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
        strategy, refined_query = intent_router.route_query(
            req.question,
            api_key=req.api_key,
            is_paid=req.is_paid,
            llm_provider=req.llm_provider,
            model_name=req.model_name
        )

        # Step 2: 執行策略 (符合 SRP 單一職責原則，不再於 API 層做 if-else 邏輯判斷)
        limit_k = 1 if req.strategy == "A" else None
        relevant_docs = strategy.retrieve(query=refined_query, limit_k=limit_k)

        # 沒找到資料，即時返回
        if not relevant_docs:
            return ChatResponse(
                answer="找不到相關的健身紀錄喔！",
                rewritten_query=refined_query,
                sources=[]
            )

        # Step 3: 交由 LLM 處理生成最後回答
        final_answer = llm_service.generate_reply(
            question=req.question, 
            context_chunks=relevant_docs,
            api_key=req.api_key,
            is_paid=req.is_paid,
            strategy=req.strategy,
            user_profile=req.user_profile,
            llm_provider=req.llm_provider,
            model_name=req.model_name
        )

        # 返回結果
        return ChatResponse(
            answer=final_answer,
            rewritten_query=refined_query,
            sources=relevant_docs
        )

    except Exception as e:
        # 若任何底層拋出異常，一律由 API 層攔截並標上HTTP 500
        raise HTTPException(status_code=502, detail=f"系統異常：{str(e)}")


@api_router.get("/health")
async def health_check():
    """確認 FAISS 健康狀態"""
    from src.services.vector_service import vector_service  # 延遲載入避免循環依賴
    return {"status": "ok", "index_size": vector_service.count()}
