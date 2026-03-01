from fastapi import APIRouter, HTTPException
from src.models.schemas import ChatRequest, ChatResponse
from src.services.intent_router import intent_router
from src.services.vector_service import vector_service
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
        # Step 1: 判定語意路由，取得改寫關鍵字
        decision = intent_router.route_query(
            req.question,
            api_key=req.api_key,
            is_paid=req.is_paid
        )

        # Step 2: 依造判斷決定取用哪種檢索邏輯
        # 實作檢索策略：若為 Strategy A，強制限制所有數量為 1
        limit_k = 1 if req.strategy == "A" else None
        
        if decision.intent == "all":
            # 保護機制：現在資料庫可能多達數千個 Chunk (如 PDF)，
            # 若意圖為全量統計(all)，為避免撐爆 LLM Context Window (413 錯誤)，
            # 強制限制最多只取最新 30 筆有日期的核心訓練紀錄
            n = limit_k if limit_k else 30
            relevant_docs = vector_service.get_top_n_by_date(n)
        elif decision.intent == "temporal":
            n = limit_k if limit_k else decision.n_count
            relevant_docs = vector_service.get_top_n_by_date(n)
        else:
            # 語意搜尋時，同樣也設定一個保護上限 (TOP_K 預設5)
            # 因為 vector_service 預設是吃 TOP_K，這裏我們直接手動切片
            search_res = vector_service.search_semantic(decision.refined_query)
            relevant_docs = search_res[:limit_k] if limit_k else search_res

        # 沒找到資料，即時返回
        if not relevant_docs:
            return ChatResponse(
                answer="找不到相關的健身紀錄喔！",
                rewritten_query=decision.refined_query,
                sources=[]
            )

        # Step 3: 交由 LLM 處理生成最後回答
        final_answer = llm_service.generate_reply(
            question=req.question, 
            context_chunks=relevant_docs,
            api_key=req.api_key,
            is_paid=req.is_paid,
            strategy=req.strategy
        )

        # 返回結果
        return ChatResponse(
            answer=final_answer,
            rewritten_query=decision.refined_query,
            sources=relevant_docs
        )

    except Exception as e:
        # 若任何底層拋出異常，一律由 API 層攔截並標上HTTP 500
        raise HTTPException(status_code=502, detail=f"系統異常：{str(e)}")


@api_router.get("/health")
async def health_check():
    """確認 FAISS 健康狀態"""
    return {"status": "ok", "index_size": vector_service.count()}
