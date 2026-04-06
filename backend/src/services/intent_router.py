from langchain_core.messages import SystemMessage, HumanMessage
from src.models.schemas import RouteDecision
from src.services.retrieval_strategy import BaseRetriever, SemanticRetriever, TemporalRetriever, AllRetriever
from src.services.llm_factory import get_llm, resolve_model_and_key

class IntentRouter:
    """
    實踐 SOLID 單一職責原則 (SRP) 與 Factory Pattern
    負責將用戶提問透過輕量化 LLM 來判斷意圖，並回傳對應的檢索實體 (Strategy)
    """
    def __init__(self):
        self._ROUTER_SYSTEM = (
            "你是一個健身紀錄檢索路由助手。\n"
            "【步驟 1：判定大分類 (intent_category)】\n"
            "如果是詢問過去的訓練數據或紀錄（描述性問答），請設為 'QA_INTENT'。\n"
            "如果是要求安排新菜單、給予未來訓練建議等生成性任務，請設為 'PLANNING_INTENT'。\n"
            "如果是詢問數值統計或聚合查詢（例如：最重多少、練了幾次、總組數、平均重量、PR 紀錄、訓練量排行），請設為 'AGGREGATE_INTENT'。\n\n"
            "【步驟 2：判定檢索策略 (intent)】\n"
            "若 intent_category 為 'AGGREGATE_INTENT'，intent 固定填 'all'。\n"
            "否則，分析使用者的問題，決定最適合的檢索策略：\n"
            "1. 'all': 使用者想要統計筆數、總金額、總次數或查看所有歷史紀錄時使用。\n"
            "2. 'temporal': 使用者詢問「最近」、「最新」、「上次」、「前N筆」紀錄時使用。\n"
            "3. 'semantic': 使用者詢問特定動作的表現、建議或具體內容時使用（語意向量搜尋）。\n\n"
            "如果是 'semantic'，請同時提供一個優化後的簡短關鍵詞。"
        )

    def route_query(
        self,
        question: str,
        chat_history: str = "",
        api_key: str = None,
        is_paid: bool = False,
        llm_provider: str = "groq",
        model_name: str = None
    ) -> tuple[BaseRetriever, str, str]:
        """
        根據客戶問題與近期對話紀錄，決定檢索方式與關鍵字
        回傳 (Retriever策略物件, 提煉後的搜尋字串, 意圖大類)
        """
        model_name, actual_api_key = resolve_model_and_key(llm_provider, model_name, api_key, is_paid)
        
        try:
            # 使用 DIP Factory 取得實例
            router_llm = get_llm(
                provider=llm_provider,
                model_name=model_name,
                api_key=actual_api_key,
                temperature=0.0
            )
            structured_router = router_llm.with_structured_output(RouteDecision)
            
            system_prompt = self._ROUTER_SYSTEM
            if chat_history:
                system_prompt += f"\n\n【近期對話語境 (Context)】\n以下是最近的對話紀錄，請參考它來解析使用者的代名詞或省略的詞彙（例如「那深蹲呢？」代表接續查深蹲）：\n{chat_history}"

            decision = structured_router.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=question),
            ])
            print(f"  🧠 Intent Router 分析: {decision.intent_category} -> {decision.intent} | N={decision.n_count} | Query='{decision.refined_query}'")
            
            # 👇 Factory 邏輯：根據分類返回對應的 Strategy，後續調用方不需理會細節
            if decision.intent == "all":
                return AllRetriever(), decision.refined_query, decision.intent_category
            elif decision.intent == "temporal":
                return TemporalRetriever(decision.n_count), decision.refined_query, decision.intent_category
            else:
                return SemanticRetriever(), decision.refined_query, decision.intent_category

        except Exception as e:
            print(f"  ⚠️ LLM 路由拋出異常，將退回語意搜尋：{e}")
            return SemanticRetriever(), question, "QA_INTENT"

# 出口單例
intent_router = IntentRouter()
