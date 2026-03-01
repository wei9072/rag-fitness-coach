from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from src.config.settings import LLM_MODEL, GROQ_API_KEY
from src.models.schemas import RouteDecision

class IntentRouter:
    """專責將用戶提問透過輕量化 LLM 來判斷意圖與路由"""
    def __init__(self):
        self._ROUTER_SYSTEM = (
            "你是一個健身紀錄檢索路由助手。"
            "分析使用者的問題，決定最適合的檢索策略：\n"
            "1. 'all': 使用者想要統計筆數、總金額、總次數或查看所有歷史紀錄時使用。\n"
            "2. 'temporal': 使用者詢問「最近」、「最新」、「上次」、「前N筆」紀錄時使用。\n"
            "3. 'semantic': 使用者詢問特定動作的表現、建議或具體內容時使用（語意向量搜尋）。\n\n"
            "如果是 'semantic'，請同時提供一個優化後的簡短關鍵詞。"
        )

    def route_query(self, question: str, api_key: str = None, is_paid: bool = False) -> RouteDecision:
        """根據客戶問題，決定檢索方式與關鍵字"""
        target_model = "llama-3.3-70b-versatile" if is_paid else LLM_MODEL
        actual_api_key = api_key if api_key else GROQ_API_KEY
        
        try:
            router_llm = ChatGroq(
                model=target_model,
                api_key=actual_api_key,
                temperature=0.0,
                max_tokens=200,
            )
            structured_router = router_llm.with_structured_output(RouteDecision)
            
            decision = structured_router.invoke([
                SystemMessage(content=self._ROUTER_SYSTEM),
                HumanMessage(content=question),
            ])
            print(f"  🧠 Intent Router 分析: {decision.intent} | N={decision.n_count} | Query='{decision.refined_query}'")
            return decision
        except Exception as e:
            print(f"  ⚠️ LLM 路由拋出異常，將退回語意搜尋：{e}")
            return RouteDecision(intent="semantic", n_count=5, refined_query=question)

# 出口單例
intent_router = IntentRouter()
