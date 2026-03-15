"""
實踐 OOP 策略模式 (Strategy Pattern) 與 SOLID 開放封閉原則 (OCP)
此模組定義了 RAG 檢索的不同策略，新增檢索邏輯時只需新增繼承 BaseRetriever 的類別，不需修改既有檢索管線。
"""
from abc import ABC, abstractmethod
from src.services.vector_service import vector_service

# 單例化 CrossEncoder，避免重複載入佔用記憶體
_reranker_model = None

def get_reranker():
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder
        print("⏳ 載入 Re-ranker 模型 (BAAI/bge-reranker-base)...")
        _reranker_model = CrossEncoder('BAAI/bge-reranker-base', max_length=512)
    return _reranker_model


class BaseRetriever(ABC):
    """
    抽象檢索策略基底類別 (Abstact Strategy)
    """
    @abstractmethod
    def retrieve(self, query: str = "", user_id: str = "default_user", limit_k: int | None = None) -> list[str]:
        pass

class SemanticRetriever(BaseRetriever):
    """
    語意檢索策略 (Concrete Strategy)
    導入「兩階段檢索 (Two-Stage Retrieval)」架構：
    - 階段一 (粗搜)：先使用 Qdrant 向量比對，並強制帶入 user_id 過濾，擴大召回率抓取前 10 筆相關資料。
    - 階段二 (精排)：透過 bge-reranker-base CrossEncoder 模型重新對 Query 與 Document 算分排序。
    最精準的分數優先回傳 (預設回傳 3 筆給 LLM)。
    """
    def __init__(self):
        # 確保模型在初次實例化時載入
        self.reranker = get_reranker()

    def retrieve(self, query: str = "", user_id: str = "default_user", limit_k: int | None = None) -> list[str]:
        # 階段一：粗搜 (Initial Retrieval - 放大範圍取 10 筆)
        initial_k = 10
        initial_docs = vector_service.search_semantic(query, user_id=user_id, k=initial_k)
        
        if not initial_docs:
            return []
            
        # 階段二：精排 (Reranking)
        # 將 query 與擷取出的 10 筆 doc 組成 pairs 讓 CrossEncoder 算分
        pairs = [[query, doc] for doc in initial_docs]
        scores = self.reranker.predict(pairs)
        
        # 將分數與文章配對，並依據分數由高到低重新排序 (分數愈高代表愈相關)
        scored_docs = list(zip(scores, initial_docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # 決定最後要回傳的數量 (若未指定，依照使用者需求預設回傳 Top 3)
        final_k = limit_k if limit_k else 3
        
        # 只提取排序後的內文字串回傳
        final_docs = [doc for score, doc in scored_docs[:final_k]]
        return final_docs

class TemporalRetriever(BaseRetriever):
    """
    時間意圖檢索策略 (Concrete Strategy)
    負責獲取最新 N 筆的紀錄，確保時序性資料的準確性。
    """
    def __init__(self, n_count: int):
        self.n_count = n_count

    def retrieve(self, query: str = "", user_id: str = "default_user", limit_k: int | None = None) -> list[str]:
        n = limit_k if limit_k else self.n_count
        return vector_service.get_top_n_by_date(n, user_id=user_id)

class AllRetriever(BaseRetriever):
    """
    全量檢索策略 (Concrete Strategy)
    提供使用者查詢整體訓練輪廓與次數的聚合型檢索。
    """
    def retrieve(self, query: str = "", user_id: str = "default_user", limit_k: int | None = None) -> list[str]:
        # 保護機制：避免全量檢索撐爆 LLM Context Window (413 錯誤)
        n = limit_k if limit_k else 30 
        return vector_service.get_top_n_by_date(n, user_id=user_id)
