"""
實踐 OOP 策略模式 (Strategy Pattern) 與 SOLID 開放封閉原則 (OCP)
此模組定義了 RAG 檢索的不同策略，新增檢索邏輯時只需新增繼承 BaseRetriever 的類別，不需修改既有檢索管線。
"""
from abc import ABC, abstractmethod
from src.services.vector_service import vector_service

class BaseRetriever(ABC):
    """
    抽象檢索策略基底類別 (Abstact Strategy)
    """
    @abstractmethod
    def retrieve(self, query: str = "", limit_k: int | None = None) -> list[str]:
        pass

class SemanticRetriever(BaseRetriever):
    """
    語意檢索策略 (Concrete Strategy)
    負責呼叫 FAISS 進行語意識別與向量相似度比對。
    """
    def retrieve(self, query: str = "", limit_k: int | None = None) -> list[str]:
        res = vector_service.search_semantic(query)
        # 由於 vector_service 預設是吃 TOP_K，這裏我們手動切片套用 limit_k (策略 A 保護機制)
        return res[:limit_k] if limit_k else res

class TemporalRetriever(BaseRetriever):
    """
    時間意圖檢索策略 (Concrete Strategy)
    負責獲取最新 N 筆的紀錄，確保時序性資料的準確性。
    """
    def __init__(self, n_count: int):
        self.n_count = n_count

    def retrieve(self, query: str = "", limit_k: int | None = None) -> list[str]:
        n = limit_k if limit_k else self.n_count
        return vector_service.get_top_n_by_date(n)

class AllRetriever(BaseRetriever):
    """
    全量檢索策略 (Concrete Strategy)
    提供使用者查詢整體訓練輪廓與次數的聚合型檢索。
    """
    def retrieve(self, query: str = "", limit_k: int | None = None) -> list[str]:
        # 保護機制：避免全量檢索撐爆 LLM Context Window (413 錯誤)
        n = limit_k if limit_k else 30 
        return vector_service.get_top_n_by_date(n)
