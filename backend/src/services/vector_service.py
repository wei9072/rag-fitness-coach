from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config.settings import BASE_DIR, TOP_K
from src.services.embedding_service import embedding_service

QDRANT_DIR = BASE_DIR / "data" / "qdrant_storage"

class VectorService:
    """負責與 Qdrant 的低階互動，實踐多租戶 (Multi-Tenancy) 物理隔離。"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print("⏳ 載入 Qdrant 向量庫 (Local Storage)...")
            cls._instance = super(VectorService, cls).__new__(cls)
            
            if not QDRANT_DIR.exists():
                raise FileNotFoundError(
                    f"找不到 Qdrant 儲存夾：{QDRANT_DIR}，請先執行 python -m src.indexer"
                )
                
            embeddings = embedding_service.get_embeddings()
            
            # 建立 Client 與 VectorStore 綁定
            cls._instance.client = QdrantClient(path=str(QDRANT_DIR))
            cls._instance.collection_name = "fitness_logs"
            cls._instance.vectorstore = QdrantVectorStore(
                client=cls._instance.client,
                collection_name=cls._instance.collection_name,
                embedding=embeddings
            )
            
            # 計算目前集合中的總點數 (Qdrant 特有 API)
            try:
                collection_info = cls._instance.client.get_collection(cls._instance.collection_name)
                ntotal = collection_info.points_count
            except Exception:
                ntotal = 0
            print(f"✅ Qdrant 載入完成（{ntotal} 筆向量）")
            
        return cls._instance

    def _get_user_filter(self, user_id: str) -> models.Filter:
        """建立針對特定使用者的 Metadata Filter"""
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.user_id",
                    match=models.MatchValue(value=user_id)
                )
            ]
        )

    def search_semantic(self, query: str, user_id: str, k: int | None = None) -> list[str]:
        """語意向量檢索 (強制帶入 user_id 隔離)"""
        search_k = k if k else TOP_K
        user_filter = self._get_user_filter(user_id)
        
        docs = self.vectorstore.similarity_search(
            query=query, 
            k=search_k,
            filter=user_filter
        )
        return [d.page_content for d in docs]

    def get_all_sorted_by_date(self, user_id: str) -> list[str]:
        """讀取特定 user_id 的全量資料並依據時間降序"""
        user_filter = self._get_user_filter(user_id)
        
        # 使用 Qdrant client 原生 scroll API 抓取特定 user 所有的點
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=user_filter,
            limit=10000, # 設定一個足夠大的上限
            with_payload=True
        )
        
        dated = []
        for point in records:
            payload = point.payload or {}
            metadata = payload.get("metadata", {})
            date = metadata.get("date", "")
            text = payload.get("page_content", "")
            dated.append((date, text))
            
        dated.sort(key=lambda x: x[0] if x[0] else "0000-00-00", reverse=True)
        return [text for _, text in dated]

    def get_top_n_by_date(self, n: int, user_id: str) -> list[str]:
        """取得特定使用者最新的 N 筆紀錄"""
        dated_texts = self.get_all_sorted_by_date(user_id)
        print(f"  📅 時間意圖檢索 (User: {user_id})：取最新 {n} 筆")
        return dated_texts[:n]

    def count(self) -> int:
        try:
            return self.client.get_collection(self.collection_name).points_count
        except:
            return 0

# 直接拋出全域單例物件
vector_service = VectorService()
