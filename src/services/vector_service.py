from langchain_community.vectorstores import FAISS
from src.config.settings import INDEX_DIR, TOP_K
from src.services.embedding_service import embedding_service

class VectorService:
    """負責與 FAISS 的低階互動，管理向量庫單例。"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print("⏳ 載入 FAISS 向量庫...")
            cls._instance = super(VectorService, cls).__new__(cls)
            
            if not INDEX_DIR.exists():
                raise FileNotFoundError(
                    f"找不到 FAISS 索引：{INDEX_DIR}，請先執行 python src/indexer.py"
                )
                
            embeddings = embedding_service.get_embeddings()
            cls._instance.vectorstore = FAISS.load_local(
                str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
            )
            # 快取 Langchain Retriever 實體
            cls._instance.retriever = cls._instance.vectorstore.as_retriever(search_kwargs={"k": TOP_K})
            print(f"✅ FAISS 載入完成（{cls._instance.vectorstore.index.ntotal} 筆向量）")
            
        return cls._instance

    def search_semantic(self, query: str) -> list[str]:
        """語意向量檢索"""
        # 動態抓取 TOP_K，避免 Singleton 初始化後參數寫死
        docs = self.vectorstore.similarity_search(query, k=TOP_K)
        return [d.page_content for d in docs]

    def get_all_sorted_by_date(self) -> list[str]:
        """讀取全量並依據時間降序"""
        all_docs = list(self.vectorstore.docstore._dict.values())
        dated = [
            (d.metadata.get("date") or "", d.page_content)
            for d in all_docs
        ]
        dated.sort(key=lambda x: x[0] if x[0] else "0000-00-00", reverse=True)
        return [text for _, text in dated]

    def get_top_n_by_date(self, n: int) -> list[str]:
        """取得最新的 N 筆紀錄"""
        dated_texts = self.get_all_sorted_by_date()
        print(f"  📅 時間意圖檢索：取最新 {n} 筆")
        return dated_texts[:n]

    def count(self) -> int:
        return self.vectorstore.index.ntotal

# 直接拋出全域單例物件
vector_service = VectorService()
