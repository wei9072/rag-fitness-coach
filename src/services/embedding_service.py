from langchain_huggingface import HuggingFaceEmbeddings
from src.config.settings import MODEL_NAME
import torch

class EmbeddingService:
    """單例模式，確保 bge-small 模型只被載入一次。"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"⏳ 載入 Embedding 模型 ({device.upper()})...")
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance.model = HuggingFaceEmbeddings(
                model_name=MODEL_NAME,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
            print("✅ Embedding 模型載入完成")
        return cls._instance

    def get_embeddings(self):
        """取得初始化後的 Embedding 模型實例。"""
        return self.model

# 直接拋出全域單例物件
embedding_service = EmbeddingService()
