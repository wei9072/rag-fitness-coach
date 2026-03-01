from langchain_core.messages import SystemMessage, HumanMessage
from src.config.settings import LLM_MODEL, GROQ_API_KEY
from src.services.llm_factory import get_llm

_SYSTEM_PROMPT_BASE = (
    "你是一位擁有 10 年經驗的專業健身顧問，根據使用者的訓練紀錄並結合自身專業知識來提供建議與回答問題。\n\n"
    "## 回覆原則\n"
    "1. **分析能力**：主動分析使用者的訓練趨勢與進步幅度，例如重量變化、組數提升等。\n"
    "2. **訓練建議**：根據紀錄給出下次訓練的具體建議，包括重量調整、組數安排、動作改進方向。\n"
    "3. **格式規範**：回覆請使用條列式或表格呈現重點，讓資訊清晰易讀。\n"
    "4. **安全提醒**：若發現紀錄中有潛在風險（如重量驟增、疑似不適備註），請主動提醒注意姿勢與傷痛預防。\n\n"
    "## 限制\n"
    "- 若紀錄中無相關資訊，請誠實回答不知道。\n"
    "- 回覆時請使用繁體中文。\n"
    "## 回話風格\n"
    "親切、專業、有耐心、會主動關心使用者狀況"
)

class LLMService:
    """專責處理主線對話與 RAG Prompt 組裝"""


    def generate_reply(self, question: str, context_chunks: list[str], api_key: str = None, is_paid: bool = False, strategy: str = "B", user_profile: str = None, llm_provider: str = "groq", model_name: str = None) -> str:
        """
        接收檢索結果與問題，交給 LLM 判讀並生成回文字串。
        實踐 FP (函數式編程)：純函數與 Map 清洗管線
        """
        
        # 定義 Pure Functions 進行資料管線處理 (無副作用)
        def truncate_if_needed(text: str) -> str:
            return f"{text[:600]}..." if strategy == "B" and len(text) > 600 else text
            
        def format_record(index: int, text: str) -> str:
            return f"【紀錄 {index+1}】{text}"

        # 透過 map 串聯資料，取代具備狀態異動思維的 for 迴圈
        cleaned_chunks = map(truncate_if_needed, context_chunks)
        formatted_chunks = [format_record(i, c) for i, c in enumerate(cleaned_chunks)]
        context_text = "\n\n".join(formatted_chunks)

        # 動態組合 System Prompt
        dynamic_system_prompt = _SYSTEM_PROMPT_BASE
        if user_profile:
            dynamic_system_prompt += f"\n\n### 使用者真實身體檔案：\n{user_profile}"

        messages = [
            SystemMessage(content=dynamic_system_prompt),
            HumanMessage(content=(
                f"以下是使用者的健身紀錄：\n\n{context_text}\n\n"
                f"使用者的問題：{question}"
            )),
        ]
        
        # 動態建立 LLM 實體 (DIP)
        if not model_name:
            if llm_provider == "groq":
                model_name = "llama-3.3-70b-versatile" if is_paid else LLM_MODEL
            elif llm_provider == "openai":
                model_name = "gpt-4o"
            elif llm_provider == "ollama":
                model_name = "llama3.1"

        actual_api_key = api_key if api_key else GROQ_API_KEY
        
        llm = get_llm(
            provider=llm_provider,
            model_name=model_name,
            api_key=actual_api_key,
            temperature=0.3
        )
        
        resp = llm.invoke(messages)
        return resp.content

llm_service = LLMService()
