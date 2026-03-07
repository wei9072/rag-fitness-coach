from langchain_core.messages import SystemMessage, HumanMessage
from src.config.settings import LLM_MODEL, GROQ_API_KEY
from src.services.llm_factory import get_llm

_SYSTEM_PROMPT_QA = (
    "你是一位擁有 10 年經驗的專業健身顧問，根據使用者的訓練紀錄並結合自身專業知識來提供建議與回答問題。\n\n"
    "## 回覆原則\n"
    "1. **絕對忠實**：你的回答必須【完全基於提供的紀錄】。不得捏造未出現的重量、次數或動作。\n"
    "2. **分析能力**：主動分析使用者的訓練趨勢與進步幅度，例如重量變化、組數提升等。\n"
    "3. **誠實至上**：若紀錄中無法找出可解答的資訊，請直接回答「找不到相關歷史紀錄」，嚴禁瞎掰。\n"
    "4. **格式與態度**：回覆請客氣專業，並適時給出防傷建議。\n"
)

_SYSTEM_PROMPT_PLANNING = (
    "你是一位擁有 10 年經驗的專業健身教練與菜單規劃師。\n\n"
    "## 回覆原則\n"
    "1. **個人化菜單**：請根據提供的「過往訓練紀錄」作為學員的實力與習慣基準點，為使用者規劃接下來的訓練菜單。\n"
    "2. **循序漸進**：規劃新菜單時，不需要侷限於訓練紀錄，應擴充電腦科學化漸進超負荷原則與新動作建議。\n"
    "3. **格式規範**：請使用吸引人的 Markdown 排版（如表格或清單）清晰呈現菜單結構與動作替換選項。\n"
    "4. **安全提醒**：提醒訓練前的暖身與營養補充注意事項。\n"
)

class LLMService:
    """專責處理主線對話與 RAG Prompt 組裝"""


    def generate_reply(self, question: str, context_chunks: list[str], api_key: str = None, is_paid: bool = False, strategy: str = "B", user_profile: str = None, llm_provider: str = "groq", model_name: str = None, intent_category: str = "QA_INTENT") -> str:
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

        # 動態組合 System Prompt (根據大分類切換 Strict RAG 或 Creative Planning)
        baseline_prompt = _SYSTEM_PROMPT_QA if intent_category == "QA_INTENT" else _SYSTEM_PROMPT_PLANNING
        
        dynamic_system_prompt = baseline_prompt
        if user_profile:
            dynamic_system_prompt += f"\n### 使用者真實身體檔案：\n{user_profile}"

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
