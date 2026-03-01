from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from src.config.settings import LLM_MODEL, GROQ_API_KEY

_SYSTEM_PROMPT = (
    "你是一位擁有 10 年經驗的專業健身教練，根據使用者的訓練紀錄並結合自身專業知識來回答問題。\n\n"
    "## 回覆原則\n"
    "1. **分析能力**：主動分析使用者的訓練趨勢與進步幅度，例如重量變化、組數提升等。\n"
    "2. **訓練建議**：根據紀錄給出下次訓練的具體建議，包括重量調整、組數安排、動作改進方向。\n"
    "3. **格式規範**：回覆請使用條列式或表格呈現重點，讓資訊清晰易讀。\n"
    "4. **安全提醒**：若發現紀錄中有潛在風險（如重量驟增、疑似不適備註），請主動提醒注意姿勢與傷痛預防。\n\n"
    "## 限制\n"
    "- 若紀錄中無相關資訊，請誠實回答不知道。\n"
    "- 回覆時請使用繁體中文。\n"
    "### 使用者設定\n"
    "姓名：陳韋成\n"
    "年齡：25\n"
    "性別：男\n"
    "身高：175cm\n"
    "體重：70kg\n"
    "目標：增肌\n"
    "訓練頻率：一週三次\n"
    "訓練經驗：一年\n"
    "訓練偏好：自由重量\n"
    "飲食偏好：高蛋白、低碳水\n"
    "飲食限制：無\n"
    "過敏：無\n"
    "傷病史：無\n"
    "其他備註：消化系統較弱，容易脹氣，建議多攝取益生菌\n"
    "### 回話風格\n"
    "親切、專業、有耐心、會主動關心使用者狀況"
)

class LLMService:
    """專責處理主線對話與 RAG Prompt 組裝"""


    def generate_reply(self, question: str, context_chunks: list[str], api_key: str = None, is_paid: bool = False, strategy: str = "B") -> str:
        """接收檢索結果與問題，交給 LLM 判讀並生成回文字串"""
        
        # 根據策略決定是否截斷
        if strategy == "B":
            # 策略 B: 智慧截斷，限制每個 chunk 不要過長而撐爆 Token Limit
            context_text = "\n\n".join(
                f"【紀錄 {i+1}】{c[:600]}..." if len(c) > 600 else f"【紀錄 {i+1}】{c}" 
                for i, c in enumerate(context_chunks)
            )
        else:
            # 策略 A: 原汁原味 (預期呼叫端只會傳入 1 筆)
            context_text = "\n\n".join(
                f"【紀錄 {i+1}】{c}" for i, c in enumerate(context_chunks)
            )

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"以下是使用者的健身紀錄：\n\n{context_text}\n\n"
                f"使用者的問題：{question}"
            )),
        ]
        
        # 動態建立 LLM 實例
        target_model = "llama-3.3-70b-versatile" if is_paid else LLM_MODEL
        actual_api_key = api_key if api_key else GROQ_API_KEY
        
        llm = ChatGroq(
            model=target_model,
            api_key=actual_api_key,
            temperature=0.3,
            max_tokens=1024,
        )
        
        resp = llm.invoke(messages)
        return resp.content

llm_service = LLMService()
