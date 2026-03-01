"""
app.py — Streamlit 聊天介面，串接 FastAPI 後端。
"""

import streamlit as st
import requests

# ── 設定 ─────────────────────────────────────────────────
API_URL = "http://localhost:8000/api/chat"

# ── 頁面配置 ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG 健身教練",
    page_icon="🏋️",
    layout="centered",
)

st.title("🏋️ RAG 健身教練")
st.caption("本地隱私檢索 + Groq 雲端生成 | Query Rewriting + 時間意圖偵測")

# ── 側邊欄 ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 引擎設定 (LLM Providers)")
    
    selected_provider = st.selectbox(
        "選擇 AI 供應商",
        options=["groq", "openai", "ollama"],
        index=0,
        format_func=lambda x: {"groq": "Groq (極速開源模型)", "openai": "OpenAI (GPT 系列)", "ollama": "Ollama (本地離線執行)"}[x]
    )
    
    # 根據供應商動態切換介面
    user_api_key = ""
    is_paid_tier = False
    
    if selected_provider != "ollama":
        user_api_key = st.text_input(f"{selected_provider.capitalize()} API Key", type="password", help=f"在此輸入 {selected_provider.capitalize()} 的 API Key，留空則使用系統預設")
        if selected_provider == "groq":
            is_paid_tier = st.checkbox("💎 這是付費版 (Pro) API Key", value=False)
            
    custom_model_name = st.text_input("自訂模型名稱 (留空使用預設值)", help="例如 Groq 可填 `llama-3.1-8b-instant`，OpenAI 填 `gpt-4o`，Ollama 填 `llama3.1`")
    
    st.divider()
    
    search_strategy = st.radio(
        "🧠 檢索策略 (Retrieval Strategy)",
        options=["A", "B"],
        index=1,
        format_func=lambda x: "策略 A (極限省流 - 僅檢索 1 筆)" if x == "A" else "策略 B (智慧截斷 - 檢索 5 筆並截斷長文)"
    )

    st.divider()
    
    st.header("👤 使用者設定")
    profile_template = st.selectbox(
        "選擇使用者範本",
        ["範本 1：年輕男性增肌 (陳韋成)", "範本 2：新手女性減脂 (林小美)", "自訂空白"]
    )
    
    if profile_template == "範本 1：年輕男性增肌 (陳韋成)":
        default_profile = "姓名：陳韋成\n年齡：25\n性別：男\n身高：175cm\n體重：70kg\n目標：增肌\n訓練頻率：一週三次\n訓練經驗：一年\n訓練偏好：自由重量\n飲食偏好：高蛋白、低碳水\n其他備註：消化系統較弱，建議多攝取益生菌"
    elif profile_template == "範本 2：新手女性減脂 (林小美)":
        default_profile = "姓名：林小美\n年齡：28\n性別：女\n身高：160cm\n體重：65kg\n目標：減脂與體態雕塑\n訓練頻率：一週兩次\n訓練經驗：新手\n訓練偏好：機械式器材或有氧\n飲食偏好：正常飲食，容易吃甜點\n其他備註：膝蓋曾經受傷，深蹲需謹慎"
    else:
        default_profile = ""
        
    user_profile_text = st.text_area("個人身體檔案", value=default_profile, height=200, help="這些資訊將作為背景知識告訴 AI 教練，讓他能針對你的身體狀況給出客製化建議。")

    st.divider()

    st.caption(
        "💡 **智慧檢索**\n\n"
        "- 系統會自動改寫問題為精準搜尋關鍵詞\n"
        "- 偵測到「最近、最新、上次、前N筆」等詞時，自動按日期排序\n"
        "- 其他問題走向量語意搜尋"
    )

    if st.button("🗑️ 清除對話"):
        st.session_state.messages = []
        st.rerun()

# ── 聊天歷史 ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示歷史訊息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("rewritten_query"):
            st.caption(f"🔄 改寫關鍵詞：`{msg['rewritten_query']}`")
        if msg.get("sources"):
            with st.expander("📋 參考紀錄"):
                for i, src in enumerate(msg["sources"]):
                    st.markdown(f"**紀錄 {i+1}**：{src}")

# ── 使用者輸入 ────────────────────────────────────────────
if prompt := st.chat_input("請輸入你的健身問題..."):
    # 顯示使用者訊息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 呼叫 API
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                payload = {
                    "question": prompt,
                    "api_key": user_api_key if user_api_key.strip() else None,
                    "llm_provider": selected_provider,
                    "model_name": custom_model_name if custom_model_name.strip() else None,
                    "is_paid": is_paid_tier,
                    "strategy": search_strategy,
                    "user_profile": user_profile_text if user_profile_text.strip() else None
                }
                resp = requests.post(API_URL, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                answer          = data["answer"]
                rewritten_query = data.get("rewritten_query", "")
                sources         = data.get("sources", [])
            except requests.exceptions.ConnectionError:
                answer          = "⚠️ 無法連線到後端 API，請確認 FastAPI 伺服器已啟動。"
                rewritten_query = ""
                sources         = []
            except Exception as e:
                answer          = f"⚠️ 發生錯誤：{e}"
                rewritten_query = ""
                sources         = []

        st.markdown(answer)
        if rewritten_query:
            st.caption(f"🔄 改寫關鍵詞：`{rewritten_query}`")
        if sources:
            with st.expander("📋 參考紀錄"):
                for i, src in enumerate(sources):
                    st.markdown(f"**紀錄 {i+1}**：{src}")

    # 儲存 AI 回覆
    st.session_state.messages.append({
        "role":            "assistant",
        "content":         answer,
        "rewritten_query": rewritten_query,
        "sources":         sources,
    })
