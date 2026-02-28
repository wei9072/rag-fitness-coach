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
                resp = requests.post(API_URL, json={"question": prompt}, timeout=30)
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
