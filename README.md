# 🏋️ RAG 健身教練

> 本地隱私檢索 + Groq 雲端生成的 AI 健身教練系統

基於 **RAG（Retrieval-Augmented Generation）** 架構，將你的健身訓練紀錄向量化後存於本地 FAISS 索引，提問時透過語意搜尋找出最相關的紀錄，再由 Groq 雲端 LLM 生成專業回答。

## ✨ 特色功能

- 🔒 **隱私優先**：資料切塊與向量搜尋 100% 在本地執行
- 🧠 **Query Rewriting**：自動將口語化問題改寫為精準搜尋關鍵詞
- 📅 **時間意圖偵測**：問「最近 5 筆紀錄」自動按日期排序
- 🔍 **語意搜尋**：使用 FAISS + BGE-small-zh 進行中文向量檢索
- ⚡ **極速生成**：透過 Groq API 呼叫 Llama 3.3 70B 模型
- 🏗️ **LangChain 架構**：標準化的 RAG pipeline

## 🛠️ 技術堆疊

| 元件 | 技術 |
|------|------|
| 框架 | [LangChain](https://python.langchain.com/) |
| 向量化模型 | [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)（~90MB） |
| 向量庫 | [FAISS](https://github.com/facebookresearch/faiss)（CPU） |
| 雲端 LLM | [Groq](https://groq.com/) — Llama 3.3 70B |
| 後端 API | [FastAPI](https://fastapi.tiangolo.com/) |
| 前端介面 | [Streamlit](https://streamlit.io/) |

## 📁 專案結構

```
rag-fitness-coach/
├── data/
│   ├── Train.txt            # 你的健身紀錄（TXT 格式）
│   └── faiss_index/         # FAISS 索引（由 indexer 產出）
├── src/
│   ├── indexer.py           # 讀取資料 → 向量化 → 建立 FAISS 索引
│   ├── api.py               # FastAPI 伺服器（RAG pipeline）
│   └── app.py               # Streamlit 聊天介面
├── .env.example             # 環境變數範本
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 設定環境變數

```bash
cp .env.example .env
# 編輯 .env，填入你的 Groq API Key
```

> 前往 [Groq Console](https://console.groq.com/) 免費取得 API Key

### 3. 準備你的訓練紀錄

將你的健身紀錄放入 `data/` 資料夾，支援 `.txt` 和 `.json` 格式。

TXT 範例格式（以雙換行分隔每次訓練）：

```
0131 1730練
深蹲 空槓 12*1、40kg 12*1、60kg 8*1、80kg 3*1、90kg 5*3
啞鈴肩推 16kg 8*4

0129 1930練
壓肩 79kg 12*3
正手滑輪高位下拉 45kg 12*4
```

### 4. 建立索引

請使用 module 模式執行，以確保 Python 能夠抓到模組路徑：

```bash
python -m src.indexer
```

### 5. 啟動服務

我們提供了一個主要入口 `main.py`，可以直接同時啟動 FastAPI 後端與 Streamlit 前端：

```bash
python main.py
```

打開瀏覽器前往 `http://localhost:8501` 即可使用。

## 💬 使用範例

| 問題 | 檢索策略 |
|------|----------|
| 我壓肩做過最重幾公斤？ | FAISS 語意搜尋 |
| 最近 5 筆訓練紀錄 | 時間意圖 → 日期排序 |
| 我的胸推有進步嗎？ | FAISS 語意搜尋 + 趨勢分析 |
| 幫我安排下次的腿部訓練 | FAISS 語意搜尋 + 訓練建議 |

## 📐 架構圖

```
使用者提問
    │
    ▼
Query Rewriting（Groq LLM 改寫為搜尋關鍵詞）
    │
    ▼
┌─────────────── 簡易規則路由 ───────────────┐
│                                             │
│  全量意圖？ ──是──▶ 回傳所有紀錄             │
│    │否                                      │
│  時間意圖？ ──是──▶ 按日期排序取最新 N 筆    │
│    │否                                      │
│  FAISS 語意搜尋（Top-K 最相關紀錄）          │
│                                             │
└─────────────────────────────────────────────┘
    │
    ▼
System Prompt（提示詞工程）+ 紀錄 + 問題 → Groq LLM 生成回答
    │
    ▼
Streamlit 顯示回覆 + 參考紀錄
```

## 🧩 目前方法與改進方向

### 1. 簡單的提示詞工程

**目前做法**

使用固定的 `SYSTEM_PROMPT`，包含：

- **角色設定**：10 年經驗的專業健身教練
- **回覆原則**：分析能力、訓練建議、格式規範、安全提醒
- **使用者檔案**：身高體重、訓練目標、飲食偏好等個人資訊
- **回話風格**：親切、專業、有耐心

```python
# 目前結構（簡化示意）
SYSTEM_PROMPT = """
角色：10年專業健身教練
回覆原則：分析趨勢 / 給建議 / 格式化 / 安全提醒
使用者檔案：姓名、身高、體重、目標...
回話風格：親切、專業
"""
```

**⚠️ 目前限制**

- 使用者檔案寫死在程式碼中，無法動態切換
- 沒有 Few-shot Examples（範例問答）引導輸出格式
- 沒有 Chain-of-Thought（思維鏈）引導推理過程

**🔧 改進方向**

| 改進項目 | 說明 |
|---------|------|
| **Few-shot Prompting** | 在 Prompt 中加入 2~3 組範例問答，讓 LLM 學會格式與推理方式 |
| **動態使用者檔案** | 將使用者資料存入 `config.yaml` 或資料庫，啟動時載入 |
| **Chain-of-Thought** | 要求 LLM 先分析紀錄趨勢，再給出建議，提升回答品質 |
| **Prompt 版本管理** | 使用 LangChain `PromptTemplate` 管理不同版本的 Prompt |
| **多語言支援** | 根據使用者語言偏好動態切換 System Prompt |

---

### 2. 從簡易規則路由邁向「語意路由 (Semantic Routing)」

**目前做法：基於正則表達式的啟發式路由**

目前系統採用 **正則表達式（Regex）** 來捕捉問題中的關鍵字，從而決定走哪一條檢索路徑：

```python
# 全量意圖（統計類問題）
_ALL_KEYWORDS = re.compile(r"幾筆|幾次|多少筆|總共|全部|所有|統計|總結")

# 時間意圖（排序類問題）
_TEMPORAL_KEYWORDS = re.compile(r"最近|最新|上次|前(\d+)筆|最近(\d+)筆")
```

| 偵測到的意圖 | 目前路由路徑 | 範例 |
|------------|------|------|
| 全量意圖 | 回傳所有紀錄 | 「我有幾筆資料」「總共練了幾次」 |
| 時間意圖 | 按日期排序取 N 筆 | 「最近 5 筆」「上次練什麼」 |
| 無特殊意圖 | FAISS 語意搜尋 Top-K | 「壓肩怎麼進步」 |

**⚠️ 目前限制 (Regex 的瓶頸)**

- **語意理解不足**：正則表達式只能匹配固定的字面詞彙，無法理解使用者的真實意圖。例如問「我練了多久時間了」，語意上是統計類問題，但因為沒有觸發關鍵字，會錯誤地掉進純向量搜尋。
- **維護成本高**：隨著系統能回答的問題類型變多，if-else 路由規則會呈指數級膨脹，且容易發生規則衝突。

**🔧 下一步進化：語意路由 (Semantic Routing)**

為了打造企業級的 RAG 系統，接下來的改版重點會將目前的硬體規則升級為**語意路由架構**：

| 技術升級項目 | 實作說明與預期效益 |
|---------|------|
| **Semantic Router 導入** | 取代 Regex，預先定義各種「意圖向量 (Intent Vectors)」。當使用者提問時，比較問題與這些意圖向量的距離，以數學方式精準決定路由。 |
| **LLM 意圖分類器** | 利用輕量且極速的 LLM (如 Llama 3 8B) 做零樣本 (Zero-shot) 的意圖判斷，能聽懂比 Regex 更複雜的口語變化。 |
| **LangChain Router Chain** | 將現有架構重構為 `MultiPromptChain` 或功能更完整的 Agent Agentic Workflow，讓系統具備多步思考 (Multi-hop Reasoning) 的能力。 |
| **自我反思與確認** | 遇到語意模糊的問題時，系統不會盲目猜測路由，而是主動發起澄清問題 (Clarifying Questions) 向使用者確認意圖。 |

## ⚙️ 自訂設定

- **TOP_K**：`src/api.py` 中的 `TOP_K` 控制每次檢索的紀錄數量（預設 10）
- **System Prompt**：`src/api.py` 中的 `SYSTEM_PROMPT` 可自訂 AI 教練的角色設定
- **使用者資料**：在 `SYSTEM_PROMPT` 中修改使用者的身體數據與訓練目標

## 📝 License

MIT

