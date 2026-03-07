import os
import sys

# 確保 tests 資料夾內的腳本可以 import 外層的 src 模組
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from src.services.intent_router import intent_router
from src.services.agent_workflow import agent_workflow
from src.services.llm_factory import get_llm
from src.config.settings import LLM_MODEL, GROQ_API_KEY

# ---------------------------------------------------------
# 1. 定義測試資料集 (Test Dataset)
# ---------------------------------------------------------
TEST_DATASET = [
    {
        "question": "我壓肩做過最重幾公斤？大約是什麼時候測的？",
        "ground_truth": "使用者壓肩的最重紀錄與日期（如果有的話）。或者明確告知找不到。"
    },
    {
        "question": "最近幾次的深蹲紀錄，我的重量有進步嗎？",
        "ground_truth": "比較近期的深蹲重量數據，並給出是否進步的趨勢分析。"
    },
    {
        "question": "可以幫我安排明天下半身的菜單嗎？",
        "ground_truth": "考量過去下半身訓練紀錄（如深蹲、硬舉等），給出一套合理的肌力訓練建議課表。"
    }
]

# ---------------------------------------------------------
# 2. 定義 LLM-as-a-Judge 的評估綱要
# ---------------------------------------------------------
class EvaluationResult(BaseModel):
    context_precision: float = Field(..., ge=0.0, le=1.0, description="0~1分，評估檢索到的資料是否包含足夠且精確的資訊來回答問題。")
    faithfulness: float = Field(..., ge=0.0, le=1.0, description="0~1分，評估生成的回答是否完全基於檢索到的資料，無幻覺。")
    answer_relevance: float = Field(..., ge=0.0, le=1.0, description="0~1分，評估生成的回答是否直接、有效且完整地回應了使用者的問題。")
    reasoning: str = Field(..., description="針對上述三個給分的綜合簡短理由（約30~50字）")

# ---------------------------------------------------------
# 3. 實作單一問答與評分邏輯
# ---------------------------------------------------------
def evaluate_single_turn(question: str, ground_truth: str) -> dict:
    """執行一次完整的 RAG 提問並交由 LLM 裁判評分"""
    print(f"\n▶️ 測試問題: {question}")
    
    # --- A. 取得 RAG 回答 (調用真實管線) ---
    # 先讓 IntentRouter 抓取負責的策略與優化字串
    strategy, refined_query, intent_category = intent_router.route_query(question)
    
    # 將策略丟進 Agentic Workflow 執行
    answer, context_docs = agent_workflow.run_agentic_rag(
        strategy=strategy,
        original_query=question,
        refined_query=refined_query,
        api_key=None,          # 使用系統預設
        is_paid=True,          # 使用性能較好的模型進行 RAG
        limit_k=None,          # 預設
        user_profile=None,
        intent_category=intent_category
    )
    
    context_text = "\n---\n".join(context_docs) if context_docs else "無檢索資料"

    print(f"  [回答長度]: {len(answer)} 字")
    print(f"  [檢索片段數]: {len(context_docs)} 筆")

    # --- B. 呼叫 LLM 裁判 (LLM-as-a-Judge) ---
    judge_prompt = f"""
你是一位嚴謹的 AI 評測工程師，正在評估一套 RAG（檢索增強生成）系統的表現。
請審查以下三個輸入，並嚴格遵循給分標準，給出 0 到 1 之間的浮點數分數（例如 0.8, 1.0, 0.0）。

【輸入資料】
[User Question]
{question}

[Intent Classification]
{intent_category}

[Ground Truth (預期理想方向)]
{ground_truth}

[Retrieved Contexts (系統檢索出的資料)]
{context_text}

[Generated Answer (系統最終產出的回答)]
{answer}

【給分標準】
1. Context Precision (檢索精準度): Contexts 是否提供了回答該問題不可或缺的重要資訊？全部資訊皆有則為1.0，完全無關為0.0。若為 'PLANNING_INTENT'，只要 Contexts 給予了基本的身體/過去背景數據也算提供幫助，給 1.0。
2. Faithfulness (忠實度): Answer 是否【僅基於】提供的 Contexts 或相關常識推論？若為 'PLANNING_INTENT'，教練自主安排未來新菜單等擴展資訊是特許的，不算幻覺，請給 1.0。若為 'QA_INTENT' 卻捏造了未記載的歷史數據請給 0.0。
3. Answer Relevance (回答相關性): Answer 有沒有正面解決 User Question？有沒有符合 Ground Truth 的預期？
"""

    judge_llm = get_llm(
        provider="groq",
        model_name="llama-3.3-70b-versatile", # 裁判必須用強大模型
        api_key=GROQ_API_KEY,                 
        temperature=0.0
    )
    
    structured_judge = judge_llm.with_structured_output(EvaluationResult)
    
    try:
        evaluation = structured_judge.invoke([HumanMessage(content=judge_prompt)])
        return {
            "question": question,
            "precision": evaluation.context_precision,
            "faithfulness": evaluation.faithfulness,
            "relevance": evaluation.answer_relevance,
            "reasoning": evaluation.reasoning
        }
    except Exception as e:
        print(f"  ⚠️ 裁判系統出錯: {e}")
        return {
            "question": question,
            "precision": 0.0, "faithfulness": 0.0, "relevance": 0.0,
            "reasoning": "裁判系統錯誤"
        }

# ---------------------------------------------------------
# 4. 主控台執跑與 Markdown 成績單產出
# ---------------------------------------------------------
def main():
    print("=" * 60)
    print("🚀 啟動 LLM-as-a-Judge 自動化 RAG 評估")
    print("=" * 60)
    
    results = []
    
    for item in TEST_DATASET:
        res = evaluate_single_turn(item["question"], item["ground_truth"])
        results.append(res)
        
    print("\n" + "=" * 60)
    print("📊 評估成績單 (Evaluation Report)")
    print("=" * 60)
    
    print("| 問題 | 精準度 (CP) | 忠實度 (F) | 相關性 (AR) | 綜合短評 |")
    print("|---|---|---|---|---|")
    
    total_cp = 0.0
    total_f = 0.0
    total_ar = 0.0
    
    for r in results:
        q_trunc = r["question"][:15] + "..." if len(r["question"]) > 15 else r["question"]
        print(f"| {q_trunc} | {r['precision']:.2f} | {r['faithfulness']:.2f} | {r['relevance']:.2f} | {r['reasoning']} |")
        total_cp += r["precision"]
        total_f += r["faithfulness"]
        total_ar += r["relevance"]
        
    count = len(results)
    print("\n**✨ 最終總平均分 ✨**")
    print(f"- 🎯 Context Precision (檢索精準度): {total_cp / count:.2f}")
    print(f"- 🛡️ Faithfulness (忠誠防幻覺度): {total_f / count:.2f}")
    print(f"- 💡 Answer Relevance (回答相關性): {total_ar / count:.2f}")

if __name__ == "__main__":
    main()
