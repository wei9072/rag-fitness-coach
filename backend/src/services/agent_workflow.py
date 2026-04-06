from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import Literal
from src.services.llm_factory import get_llm, resolve_model_and_key
from src.services.llm_service import llm_service
from src.services.retrieval_strategy import BaseRetriever
from src.models.schemas import SqlGeneration
from src.db.database import execute_safe_select

class ContextEvaluation(BaseModel):
    """用於約束 LLM 結構化輸出的評估結果"""
    is_relevant: Literal["YES", "NO"] = Field(description="判斷檢索資料是否足以回答使用者的問題，只能填 'YES' 或 'NO'")

class QueryRewrite(BaseModel):
    """用於約束 LLM 結構化輸出的改寫關鍵字"""
    new_queries: list[str] = Field(description="根據原始問題，產生 2 個更精準、更具體或同義的搜尋關鍵字")

class AgenticWorkflow:
    """
    實踐 Agentic RAG 工作流：
    涵蓋「檢索 (Retrieval)」->「評估反思 (Self-Reflection)」->「查詢改寫 (Query Rewrite)」的閉環機制。
    只負責調度管線，實際邏輯委託給 strategy、LLMService。
    """
    def __init__(self):
        self.max_retries = 2

        # Text-to-SQL Prompt
        self.text_to_sql_system_prompt = (
            "你是一個 Text-to-SQL 助手。根據使用者的健身相關問題，生成精確的 SQLite SELECT 語句。\n\n"
            "## 資料表結構\n"
            "TABLE: training_logs\n"
            "- user_id TEXT (使用者 ID)\n"
            "- date TEXT (YYYY-MM-DD 格式)\n"
            "- exercise TEXT (動作名稱，例如 '深蹲', '壓肩', '啞鈴肩推', '史密斯RDL', '正手滑輪高位下拉')\n"
            "- weight_kg REAL (重量公斤，NULL 表示自重)\n"
            "- reps INTEGER (每組反覆次數)\n"
            "- sets INTEGER (該重量×次數的組數)\n"
            "- note TEXT (備註，例如 '空槓', '單邊', '自重')\n\n"
            "## 規則\n"
            "1. 只能生成 SELECT 語句\n"
            "2. 必須包含 WHERE user_id = :user_id 條件\n"
            "3. exercise 欄位用 LIKE '%關鍵字%' 做模糊匹配\n"
            "4. 常用聚合: MAX(weight_kg) 最重, COUNT(DISTINCT date) 訓練天數, SUM(reps * sets) 總次數\n"
            "5. 總訓練量 (volume) = SUM(weight_kg * reps * sets)\n"
            "6. 日期比較用字串比較 (YYYY-MM-DD 天然排序)\n"
            "7. 適當使用 GROUP BY、ORDER BY 組織結果\n"
        )

        # 評估用 Prompt
        self.eval_system_prompt = (
            "你是一個嚴格的上下文評估員 (Context Evaluator)。"
            "你的任務是判斷給定的「檢索資料」是否包含了能夠回答「使用者問題」的相關資訊。"
            "如果資料中包含可以解答或推論出答案的線索、數據、動作名稱，請回傳 'YES'。"
            "如果資料完全無關、或是沒有足夠資訊可以回答該問題，請回傳 'NO'。"
            "注意：請寬容一點，只要有部分相關即可標記為 YES。"
        )
        
        # 改寫用 Prompt
        self.rewrite_system_prompt = (
            "你是一個專業的搜尋關鍵字優化助手 (Query Rewriter)。"
            "使用者的原始問題在先前的資料庫檢索中沒有找到相關資料。"
            "請分析使用者的原始問題，將其轉換為 2 個更精準、不同詞彙或不同面向的「搜尋關鍵字」(例如健身術語、同義詞、繁簡轉換)。"
            "回傳的結果必須是一個包含 2 個字串的列表。"
        )

    def run_agentic_rag(
        self,
        strategy: BaseRetriever,
        original_query: str,
        refined_query: str,
        chat_history: str = "",
        user_id: str = "default_user",
        api_key: str | None = None,
        is_paid: bool = False,
        llm_provider: str = "groq",
        model_name: str | None = None,
        user_profile: str | None = None,
        intent_category: str = "QA_INTENT"
    ) -> tuple[str, list[str]]:
        """
        執行 Agent Loop。
        回傳: (最終生成的回答, 使用的檢索文件列表)
        """
        # ── AGGREGATE_INTENT → Text-to-SQL 短路 ──
        if intent_category == "AGGREGATE_INTENT":
            print("📊 [Agent Loop] AGGREGATE_INTENT detected, routing to Text-to-SQL...")
            return self._run_aggregate_sql(
                original_query=original_query,
                user_id=user_id,
                chat_history=chat_history,
                api_key=api_key,
                is_paid=is_paid,
                llm_provider=llm_provider,
                model_name=model_name,
                user_profile=user_profile,
            )

        current_query = refined_query
        retries = 0
        all_retrieved_docs = []
        
        model_name, actual_api_key = resolve_model_and_key(llm_provider, model_name, api_key, is_paid)
        
        # 建立共用的決策 LLM (透過 Factory 實踐 DIP)
        decision_llm = get_llm(
            provider=llm_provider,
            model_name=model_name,
            api_key=actual_api_key,
            temperature=0.0
        )
        
        # 綁定 Pydantic Schema
        evaluator = decision_llm.with_structured_output(ContextEvaluation)
        rewriter = decision_llm.with_structured_output(QueryRewrite)

        while retries <= self.max_retries:
            print(f"\n🔄 [Agent Loop] 迭代 {retries}: 使用關鍵字 '{current_query}' 進行檢索...")
            
            # 步驟 A：檢索資料 (強制隔離 user_id)
            docs = strategy.retrieve(query=current_query, user_id=user_id)
            all_retrieved_docs.extend(docs) # 紀錄歷程
            
            if not docs and intent_category != "PLANNING_INTENT":
                print("⚠️ [Agent Loop] 找不到任何切塊資料，直接標記為 NO 並準備改寫。")
                eval_result = "NO"
            elif intent_category == "PLANNING_INTENT":
                print("📅 [Agent Loop] 意圖為規劃菜單 (PLANNING_INTENT)，跳過嚴格的上下文驗證，直接根據可用基準生成計畫。")
                eval_result = "YES"
            else:
                # 步驟 B：評估相關性 (Self-Reflection)
                context_text = "\n\n".join([f"【紀錄 {i+1}】{d}" for i, d in enumerate(docs)])
                eval_msg = [
                    SystemMessage(content=self.eval_system_prompt),
                    HumanMessage(content=f"使用者問題：{original_query}\n\n檢索到的資料：\n{context_text}")
                ]
                
                try:
                    eval_res = evaluator.invoke(eval_msg)
                    eval_result = eval_res.is_relevant
                except Exception as e:
                    print(f"⚠️ [Agent Loop] 評估模型發生錯誤: {e}，保底假定為 YES。")
                    eval_result = "YES"
                    
            print(f"🧐 [Agent Loop] Context Relevance 評估結果: {eval_result}")

            # 步驟 C：決策與生成
            if eval_result == "YES":
                print("✅ [Agent Loop] 資料相關！結束檢索迴圈，進入生成階段...")
                final_answer = llm_service.generate_reply(
                    question=original_query,
                    context_chunks=docs,
                    chat_history=chat_history,
                    api_key=api_key,
                    is_paid=is_paid,
                    user_profile=user_profile,
                    llm_provider=llm_provider,
                    model_name=model_name,
                    intent_category=intent_category
                )
                return final_answer, docs
                
            # 若為 NO，是否已達上限？
            if retries == self.max_retries:
                print("🛑 [Agent Loop] 已達最大重試次數，依然無相關資料。避免幻覺，直接回覆。")
                return ("在您的健身紀錄中找不到相關資訊，建議您換個方式詢問，或是先新增相關的訓練紀錄喔！", all_retrieved_docs)
                
            # 還未達上限，進行 Query Rewrite
            retries += 1
            print(f"✍️ [Agent Loop] 觸發「查詢改寫」(第 {retries} 次重試)...")
            
            try:
                rewrite_msg = [
                    SystemMessage(content=self.rewrite_system_prompt),
                    HumanMessage(content=f"原始問題：{original_query}\n先前使用的搜尋關鍵字 (已失效)：{current_query}")
                ]
                rewrite_res = rewriter.invoke(rewrite_msg)
                new_queries = rewrite_res.new_queries
                # 將新生成的關鍵字串接成一個豐富的搜尋字串，丟回迴圈
                current_query = " ".join(new_queries)
                print(f"✨ [Agent Loop] LLM 改寫後的新關鍵字: {current_query}")
            except Exception as e:
                print(f"⚠️ [Agent Loop] 查詢改寫發生錯誤: {e}，強制中斷。")
                return ("系統在查找資料時遇到了點問題，請稍後再試。", all_retrieved_docs)
                
        # 防呆回傳
        return "在您的健身紀錄中找不到足夠且精準的資訊。", all_retrieved_docs

    # ════════════════════════════════════════
    # Text-to-SQL 聚合查詢
    # ════════════════════════════════════════

    def _run_aggregate_sql(
        self,
        original_query: str,
        user_id: str,
        chat_history: str = "",
        api_key: str | None = None,
        is_paid: bool = False,
        llm_provider: str = "groq",
        model_name: str | None = None,
        user_profile: str | None = None,
    ) -> tuple[str, list[str]]:
        """Text-to-SQL 路線：LLM 生成 SQL → 唯讀執行 → LLM 組裝自然語言回覆。"""

        model_name, actual_api_key = resolve_model_and_key(llm_provider, model_name, api_key, is_paid)

        sql_llm = get_llm(
            provider=llm_provider,
            model_name=model_name,
            api_key=actual_api_key,
            temperature=0.0,
        )
        sql_generator = sql_llm.with_structured_output(SqlGeneration)

        # Step A: 動態補充可用的 exercise 清單
        prompt = self.text_to_sql_system_prompt
        try:
            exercises = execute_safe_select(
                "SELECT DISTINCT exercise FROM training_logs WHERE user_id = :user_id",
                {"user_id": user_id},
            )
            if exercises:
                names = ", ".join(f"'{r['exercise']}'" for r in exercises)
                prompt += f"\n## 資料庫中現有的動作名稱\n{names}\n"
        except Exception:
            pass

        if chat_history:
            prompt += f"\n## 近期對話語境\n{chat_history}"

        # Step B: 生成 SQL
        try:
            sql_result = sql_generator.invoke([
                SystemMessage(content=prompt),
                HumanMessage(content=original_query),
            ])
            generated_sql = sql_result.sql.strip()
            print(f"  🗄️ [Text-to-SQL] SQL: {generated_sql}")
            print(f"  🗄️ [Text-to-SQL] 說明: {sql_result.explanation}")
        except Exception as e:
            print(f"  ⚠️ [Text-to-SQL] SQL 生成失敗: {e}")
            return "抱歉，無法理解您的數據查詢需求，請試著換個方式提問。", []

        # Step C: 安全執行
        try:
            rows = execute_safe_select(generated_sql, {"user_id": user_id})
        except Exception as e:
            print(f"  ⚠️ [Text-to-SQL] SQL 執行失敗: {e}")
            return f"查詢執行失敗，請換個方式提問。（錯誤：{e}）", [f"[SQL ERROR] {generated_sql}"]

        # Step D: 格式化查詢結果
        if not rows:
            result_text = "查詢結果：無符合條件的紀錄。"
        else:
            lines = [str(row) for row in rows[:30]]
            result_text = f"查詢結果（{len(rows)} 筆）：\n" + "\n".join(lines)

        context = f"SQL 查詢：{generated_sql}\n說明：{sql_result.explanation}\n\n{result_text}"

        # Step E: 用 LLM 生成自然語言回答
        final_answer = llm_service.generate_reply(
            question=original_query,
            context_chunks=[context],
            chat_history=chat_history,
            api_key=api_key,
            is_paid=is_paid,
            user_profile=user_profile,
            llm_provider=llm_provider,
            model_name=model_name,
            intent_category="QA_INTENT",  # 用嚴謹的 QA prompt
        )

        return final_answer, [f"[SQL] {generated_sql}"]


# 輸出單例
agent_workflow = AgenticWorkflow()
