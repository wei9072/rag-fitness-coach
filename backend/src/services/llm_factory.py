from langchain_core.language_models.chat_models import BaseChatModel


def resolve_model_and_key(
    llm_provider: str = "groq",
    model_name: str | None = None,
    api_key: str | None = None,
    is_paid: bool = False,
) -> tuple[str, str | None]:
    """
    統一的模型名稱 + API Key 解析邏輯（消除各模組重複程式碼）。
    回傳 (resolved_model_name, resolved_api_key)。
    """
    from src.config.settings import GROQ_API_KEY, LLM_MODEL

    if not model_name:
        if llm_provider == "groq":
            model_name = "llama-3.3-70b-versatile" if is_paid else LLM_MODEL
        elif llm_provider == "openai":
            model_name = "gpt-4o-mini"
        elif llm_provider == "ollama":
            model_name = "llama3.1"

    resolved_key = api_key if api_key else GROQ_API_KEY
    return model_name, resolved_key


def get_llm(provider: str, model_name: str, api_key: str = None, temperature: float = 0.0) -> BaseChatModel:
    """
    LLM 供應商工廠 (實踐依賴反轉原則 DIP)
    負責實例化並回傳對應的 LangChain ChatModel。
    """
    provider = provider.lower()
    
    if provider == "groq":
        from langchain_groq import ChatGroq
        if not api_key:
            raise ValueError("使用 Groq 模型需要提供 API Key")
        return ChatGroq(model=model_name, api_key=api_key, temperature=temperature)
        
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        if not api_key:
            raise ValueError("使用 OpenAI 模型需要提供 API Key")
        return ChatOpenAI(model=model_name, api_key=api_key, temperature=temperature)
        
    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        # Ollama 通常跑在 localhost，不需要 API Key
        return ChatOllama(model=model_name, temperature=temperature)
        
    else:
        raise ValueError(f"不支援的模型供應商: {provider}")
