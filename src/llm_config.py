"""LLM configuration for the Data Cleaning & EDA Agent."""

from langchain_core.language_models import BaseChatModel

# Default models per provider
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-20241022",
    "groq": "llama-3.1-70b-versatile",
    "bedrock": "eu.amazon.nova-pro-v1:0",
    # "bedrock": "meta-llama3-8b-instruct-v1.0",
}

SUPPORTED_PROVIDERS = set(DEFAULT_MODELS.keys())


def get_llm(provider: str = "openai", model: str = None, **kwargs) -> BaseChatModel:
    """
    Initialize and return a tool-calling capable LLM.

    Args:
        provider: "openai" | "anthropic" | "groq" | "bedrock"
        model: Model name override. If None, uses the default for the provider.
        **kwargs: Additional keyword arguments passed to the chat model constructor.

    Returns:
        Configured LangChain chat model.

    Raises:
        ValueError: If the provider is not supported.
    """
    provider = provider.lower()

    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            f"Supported providers: {sorted(SUPPORTED_PROVIDERS)}"
        )

    model_name = model or DEFAULT_MODELS[provider]

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, **kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name, **kwargs)

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model_name, **kwargs)

    # provider == "bedrock"
    from langchain_aws import ChatBedrock

    return ChatBedrock(model_id=model_name, **kwargs)
