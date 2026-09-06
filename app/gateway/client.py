
from types import SimpleNamespace

from langchain_groq import ChatGroq
from openai import OpenAI

from app.config import settings


# The project was previously using Portkey with an inline config object, but Portkey
# rejects that pattern for this account and returns a 400 asking for a saved config slug.
# To keep the app working reliably, route the app through Groq directly while keeping
# the old `portkey_client.chat.completions.create(...)` interface for compatibility.

_groq_client = OpenAI(
    api_key=settings.GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


class _GroqCompletionsCompat:
    @staticmethod
    def create(messages, model=None, temperature=0.1, **kwargs):
        return _groq_client.chat.completions.create(
            model=model or settings.GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )


portkey_client = SimpleNamespace(
    chat=SimpleNamespace(completions=_GroqCompletionsCompat())
)


def get_langchain_llm(feature: str = "rag") -> ChatGroq:
    """
    Return a direct Groq-backed LangChain LLM.
    """
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0,
        max_retries=2,
    )


def extract_cache_status(response) -> str:
    """
    Compatibility shim: direct Groq calls do not expose Portkey cache headers.
    """
    return "DIRECT"