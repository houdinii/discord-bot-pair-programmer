# services/ai_service.py

import os

from langchain.schema import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI


class AIService:
    def __init__(self):
        # LangChain handles all HTTP under the hood—no proxies errors
        self.openai_llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        # noinspection PyArgumentList
        self.anthropic_llm = ChatAnthropic(
            model_name="claude-3-7-sonnet-latest",
            temperature=0.7,
            timeout=300,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    async def get_ai_response(
        self,
        provider: str,
        user_message: str,
        context: str = None
    ) -> str:
        """
        provider: "openai" or "anthropic"
        model is ignored here (we've hard-wired one per provider),
        but you could map it if you want multiple.
        """
        # Build LangChain messages
        msgs = []
        if context:
            msgs.append(SystemMessage(content=f"You are a helpful pair-programmer assistant.\nContext:\n{context}"))
        msgs.append(HumanMessage(content=user_message))

        # Choose the right LLM
        llm = self.openai_llm if provider == "openai" else self.anthropic_llm

        # LangChain will select the model you passed in at init
        result = await llm.agenerate([[*msgs]])
        return result.generations[0][0].text

    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        # If you still need token counting
        from tiktoken import encoding_for_model
        try:
            enc = encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            return len(text.split()) * 1  # rough fallback
