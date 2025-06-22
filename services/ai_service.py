# services/ai_service.py

import os

from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import SystemMessage, HumanMessage


class AIService:
    def __init__(self):
        # Updated with correct model names
        self.openai_llm = ChatOpenAI(
            model_name="chatgpt-4o-latest",  # Updated default
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.anthropic_llm = ChatAnthropic(
            model="claude-sonnet-4-0",  # Updated default
            temperature=0.7,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        # Available models mapping
        self.available_models = {
            "openai": [
                "gpt-4o", "chatgpt-4o-latest", "gpt-4o-mini",
                "gpt-4.1", "gpt-4.1-mini", "gpt-3.5-turbo",
                "gpt-4", "gpt-4-turbo", "gpt-4.5",
                "o1", "o1-2024-12-17", "o1-preview", "o1-mini",
                "o3-mini", "o3"
            ],
            "anthropic": [
                "claude-opus-4-0", "claude-sonnet-4-0",
                "claude-3-7-sonnet-latest", "claude-3-5-sonnet-latest",
                "claude-3-5-haiku-latest", "claude-3-opus-latest"
            ]
        }

    async def get_ai_response(
            self,
            provider: str,
            model: str,
            user_message: str,
            context: str = None
    ) -> str:
        """
        provider: "openai" or "anthropic"
        model: specific model name
        """
        # Build LangChain messages
        msgs = []
        if context:
            msgs.append(SystemMessage(content=f"You are a helpful pair-programmer assistant.\nContext:\n{context}"))
        msgs.append(HumanMessage(content=user_message))

        # Choose the right LLM with specific model
        if provider == "openai":
            llm = ChatOpenAI(
                model_name=model,
                temperature=0.7,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        else:  # anthropic
            llm = ChatAnthropic(
                model=model,
                temperature=0.7,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
            )

        # Get response
        result = await llm.agenerate([msgs])
        return result.generations[0][0].text

    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        # If you still need token counting
        from tiktoken import encoding_for_model
        try:
            enc = encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            return len(text.split()) * 1  # rough fallback
