# services/ai_service.py

import os
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from utils.logger import get_logger, log_method

logger = get_logger(__name__)


# noinspection PyArgumentList
class AIService:
    def __init__(self):
        logger.logger.info("Initializing AIService")

        # Updated with correct model names
        self.openai_llm = ChatOpenAI(
            model="chatgpt-4o-latest",  # Updated default
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.anthropic_llm = ChatAnthropic(
            model_name="claude-sonnet-4-0",  # Updated default
            temperature=0.7,
            api_key=os.getenv("ANTHROPIC_API_KEY")
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

        logger.logger.info(
            f"AIService initialized with {len(self.available_models['openai'])} OpenAI models and {len(self.available_models['anthropic'])} Anthropic models")

    @log_method()
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
        logger.log_data('IN', 'AI_REQUEST', {
            'provider': provider,
            'model': model,
            'message_length': len(user_message),
            'has_context': bool(context),
            'context_length': len(context) if context else 0
        })

        # Build LangChain messages
        msgs = []

        # Enhanced system message with context
        if context:
            system_content = f"""You are a helpful pair-programmer assistant with memory of our previous conversations.

    RELEVANT CONTEXT FROM OUR HISTORY:
    {context}

    Remember to use the above context when answering. If the user refers to something we discussed before, use that information.
    If you're unsure about something we previously discussed, you can mention what you remember from the context above."""

            msgs.append(SystemMessage(content=system_content))
            logger.logger.debug(f"Added system message with {len(context)} chars of context")
        else:
            # Even without context, set up the assistant's personality
            msgs.append(SystemMessage(
                content="You are a helpful pair-programmer assistant. Since we have no previous context, please let the user know if they reference something we haven't discussed yet."))

        msgs.append(HumanMessage(content=user_message))

        logger.logger.debug(f"Sending request to {provider}/{model}")

        # Choose the right LLM with specific model
        if provider == "openai":
            llm = ChatOpenAI(
                model=model,
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        else:  # anthropic
            llm = ChatAnthropic(
                model_name=model,
                temperature=0.7,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )

        # Get response
        result = await llm.agenerate([msgs])
        response_text = result.generations[0][0].text

        logger.log_data('OUT', 'AI_RESPONSE', {
            'provider': provider,
            'model': model,
            'response_length': len(response_text),
            'response_preview': response_text[:200] + '...' if len(response_text) > 200 else response_text
        })

        return response_text

    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        # If you still need token counting
        from tiktoken import encoding_for_model
        try:
            enc = encoding_for_model(model)
            token_count = len(enc.encode(text))
            logger.logger.debug(f"Counted {token_count} tokens for {len(text)} chars")
            return token_count
        except Exception as e:
            logger.logger.warning(f"Token counting failed: {e}, using fallback")
            return len(text.split()) * 1  # rough fallback
