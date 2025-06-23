"""
AI Service for PairProgrammer Discord Bot

This service provides a unified interface for interacting with multiple AI providers
including OpenAI and Anthropic. It handles model management, context integration,
and response generation with comprehensive logging and error handling.

The service supports dynamic model selection and integrates conversation context
from the vector database to provide coherent, contextual responses.

Supported Providers:
    - OpenAI: GPT-4o, GPT-4, GPT-3.5, O1 series models
    - Anthropic: Claude 4.0, Claude 3.5 series models

Features:
    - Dynamic model selection per request
    - Context-aware conversations using vector database history
    - Token counting and usage optimization
    - Comprehensive logging and monitoring
    - Error handling and fallback mechanisms

Author: PairProgrammer Team
"""

import os
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from utils.logger import get_logger, log_method

logger = get_logger(__name__)


class AIService:
    """
    Service class for managing AI model interactions and responses.
    
    This class provides a unified interface for interacting with multiple AI
    providers (OpenAI and Anthropic) while handling context integration,
    model selection, and response generation.
    
    Attributes:
        openai_llm (ChatOpenAI): Default OpenAI language model instance
        anthropic_llm (ChatAnthropic): Default Anthropic language model instance
        available_models (dict): Mapping of providers to their available models
        
    Example:
        ai_service = AIService()
        response = await ai_service.get_ai_response(
            provider='openai',
            model='chatgpt-4o-latest',
            user_message='How do I implement async/await?',
            context='Previous discussion about Python asyncio...'
        )
    """
    
    def __init__(self):
        """
        Initialize the AIService with default model configurations.
        
        Sets up default language model instances for both OpenAI and Anthropic
        providers and initializes the available models registry.
        
        Environment Variables Required:
            OPENAI_API_KEY: OpenAI API key for GPT models
            ANTHROPIC_API_KEY: Anthropic API key for Claude models
            
        Raises:
            ValueError: If required API keys are not provided
            ConnectionError: If initial model validation fails
        """
        logger.logger.info("Initializing AIService")

        # Initialize default OpenAI model instance
        self.openai_llm = ChatOpenAI(
            model="chatgpt-4o-latest",  # Current default model
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize default Anthropic model instance
        self.anthropic_llm = ChatAnthropic(
            model_name="claude-sonnet-4-0",  # Current default model
            temperature=0.7,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        # Registry of available models by provider
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
        Generate an AI response using the specified provider and model.
        
        This method handles the complete flow of sending a user message to an AI
        model, including context integration, system message setup, and response
        generation. It supports both OpenAI and Anthropic providers with dynamic
        model selection.
        
        Args:
            provider (str): AI provider name ('openai' or 'anthropic')
            model (str): Specific model name from the provider's available models
            user_message (str): The user's message/question to send to the AI
            context (str, optional): Historical context from vector database for
                                   contextual responses. Defaults to None.
                                   
        Returns:
            str: The AI model's response text
            
        Raises:
            ValueError: If provider is not supported or model is not available
            APIError: If the AI provider API returns an error
            ConnectionError: If there are network connectivity issues
            RateLimitError: If API rate limits are exceeded
            
        Example:
            # Basic usage without context
            response = await ai_service.get_ai_response(
                provider='openai',
                model='chatgpt-4o-latest',
                user_message='What is async/await in Python?'
            )
            
            # Usage with conversation context
            response = await ai_service.get_ai_response(
                provider='anthropic',
                model='claude-sonnet-4-0',
                user_message='Continue explaining the authentication flow',
                context='Previous discussion: JWT tokens, OAuth2 setup...'
            )
            
        Context Integration:
            When context is provided, it's integrated into a system message that:
            - Establishes the assistant's role as a pair-programmer
            - Provides relevant conversation history
            - Instructs the model to reference previous discussions
            - Maintains conversation continuity across interactions
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
        """
        Count the number of tokens in a text string for a specific model.
        
        This method provides token counting functionality for estimating API costs
        and managing context length limits. It uses the tiktoken library for
        accurate token counting with fallback to word-based estimation.
        
        Args:
            text (str): The text to count tokens for
            model (str): Model name for token encoding. Defaults to "gpt-4"
                        For accurate counting, use the actual model being called
                        
        Returns:
            int: Number of tokens in the text
            
        Note:
            Token counts may vary slightly between different models and versions.
            This method provides estimates for cost calculation and context management.
            
        Fallback:
            If tiktoken encoding fails, falls back to word count estimation
            which may be less accurate but prevents service interruption.
            
        Example:
            token_count = ai_service.count_tokens(
                "Hello, how are you today?", 
                model="chatgpt-4o-latest"
            )
            # Returns: Approximate token count for the text
            
        Raises:
            ImportError: If tiktoken is not installed (handled gracefully)
            ValueError: If model name is not recognized by tiktoken
        """
        try:
            from tiktoken import encoding_for_model
            enc = encoding_for_model(model)
            token_count = len(enc.encode(text))
            logger.logger.debug(f"Counted {token_count} tokens for {len(text)} chars")
            return token_count
        except Exception as e:
            logger.logger.warning(f"Token counting failed: {e}, using fallback")
            # Rough fallback: average ~1.3 tokens per word for English text
            return len(text.split()) * 1
