"""LLM abstraction layer supporting multiple providers (Vertex AI, OpenAI).

This module provides a unified interface for LLM calls, with automatic
fallback and provider selection based on environment configuration.

Usage:
    from app.core.llm import get_llm_client, LLMProvider

    client = get_llm_client()
    response = client.chat(
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.1
    )

    # For structured JSON output (Gemini):
    response = client.chat(
        messages=[...],
        temperature=0.1,
        json_mode=True,
        json_schema={...}  # Optional: enforce specific schema
    )
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# Provider constants
LLM_PROVIDER_VERTEX = "vertex"
LLM_PROVIDER_OPENAI = "openai"


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send a chat completion request and return the response text.

        Args:
            messages: List of chat messages with role and content
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            json_mode: If True, force JSON output (no markdown code blocks)
            json_schema: Optional JSON schema to enforce structure (Gemini only)
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM client is properly configured and available."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name for logging."""
        pass


class VertexAIClient(LLMClient):
    """Google Gemini client implementation using google-genai SDK."""

    def __init__(self):
        self._client = None
        self._project_id = os.getenv("GCP_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT", "plotpointe"))
        self._location = os.getenv("GCP_LOCATION", "us-central1")
        self._model_name = os.getenv("VERTEX_MODEL", "gemini-2.0-flash")

    def _ensure_client(self):
        if self._client is None:
            try:
                from google import genai

                # Use Vertex AI backend with Application Default Credentials
                self._client = genai.Client(
                    vertexai=True,
                    project=self._project_id,
                    location=self._location,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Google GenAI: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        self._ensure_client()
        from google.genai import types

        # Convert chat messages to Gemini format
        # Extract system instruction and user content
        system_instruction = None
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_instruction = content
            else:
                # Map "assistant" to "model" for Gemini
                gemini_role = "model" if role == "assistant" else "user"
                contents.append(types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=content)]
                ))

        # Build generation config with optional JSON mode
        config_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens or 2048,
            "system_instruction": system_instruction,
        }

        # Enable structured JSON output if requested
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
            if json_schema:
                config_kwargs["response_schema"] = json_schema

        config = types.GenerateContentConfig(**config_kwargs)

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=config,
        )
        return response.text

    def is_available(self) -> bool:
        try:
            from google import genai  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def provider_name(self) -> str:
        return f"gemini/{self._model_name}"


class OpenAIClient(LLMClient):
    """OpenAI client implementation (legacy support)."""

    def __init__(self):
        self._client = None
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        self._ensure_client()
        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        # Enable JSON mode for OpenAI (schema not supported directly)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def is_available(self) -> bool:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return False
        try:
            from openai import OpenAI  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def provider_name(self) -> str:
        return f"openai/{self._model}"


# Cached client instance
_llm_client: Optional[LLMClient] = None


def get_llm_client(force_provider: Optional[str] = None) -> Optional[LLMClient]:
    """Get the configured LLM client.
    
    Provider priority:
    1. force_provider parameter
    2. LLM_PROVIDER env var
    3. Auto-detect: Vertex AI if available, then OpenAI
    
    Returns None if no provider is available.
    """
    global _llm_client
    
    provider = force_provider or os.getenv("LLM_PROVIDER", "auto")
    
    if provider == LLM_PROVIDER_VERTEX:
        client = VertexAIClient()
        return client if client.is_available() else None
    elif provider == LLM_PROVIDER_OPENAI:
        client = OpenAIClient()
        return client if client.is_available() else None
    else:  # auto
        # Try Vertex AI first (preferred for GCP), then OpenAI
        vertex = VertexAIClient()
        if vertex.is_available():
            return vertex
        openai_client = OpenAIClient()
        if openai_client.is_available():
            return openai_client
        return None

