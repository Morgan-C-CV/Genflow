"""Provider-agnostic chat model wrapper.

Genflow's agents only need ``model.generate_content(text) -> response.text``.
This module keeps that contract while allowing either Google Gemini or any
OpenAI-compatible chat endpoint (DeepSeek's official API by default).

Use :func:`build_llm_model` rather than constructing a provider client directly,
so the provider choice stays in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import requests

from app.core.config import settings


@dataclass
class LLMResponse:
    """Minimal response shape; mirrors the Gemini SDK's ``.text`` attribute."""

    text: str
    raw: Any = None


class LLMRequestError(RuntimeError):
    pass


class DeepSeekChatModel:
    """OpenAI-compatible chat completions client.

    Authored for DeepSeek's official endpoint, but works with any service that
    implements ``POST {base_url}/chat/completions``.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        system_instruction: str,
        base_url: str = "https://api.deepseek.com",
        response_mime_type: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
    ):
        if not api_key.strip():
            raise RuntimeError("DEEPSEEK_API_KEY is required for DeepSeekChatModel.")
        self._api_key = api_key.strip()
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.base_url = base_url.rstrip("/")
        self.response_mime_type = response_mime_type
        self.temperature = temperature
        self.timeout = timeout or settings.LLM_REQUEST_TIMEOUT

    def __repr__(self) -> str:  # never leak the credential
        return (
            f"DeepSeekChatModel(model_name={self.model_name!r}, "
            f"base_url={self.base_url!r}, temperature={self.temperature!r})"
        )

    def generate_content(self, content: str) -> LLMResponse:
        payload: dict = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_instruction},
                {"role": "user", "content": content},
            ],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.response_mime_type == "application/json":
            payload["response_format"] = {"type": "json_object"}

        url = f"{self.base_url}/chat/completions"
        try:
            response = requests.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise LLMRequestError(f"LLM endpoint {self.base_url} is unreachable: {exc}") from exc

        if response.status_code >= 400:
            detail = response.text[:600]
            try:
                body = response.json()
                error = body.get("error") if isinstance(body, dict) else None
                if isinstance(error, dict) and error.get("message"):
                    detail = f"{error.get('message')} (status {error.get('code', response.status_code)})"
            except ValueError:
                pass
            raise LLMRequestError(
                f"LLM request failed with HTTP {response.status_code}: {detail}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise LLMRequestError("LLM returned a non-JSON response body.") from exc

        choices = body.get("choices") or []
        if not choices:
            raise LLMRequestError(f"LLM returned no choices: {str(body)[:300]}")

        message = choices[0].get("message") or {}
        text = message.get("content")
        if text is None:
            # Some reasoning models can emit only reasoning content.
            text = message.get("reasoning_content") or ""
        return LLMResponse(text=str(text), raw=body)


def active_provider() -> str:
    """Resolve which LLM provider to use for this process."""
    explicit = (settings.LLM_PROVIDER or "").strip().lower()
    if explicit in {"gemini", "deepseek"}:
        return explicit
    return "deepseek" if settings.DEEPSEEK_API_KEY.strip() else "gemini"


def build_llm_model(
    model_name: Optional[str] = None,
    system_instruction: str = "",
    response_mime_type: Optional[str] = None,
    temperature: Optional[float] = None,
):
    """Build a chat model for the configured provider.

    Both returned types expose ``generate_content(content) -> object with .text``.
    """
    provider = active_provider()

    if provider == "deepseek":
        return DeepSeekChatModel(
            api_key=settings.DEEPSEEK_API_KEY,
            model_name=model_name or settings.DEEPSEEK_MODEL,
            system_instruction=system_instruction,
            base_url=settings.DEEPSEEK_BASE_URL,
            response_mime_type=response_mime_type,
            temperature=temperature,
        )

    from app.core.genai_client import GenAIModel

    api_key = settings.GOOGLE_API_KEY.strip()
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is required when LLM_PROVIDER=gemini. "
            "Set DEEPSEEK_API_KEY to use the DeepSeek endpoint instead."
        )

    from google import genai

    client = genai.Client(api_key=api_key)
    return GenAIModel(
        client=client,
        model_name=model_name or settings.GEMINI_MODEL,
        system_instruction=system_instruction,
        response_mime_type=response_mime_type,
        temperature=temperature,
    )
