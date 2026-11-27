"""Multi-provider LLM service (Groq + Gemini) with failover, structured outputs, and streaming."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from collections.abc import AsyncIterator
from typing import Optional, TypeVar

import httpx
import structlog
from google import genai
from google.genai import types
from pydantic import BaseModel

from app.core.config import get_settings


logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_GROQ_CASCADE = (
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
)

DEFAULT_GEMINI_CASCADE = (
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
)

_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class GeminiServiceUnavailableError(Exception):
    """Raised when all configured LLM providers/models are unavailable."""


def _clean_json_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = _FENCE_PATTERN.sub("", cleaned).strip()
    return cleaned


class GeminiLegalLLM:
    """
    LLM service for grounded legal-document QA and structured summarization.

    Supports:
    - LLM_PROVIDER=groq   -> Groq primary cascade, Gemini automatic backup
    - LLM_PROVIDER=gemini -> Gemini primary cascade, Groq automatic backup
    """

    INITIAL_BACKOFF_SECONDS = 0.8
    MAX_BACKOFF_SECONDS = 3.0

    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = (self.settings.llm_provider or "gemini").strip().lower()

        gemini_key = getattr(self.settings, "gemini_api_key", None)
        if hasattr(gemini_key, "get_secret_value"):
            gemini_key = gemini_key.get_secret_value()
        self.gemini_api_key = gemini_key or os.getenv("GEMINI_API_KEY") or None

        groq_key = getattr(self.settings, "groq_api_key", None)
        if hasattr(groq_key, "get_secret_value"):
            groq_key = groq_key.get_secret_value()
        self.groq_api_key = groq_key or os.getenv("GROQ_API_KEY") or None

        if not self.gemini_api_key and not self.groq_api_key:
            raise RuntimeError(
                "Neither GROQ_API_KEY nor GEMINI_API_KEY is configured."
            )

        self.timeout_seconds = max(5, int(self.settings.llm_timeout_seconds))
        self.max_attempts = max(1, int(self.settings.llm_max_attempts))
        self.thinking_budget = self.settings.llm_thinking_budget

        self.client: Optional[genai.Client] = None
        if self.gemini_api_key:
            self.client = genai.Client(
                api_key=self.gemini_api_key,
                http_options=types.HttpOptions(
                    timeout=self.timeout_seconds * 1000,
                ),
            )

        # Build Groq cascade
        groq_cascade: list[str] = []
        if self.provider == "groq":
            for candidate in (
                self.settings.llm_model,
                self.settings.llm_fallback_model,
            ):
                if (
                    candidate
                    and not candidate.startswith("gemini")
                    and candidate not in groq_cascade
                ):
                    groq_cascade.append(candidate)
        for candidate in DEFAULT_GROQ_CASCADE:
            if candidate not in groq_cascade:
                groq_cascade.append(candidate)
        self.groq_cascade = groq_cascade

        # Build Gemini cascade
        gemini_cascade: list[str] = []
        if self.provider == "gemini":
            for candidate in (
                self.settings.llm_model,
                self.settings.llm_fallback_model,
            ):
                if (
                    candidate
                    and candidate.startswith("gemini")
                    and candidate not in gemini_cascade
                ):
                    gemini_cascade.append(candidate)
        for candidate in DEFAULT_GEMINI_CASCADE:
            if candidate not in gemini_cascade:
                gemini_cascade.append(candidate)
        self.gemini_cascade = gemini_cascade

        if self.provider == "groq" and self.groq_api_key:
            self.primary_model = f"groq/{self.groq_cascade[0]}"
            self.fallback_model = (
                f"groq/{self.groq_cascade[1]}"
                if len(self.groq_cascade) > 1
                else self.gemini_cascade[0]
            )
        else:
            self.primary_model = self.gemini_cascade[0]
            self.fallback_model = (
                self.gemini_cascade[1]
                if len(self.gemini_cascade) > 1
                else None
            )

        logger.info(
            "llm_service_initialized",
            provider=self.provider,
            primary_model=self.primary_model,
            groq_enabled=bool(self.groq_api_key),
            gemini_enabled=bool(self.gemini_api_key),
            timeout_seconds=self.timeout_seconds,
        )

    # ========================================================================
    # GROQ EXECUTION (OpenAI-compatible REST API via httpx)
    # ========================================================================

    def _call_groq_once(
        self,
        model_name: str,
        prompt: str,
        *,
        temperature: float = 0.2,
        response_schema: type[BaseModel] | None = None,
        use_json_schema_mode: bool = True,
    ) -> str:
        if not self.groq_api_key:
            raise GeminiServiceUnavailableError("GROQ_API_KEY is not configured.")

        messages: list[dict[str, str]] = []
        if response_schema is not None:
            schema_json = json.dumps(response_schema.model_json_schema())
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "You are a precise legal AI. Respond ONLY with a valid JSON object "
                        "matching the schema below. Do not include markdown fences or extra text.\n"
                        f"JSON_SCHEMA:\n{schema_json}"
                    ),
                }
            )
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }

        if model_name.startswith("openai/gpt-oss-"):
            payload["reasoning_effort"] = "low"

        if response_schema is not None:
            if use_json_schema_mode:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_schema.__name__,
                        "strict": False,
                        "schema": response_schema.model_json_schema(),
                    },
                }
            else:
                payload["response_format"] = {"type": "json_object"}

        logger.info("groq_request_started", model=model_name)
        start_time = time.perf_counter()

        with httpx.Client(timeout=self.timeout_seconds) as http_client:
            resp = http_client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

            # If strict/json_schema mode triggers a 400 on a schema edge case, retry with json_object.
            if (
                resp.status_code == 400
                and response_schema is not None
                and use_json_schema_mode
            ):
                logger.warning(
                    "groq_json_schema_fallback_to_json_object",
                    model=model_name,
                    detail=resp.text[:250],
                )
                return self._call_groq_once(
                    model_name,
                    prompt,
                    temperature=temperature,
                    response_schema=response_schema,
                    use_json_schema_mode=False,
                )

            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Groq HTTP {resp.status_code}: {resp.text[:350]}"
                )

            data = resp.json()

        content = str(data["choices"][0]["message"]["content"] or "").strip()
        if not content:
            raise RuntimeError(f"Groq model {model_name} returned empty content.")

        if response_schema is not None:
            content = _clean_json_text(content)
            # Validate immediately so malformed JSON hops to the next cascade model.
            response_schema.model_validate_json(content)

        logger.info(
            "groq_request_success",
            model=model_name,
            elapsed_seconds=round(time.perf_counter() - start_time, 3),
        )
        return content

    def _try_groq_cascade(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        response_schema: type[BaseModel] | None = None,
        errors: dict[str, str],
    ) -> Optional[tuple[str, str]]:
        if not self.groq_api_key:
            return None

        for model_name in self.groq_cascade:
            try:
                text = self._call_groq_once(
                    model_name,
                    prompt,
                    temperature=temperature,
                    response_schema=response_schema,
                )
                return text, f"groq/{model_name}"
            except Exception as exc:
                errors[f"groq/{model_name}"] = str(exc)
                logger.warning(
                    "groq_api_error",
                    model=model_name,
                    message=str(exc),
                )
        return None

    # ========================================================================
    # GEMINI EXECUTION
    # ========================================================================

    @staticmethod
    def _should_failover_immediately(exc: Exception) -> bool:
        upper = str(exc).upper()
        return any(
            marker in upper
            for marker in (
                "503",
                "UNAVAILABLE",
                "429",
                "RESOURCE_EXHAUSTED",
                "404",
                "NOT_FOUND",
                "504",
                "DEADLINE_EXCEEDED",
            )
        )

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        status_code: Optional[int] = getattr(exc, "status_code", None)
        if status_code is None:
            status_code = getattr(exc, "code", None)
        if isinstance(status_code, int):
            return status_code in {408, 500, 502}
        upper = str(exc).upper()
        return any(
            marker in upper
            for marker in ("408", "500", "502", "TIMEOUT", "TIMED OUT", "INTERNAL")
        )

    @staticmethod
    def _is_thinking_unsupported_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "thinking" in message and (
            "400" in message
            or "invalid" in message
            or "unsupported" in message
            or "not supported" in message
        )

    @classmethod
    def _backoff_delay(cls, retry_number: int) -> float:
        exponential_delay = min(
            cls.INITIAL_BACKOFF_SECONDS * (2 ** (retry_number - 1)),
            cls.MAX_BACKOFF_SECONDS,
        )
        return exponential_delay + random.uniform(0.0, 0.25)

    def _build_gemini_config(
        self,
        *,
        temperature: float = 0.2,
        response_schema: type[BaseModel] | None = None,
        use_thinking_budget: bool = True,
    ) -> types.GenerateContentConfig:
        kwargs: dict = {
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            "temperature": temperature,
        }

        if response_schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = response_schema

        if (
            use_thinking_budget
            and self.thinking_budget is not None
            and hasattr(types, "ThinkingConfig")
        ):
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )

        return types.GenerateContentConfig(**kwargs)

    def _call_gemini_once(
        self,
        model_name: str,
        prompt: str,
        *,
        temperature: float = 0.2,
        response_schema: type[BaseModel] | None = None,
    ) -> str:
        if self.client is None:
            raise GeminiServiceUnavailableError("GEMINI_API_KEY is not configured.")

        logger.info("gemini_request_started", model=model_name)
        start_time = time.perf_counter()

        config = self._build_gemini_config(
            temperature=temperature,
            response_schema=response_schema,
            use_thinking_budget=True,
        )

        try:
            response = self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            if (
                self.thinking_budget is not None
                and self._is_thinking_unsupported_error(exc)
            ):
                fallback_config = self._build_gemini_config(
                    temperature=temperature,
                    response_schema=response_schema,
                    use_thinking_budget=False,
                )
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=fallback_config,
                )
            else:
                raise

        text = getattr(response, "text", None)
        if not text or not text.strip():
            raise RuntimeError("Gemini returned an empty response.")

        cleaned = _clean_json_text(text) if response_schema is not None else text.strip()
        if response_schema is not None:
            response_schema.model_validate_json(cleaned)

        logger.info(
            "gemini_request_success",
            model=model_name,
            elapsed_seconds=round(time.perf_counter() - start_time, 3),
        )
        return cleaned

    def _try_gemini_cascade(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        response_schema: type[BaseModel] | None = None,
        errors: dict[str, str],
    ) -> Optional[tuple[str, str]]:
        if self.client is None:
            return None

        for model_name in self.gemini_cascade:
            try:
                text = self._call_gemini_once(
                    model_name,
                    prompt,
                    temperature=temperature,
                    response_schema=response_schema,
                )
                return text, model_name
            except Exception as exc:
                errors[model_name] = str(exc)
                logger.warning(
                    "gemini_api_error",
                    model=model_name,
                    failover_immediately=self._should_failover_immediately(exc),
                    message=str(exc),
                )
        return None

    # ========================================================================
    # UNIFIED ROUTING
    # ========================================================================

    def _generate_with_fallback(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        response_schema: type[BaseModel] | None = None,
    ) -> tuple[str, str]:
        errors: dict[str, str] = {}

        if self.provider == "groq" and self.groq_api_key:
            order = (self._try_groq_cascade, self._try_gemini_cascade)
        else:
            order = (self._try_gemini_cascade, self._try_groq_cascade)

        for provider_fn in order:
            result = provider_fn(
                prompt,
                temperature=temperature,
                response_schema=response_schema,
                errors=errors,
            )
            if result is not None:
                return result

        logger.error("all_llm_models_unavailable", errors=errors)
        raise GeminiServiceUnavailableError(
            "All configured LLM providers are temporarily unavailable. Please retry in a moment."
        )

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
    ) -> str:
        """Async text generation."""
        text, model_used = await asyncio.to_thread(
            self._generate_with_fallback,
            prompt,
            temperature=temperature,
        )
        logger.info("llm_generation_complete", model=model_used)
        return text

    async def generate_structured(
        self,
        prompt: str,
        response_schema: type[T],
        *,
        temperature: float = 0.1,
    ) -> T:
        """Async structured output generation validated against a Pydantic schema."""
        raw_json, model_used = await asyncio.to_thread(
            self._generate_with_fallback,
            prompt,
            temperature=temperature,
            response_schema=response_schema,
        )
        logger.info(
            "llm_structured_generation_complete",
            model=model_used,
            schema=response_schema.__name__,
        )
        return response_schema.model_validate_json(raw_json)

    async def generate_stream(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        """Async chunked generation across configured providers."""
        text = await self.generate(prompt, temperature=temperature)
        words = text.split(" ")
        chunk_size = 8
        for i in range(0, len(words), chunk_size):
            piece = " ".join(words[i : i + chunk_size])
            if i + chunk_size < len(words):
                piece += " "
            if piece:
                yield piece