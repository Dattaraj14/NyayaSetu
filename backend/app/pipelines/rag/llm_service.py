import asyncio
import os
import random
import time
from typing import Optional

import structlog
from google import genai
from google.genai import types

from app.core.config import get_settings


logger = structlog.get_logger(__name__)

settings = get_settings()


class GeminiServiceUnavailableError(Exception):
    """Raised when all configured Gemini models are temporarily unavailable."""


class GeminiLegalLLM:
    """
    Gemini service for grounded legal-document question answering.

    Primary model:
        gemini-3.8-flash

    Fallback model:
        gemini-3.6-flash

    The service uses exponential backoff with jitter for transient
    Gemini API errors such as 429 and 5xx responses.
    """

    PRIMARY_MODEL = "gemini-3.8-flash"
    FALLBACK_MODEL = "gemini-3.6-flash"

    MAX_PRIMARY_ATTEMPTS = 3
    MAX_FALLBACK_ATTEMPTS = 3

    INITIAL_BACKOFF_SECONDS = 2.0
    MAX_BACKOFF_SECONDS = 8.0

    def __init__(self):
        self.settings = settings

        api_key = getattr(self.settings, "gemini_api_key", None)

        # Support both plain strings and Pydantic SecretStr.
        if hasattr(api_key, "get_secret_value"):
            api_key = api_key.get_secret_value()

        # Fallback to environment variable if necessary.
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured."
            )

        self.client = genai.Client(api_key=api_key)

        configured_model = getattr(
            self.settings,
            "llm_model",
            None,
        )

        self.primary_model = (
            configured_model
            if configured_model
            else self.PRIMARY_MODEL
        )

        self.fallback_model = self.FALLBACK_MODEL

        logger.info(
            "gemini_service_initialized",
            primary_model=self.primary_model,
            fallback_model=self.fallback_model,
        )

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        """
        Determine whether an exception represents a transient API failure.

        Retryable:
            408
            429
            500
            502
            503
            504

        Non-retryable client/configuration errors are returned immediately.
        """

        status_code: Optional[int] = getattr(
            exc,
            "status_code",
            None,
        )

        if status_code is None:
            status_code = getattr(
                exc,
                "code",
                None,
            )

        if isinstance(status_code, int):
            return status_code in {
                408,
                429,
                500,
                502,
                503,
                504,
            }

        message = str(exc).upper()

        retryable_codes = (
            "408",
            "429",
            "500",
            "502",
            "503",
            "504",
            "RESOURCE_EXHAUSTED",
            "UNAVAILABLE",
            "TIMEOUT",
            "TIMED OUT",
            "INTERNAL",
            "BAD GATEWAY",
            "GATEWAY TIMEOUT",
        )

        return any(
            code in message
            for code in retryable_codes
        )

    @classmethod
    def _backoff_delay(cls, retry_number: int) -> float:
        """
        Calculate exponential backoff with small random jitter.

        retry_number:
            1 -> approximately 2 seconds
            2 -> approximately 4 seconds
            3 -> approximately 8 seconds
        """

        exponential_delay = min(
            cls.INITIAL_BACKOFF_SECONDS
            * (2 ** (retry_number - 1)),
            cls.MAX_BACKOFF_SECONDS,
        )

        jitter = random.uniform(0.0, 0.5)

        return exponential_delay + jitter

    def _generate_once(
        self,
        model_name: str,
        prompt: str,
    ) -> str:
        """
        Perform exactly one Gemini API request.
        """

        logger.info(
            "gemini_request_started",
            model=model_name,
        )

        response = self.client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
                temperature=0.2,
            ),
        )

        text = getattr(response, "text", None)

        if not text or not text.strip():
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return text.strip()

    def _generate_model(
        self,
        model_name: str,
        prompt: str,
        max_attempts: int,
    ) -> str:
        """
        Generate content with exponential backoff for transient errors.
        """

        last_exception: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            logger.info(
                "gemini_request_attempt",
                model=model_name,
                attempt=attempt,
                max_attempts=max_attempts,
            )

            try:
                start_time = time.perf_counter()

                result = self._generate_once(
                    model_name=model_name,
                    prompt=prompt,
                )

                elapsed = (
                    time.perf_counter() - start_time
                )

                logger.info(
                    "gemini_request_success",
                    model=model_name,
                    attempt=attempt,
                    elapsed_seconds=round(elapsed, 3),
                )

                return result

            except Exception as exc:
                last_exception = exc

                retryable = self._is_retryable_error(
                    exc
                )

                logger.warning(
                    "gemini_api_error",
                    model=model_name,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    retryable=retryable,
                    status_code=getattr(
                        exc,
                        "status_code",
                        None,
                    ),
                    message=str(exc),
                )

                # Do not retry permanent errors.
                if not retryable:
                    raise

                # No more retries available.
                if attempt >= max_attempts:
                    break

                delay = self._backoff_delay(attempt)

                logger.info(
                    "gemini_retry_scheduled",
                    model=model_name,
                    attempt=attempt,
                    next_attempt=attempt + 1,
                    delay_seconds=round(delay, 2),
                )

                time.sleep(delay)

        if last_exception:
            raise last_exception

        raise GeminiServiceUnavailableError(
            f"{model_name} failed without an exception."
        )

    def _generate_with_fallback(
        self,
        prompt: str,
    ) -> str:
        """
        Try the primary Gemini model first.

        If it experiences transient availability problems,
        use the fallback model.
        """

        primary_exception: Optional[Exception] = None

        try:
            return self._generate_model(
                model_name=self.primary_model,
                prompt=prompt,
                max_attempts=self.MAX_PRIMARY_ATTEMPTS,
            )

        except Exception as exc:
            primary_exception = exc

            logger.warning(
                "gemini_fallback_triggered",
                failed_model=self.primary_model,
                fallback_model=self.fallback_model,
                status_code=getattr(
                    exc,
                    "status_code",
                    None,
                ),
                message=str(exc),
            )

        try:
            return self._generate_model(
                model_name=self.fallback_model,
                prompt=prompt,
                max_attempts=self.MAX_FALLBACK_ATTEMPTS,
            )

        except Exception as fallback_exception:

            logger.error(
                "all_gemini_models_unavailable",
                primary_model=self.primary_model,
                fallback_model=self.fallback_model,
                primary_error=str(primary_exception),
                fallback_error=str(fallback_exception),
            )

            raise GeminiServiceUnavailableError(
                "Gemini services are temporarily unavailable. "
                "Please try again in a moment."
            ) from fallback_exception

    async def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Async public interface used by the chat service.

        Gemini's synchronous client call is executed in a worker
        thread so the FastAPI event loop is not blocked.
        """

        logger.info(
            "llm_generation_started",
            primary_model=self.primary_model,
            fallback_model=self.fallback_model,
        )

        result = await asyncio.to_thread(
            self._generate_with_fallback,
            prompt,
        )

        logger.info(
            "llm_generation_complete",
            model=self.primary_model,
        )

        return result