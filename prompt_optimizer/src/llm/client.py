"""LLM API client wrappers with retry logic and PDF support.

Supported providers:
* **Gemini** (Google AI Studio free tier) — PDF extraction via native upload.
* **Groq** (free tier, Llama 3.1 8B) — text-only mutation and scoring.
"""

import json
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.llm.logger import LLMCallLogger


# ---------------------------------------------------------------------------
#  Abstract base
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Abstract base class for LLM API clients."""

    def __init__(
        self,
        model: str,
        api_key_env: str,
        temperature: float = 0,
        max_output_tokens: int = 8192,
        logger: LLMCallLogger | None = None,
    ) -> None:
        self.model = model
        self.api_key = os.environ.get(api_key_env, "")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.logger = logger
        self.provider = "unknown"

        if not self.api_key:
            raise ValueError(
                f"API key not found. Set the {api_key_env} environment variable."
            )

    @abstractmethod
    def extract(self, prompt: str, pdf_path: Path) -> dict:
        """Send a PDF + prompt and return structured JSON."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Text-only generation (for mutation and scoring)."""

    def judge(self, prompt: str) -> dict:
        """Convenience wrapper: generate text and parse as JSON."""
        raw = self.generate("You are a precise evaluation judge.", prompt)
        return _parse_json(raw)


# ---------------------------------------------------------------------------
#  Gemini client (PDF extraction)
# ---------------------------------------------------------------------------

class GeminiClient(LLMClient):
    """Google Gemini client via ``google-genai`` SDK."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.provider = "gemini"
        from google import genai
        from google.genai import types
        self._client = genai.Client(api_key=self.api_key)
        self._types = types
        self._current_key_idx = 0

    def _rotate_key(self) -> None:
        """Rotate to the next available API key in GOOGLE_API_KEY, GOOGLE_API_KEY_2, GOOGLE_API_KEY_3, GOOGLE_API_KEY_4."""
        from google import genai
        keys = [
            os.environ.get("GOOGLE_API_KEY"),
            os.environ.get("GOOGLE_API_KEY_2"),
            os.environ.get("GOOGLE_API_KEY_3"),
            os.environ.get("GOOGLE_API_KEY_4")
        ]
        keys = [k for k in keys if k]
        if len(keys) <= 1:
            return
        self._current_key_idx = (self._current_key_idx + 1) % len(keys)
        next_key = keys[self._current_key_idx]
        self.api_key = next_key
        self._client = genai.Client(api_key=next_key)
        print(f"  [INFO] Rotated Gemini API key to key index {self._current_key_idx + 1}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def extract(self, prompt: str, pdf_path: Path) -> dict:
        """Upload a PDF to Gemini and extract structured JSON."""
        start = time.time()
        error_msg: str | None = None
        raw_text = ""
        response = None
        try:
            pdf_bytes = Path(pdf_path).read_bytes()
            response = self._client.models.generate_content(
                model=self.model,
                contents=[
                    self._types.Part.from_bytes(
                        data=pdf_bytes,
                        mime_type="application/pdf",
                    ),
                    prompt,
                ],
                config=self._types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                ),
            )
            raw_text = response.text
            result = _parse_json_safe(raw_text)
            return result
        except Exception as exc:
            if "429" in str(exc) or "quota" in str(exc).lower() or "limit" in str(exc).lower():
                self._rotate_key()
            error_msg = str(exc)
            raise
        finally:
            latency = time.time() - start
            self._log_call(
                call_type="extract",
                prompt=prompt[:500],
                output=raw_text,
                latency=latency,
                error=error_msg,
                has_pdf=True,
                response=response,
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Text-only generation via Gemini."""
        start = time.time()
        error_msg: str | None = None
        raw_text = ""
        response = None
        try:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self._client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=self._types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                ),
            )
            raw_text = response.text
            return raw_text
        except Exception as exc:
            if "429" in str(exc) or "quota" in str(exc).lower() or "limit" in str(exc).lower():
                self._rotate_key()
            error_msg = str(exc)
            raise
        finally:
            latency = time.time() - start
            self._log_call(
                call_type="generate",
                prompt=f"{system_prompt[:200]}|{user_prompt[:300]}",
                output=raw_text,
                latency=latency,
                error=error_msg,
                has_pdf=False,
                response=response,
            )

    def _log_call(
        self,
        call_type: str,
        prompt: str,
        output: str,
        latency: float,
        error: str | None,
        has_pdf: bool,
        response: Any = None,
    ) -> None:
        """Log an LLM call if a logger is attached."""
        if self.logger is None:
            return
        input_tokens, output_tokens = 0, 0
        if response is not None:
            try:
                usage = response.usage_metadata
                input_tokens = (
                    getattr(usage, "prompt_token_count", 0) or 0
                )
                output_tokens = (
                    getattr(usage, "candidates_token_count", 0) or 0
                )
            except Exception:
                pass
        self.logger.log(
            call_type=call_type,
            provider=self.provider,
            model=self.model,
            input_prompt=prompt,
            output=output[:2000],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency,
            success=error is None,
            error=error,
            has_pdf=has_pdf,
        )


# ---------------------------------------------------------------------------
#  Groq client (mutation + scoring)
# ---------------------------------------------------------------------------

class GroqClient(LLMClient):
    """Groq client via the ``groq`` SDK (OpenAI-compatible)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.provider = "groq"

        from groq import Groq

        self._client = Groq(api_key=self.api_key)

    def extract(self, prompt: str, pdf_path: Path) -> dict:
        """Groq does not support PDF input; raises an error."""
        raise NotImplementedError(
            "Groq does not support PDF input. Use Gemini for extraction."
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Text generation via Groq chat completion.

        Args:
            system_prompt: System-level instructions.
            user_prompt: The user's message.

        Returns:
            The assistant's text response.
        """
        start = time.time()
        error_msg: str | None = None
        raw_text = ""
        completion = None

        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
            )
            raw_text = completion.choices[0].message.content or ""
            return raw_text
        except Exception as exc:
            error_msg = str(exc)
            raise
        finally:
            latency = time.time() - start
            input_tokens, output_tokens = 0, 0
            if completion is not None and hasattr(completion, "usage"):
                usage = completion.usage
                input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or 0

            if self.logger is not None:
                self.logger.log(
                    call_type="generate",
                    provider=self.provider,
                    model=self.model,
                    input_prompt=f"{system_prompt[:200]}|{user_prompt[:300]}",
                    output=raw_text[:2000],
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_seconds=latency,
                    success=error_msg is None,
                    error=error_msg,
                    has_pdf=False,
                )


# ---------------------------------------------------------------------------
#  Factory
# ---------------------------------------------------------------------------

def create_client(
    provider: str,
    model: str,
    api_key_env: str,
    temperature: float = 0,
    max_output_tokens: int = 8192,
    logger: LLMCallLogger | None = None,
) -> LLMClient:
    """Create an LLM client for the specified provider.

    Args:
        provider: ``'gemini'`` or ``'groq'``.
        model: Model identifier (e.g. ``'gemini-1.5-flash'``).
        api_key_env: Name of the environment variable holding the API key.
        temperature: Sampling temperature.
        max_output_tokens: Maximum tokens in the response.
        logger: Optional call logger.

    Returns:
        An ``LLMClient`` subclass instance.
    """
    classes = {
        "gemini": GeminiClient,
        "groq": GroqClient,
    }

    cls = classes.get(provider.lower())
    if cls is None:
        raise ValueError(
            f"Unknown provider '{provider}'. Supported: {list(classes)}"
        )

    return cls(
        model=model,
        api_key_env=api_key_env,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        logger=logger,
    )


# ---------------------------------------------------------------------------
#  JSON parsing helper
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    """Extract and parse JSON from an LLM response.

    Handles responses wrapped in markdown code fences.

    Args:
        text: Raw LLM output string.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If no valid JSON can be extracted.
    """
    cleaned = text.strip()

    fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(fence_pattern, cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(cleaned[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}")


def _repair_truncated_json(raw: str) -> str:
    """Close open brackets in a truncated JSON string."""
    stack = []
    in_string = False
    escape_next = False

    for i, char in enumerate(raw):
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in "{[":
            stack.append(char)
        elif char in "}]":
            if stack:
                stack.pop()

    result = raw.rstrip().rstrip(",")
    for bracket in reversed(stack):
        result += "}" if bracket == "{" else "]"
    return result


def _parse_json_safe(raw: str) -> dict:
    """Parse JSON with truncation recovery fallback."""
    # Clean markdown fencing
    clean = re.sub(r"^```[a-zA-Z]*\n?", "", raw.strip())
    clean = re.sub(r"\n?```$", "", clean).strip()

    # Normal parse
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Recovery parse
    try:
        repaired = _repair_truncated_json(clean)
        result = json.loads(repaired)
        print("  [WARN] JSON was truncated -- recovered partial result")
        return result
    except Exception:
        pass

    print("  [ERROR] JSON parsing failed completely")
    return {}
