"""Minimal client for the external LLM used to synthesise neighbours.

Only the stdlib is used, so the pipeline adds no dependency to the project.
Any OpenAI-compatible chat-completions endpoint works (OpenAI, DeepSeek,
DashScope, vLLM's OpenAI server, Ollama's `/v1`, ...):

```bash
export FWT_LLM_BASE_URL=https://api.deepseek.com/v1
export FWT_LLM_API_KEY=sk-...
python fwt/scripts/build_neighbors.py --backend llm --llm-model deepseek-chat
```

`backend=template` needs no endpoint at all and is the offline default.
"""

from __future__ import annotations
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL_ENV = "FWT_LLM_BASE_URL"
DEFAULT_API_KEY_ENV = "FWT_LLM_API_KEY"


class LLMClient(Protocol):
    """Anything that turns a prompt into text."""

    def complete(self, prompt: str, system: Optional[str] = None) -> str: ...


class OpenAICompatibleClient:
    """Chat-completions client with retries, built on `urllib`."""

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.9,
        max_tokens: int = 2048,
        timeout: float = 120.0,
        max_retries: int = 4,
        backoff: float = 2.0,
    ):
        base_url = base_url or os.environ.get(DEFAULT_BASE_URL_ENV, "")
        if not base_url:
            raise ValueError(
                "No LLM endpoint configured: pass --llm-base-url or set "
                f"${DEFAULT_BASE_URL_ENV} (or use --backend template)."
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get(DEFAULT_API_KEY_ENV, "")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
            except (urllib.error.URLError, KeyError, TimeoutError, json.JSONDecodeError) as e:
                last_error = e
                wait = self.backoff ** (attempt + 1)
                logger.warning(
                    "LLM request failed (attempt %d/%d): %s - retrying in %.0fs",
                    attempt + 1, self.max_retries, e, wait,
                )
                time.sleep(wait)
        raise RuntimeError(f"LLM request failed after {self.max_retries} attempts") from last_error


class EchoClient:
    """Test double: replays canned replies in order."""

    def __init__(self, replies: List[str]):
        self.replies = list(replies)
        self.calls: List[Dict[str, Any]] = []

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        self.calls.append({"prompt": prompt, "system": system})
        if not self.replies:
            raise RuntimeError("EchoClient ran out of canned replies")
        return self.replies.pop(0)


def get_client(
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> LLMClient:
    return OpenAICompatibleClient(model=model, base_url=base_url, api_key=api_key, **kwargs)
