"""OpenAI-compatible client for Qwen models."""

import os
import json
from typing import Optional, Any
from dataclasses import dataclass

import requests


@dataclass
class ClientConfig:
    """Configuration for the API client."""
    base_url: str = "http://localhost:1234/v1"
    model: str = "qwen"
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    timeout: int = 120
    temperature: float = 0.7
    max_tokens: int = 4096

    def __post_init__(self):
        if self.api_key is None and self.api_key_env:
            self.api_key = os.environ.get(self.api_key_env)


class ModelClient:
    """Client for OpenAI-compatible chat completions endpoint."""

    def __init__(self, config: ClientConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.session = requests.Session()
        if config.api_key:
            self.session.headers["Authorization"] = f"Bearer {config.api_key}"
        self.session.headers["Content-Type"] = "application/json"

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Call the chat completions endpoint."""
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        
        if response_format:
            payload["response_format"] = response_format
        
        response = self.session.post(url, json=payload, timeout=self.config.timeout)
        response.raise_for_status()
        return response.json()

    def extract_content(self, response: dict[str, Any]) -> str:
        """Extract the assistant's message content from a response."""
        choices = response.get("choices", [])
        if not choices:
            raise ValueError("No choices in response")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if not content:
            raise ValueError("Empty content in response")
        return content

    def check_health(self) -> bool:
        """Check if the endpoint is reachable."""
        try:
            # Try a minimal request
            response = self.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
            )
            return bool(response.get("choices"))
        except Exception:
            return False
