"""Thin wrapper around the Codex API (OpenAI-compatible).

Reads credentials from ~/.mcp.json (the Codex MCP server config).
Falls back to environment variables OPENAI_API_KEY / OPENAI_BASE_URL.

In production, this calls the Codex API directly via HTTP. In tests, we pass a mock instead.
"""
import json
import os
from pathlib import Path

import requests


def _load_credentials():
    """Load Codex credentials from ~/.mcp.json or env vars."""
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("OPENAI_MODEL")

    # Try to read from ~/.mcp.json if env vars are missing
    mcp_json_path = Path.home() / ".mcp.json"
    if mcp_json_path.exists() and (not api_key or not base_url):
        try:
            with open(mcp_json_path, "r") as f:
                mcp_config = json.load(f)
            codex_cfg = mcp_config.get("mcpServers", {}).get("codex", {})
            env = codex_cfg.get("env", {})
            api_key = api_key or env.get("OPENAI_API_KEY")
            base_url = base_url or env.get("OPENAI_BASE_URL")
            model = model or env.get("OPENAI_MODEL", "gpt-5.4")
        except Exception:
            pass

    return api_key, base_url, model or "gpt-5.4"


class CodexLLMClient:
    """Calls Codex API (OpenAI-compatible chat completions) via HTTP."""

    def __init__(
        self,
        url: str = None,
        api_key: str = None,
        model: str = None,
        timeout: int = 60,
    ):
        cred_key, cred_url, cred_model = _load_credentials()
        self.api_key = api_key or cred_key
        base = url or cred_url or "https://codex.miaomiaocode.com/v1"
        # Ensure URL ends with /chat/completions
        if not base.endswith("/chat/completions"):
            base = base.rstrip("/") + "/chat/completions"
        self.url = base
        self.model = model or cred_model or "gpt-5.4"
        self.timeout = timeout

    def chat(self, prompt: str) -> str:
        """Send prompt to Codex API, return response text."""
        if not self.api_key:
            raise RuntimeError(
                "No Codex API key found. Set OPENAI_API_KEY env var or configure ~/.mcp.json"
            )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 300,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Use a fresh session each call to avoid connection-pool leaks.
        with requests.Session() as s:
            response = s.post(self.url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"]
