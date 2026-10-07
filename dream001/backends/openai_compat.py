"""
DREAM-006 fallback tier: any OpenAI-compatible endpoint exposing top-k logprobs.

Top-k truncation systematically UNDERestimates entropy. Coverage (observed
probability mass) is returned per token so the bias is visible in every record.
This tier lags internal state; it is a fallback, not a substitute.
"""

from __future__ import annotations
import json, math, urllib.request
from typing import List, Tuple


class OpenAICompatBackend:
    def __init__(self, base_url: str, model: str, top_logprobs: int = 20,
                 max_tokens: int = 128, temperature: float = 0.0, api_key: str = "none"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.top_logprobs = top_logprobs
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key
        self.name = model

    def generate(self, prompt: str) -> Tuple[str, List[float], List[float]]:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        choice = data["choices"][0]
        text = choice["message"]["content"]
        entropies, coverage = [], []
        for tokinfo in choice.get("logprobs", {}).get("content", []):
            lps = [t["logprob"] for t in tokinfo.get("top_logprobs", [])]
            if not lps:
                continue
            ps = [math.exp(lp) for lp in lps]
            entropies.append(-sum(p * lp for p, lp in zip(ps, lps)))
            coverage.append(min(sum(ps), 1.0))
        return text, entropies, coverage
