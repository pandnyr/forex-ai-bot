"""
Model Factory - LLM Provider Abstraction via OpenRouter
OpenRouter provides access to Claude and other models via OpenAI-compatible API
"""
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ModelFactory:
    """Factory for creating LLM calls via OpenRouter with cost tracking."""

    _instance = None
    _total_input_tokens = 0
    _total_output_tokens = 0
    _total_cost_usd = 0.0

    # Pricing per 1M tokens (OpenRouter pricing, approximate)
    PRICING = {
        "anthropic/claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "anthropic/claude-3.5-sonnet-20241022": {"input": 3.0, "output": 15.0},
        "anthropic/claude-3.5-haiku-20241022": {"input": 0.80, "output": 4.0},
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._client = None
        return cls._instance

    @property
    def client(self):
        if self._client is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not found in environment")
            self._client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=api_key,
            )
        return self._client

    def call(self, model: str, prompt: str, system: str = "", max_tokens: int = 1024,
             temperature: float = 0.3, json_mode: bool = False) -> str:
        """Call an LLM model via OpenRouter and return the text response."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        response = self.client.chat.completions.create(**kwargs)

        # Track usage
        usage = response.usage
        if usage:
            self._total_input_tokens += usage.prompt_tokens or 0
            self._total_output_tokens += usage.completion_tokens or 0

            pricing = self.PRICING.get(model, {"input": 3.0, "output": 15.0})
            cost = ((usage.prompt_tokens or 0) * pricing["input"] +
                    (usage.completion_tokens or 0) * pricing["output"]) / 1_000_000
            self._total_cost_usd += cost

        text = response.choices[0].message.content if response.choices else ""

        if json_mode:
            # Try to extract JSON object from response
            try:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start != -1 and end > start:
                    return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
            # Try array
            try:
                start = text.find("[")
                end = text.rfind("]") + 1
                if start != -1 and end > start:
                    return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
            return text

        return text

    def call_brain(self, prompt: str, **kwargs) -> str:
        """Call the Brain model (Sonnet) for critical decisions."""
        from ..config import BRAIN_MODEL
        return self.call(BRAIN_MODEL, prompt, **kwargs)

    def call_muscle(self, prompt: str, **kwargs) -> str:
        """Call the Muscle model (Haiku) for routine monitoring."""
        from ..config import MUSCLE_MODEL
        return self.call(MUSCLE_MODEL, prompt, **kwargs)

    @property
    def total_cost(self) -> float:
        return self._total_cost_usd

    @property
    def usage_summary(self) -> dict:
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_cost_usd": round(self._total_cost_usd, 4),
        }

    def reset_tracking(self):
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost_usd = 0.0
