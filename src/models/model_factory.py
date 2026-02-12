"""
Model Factory - LLM Provider Abstraction
Supports Brain (Sonnet) and Muscle (Haiku) pattern
"""
import os
import json
import anthropic
from dotenv import load_dotenv

load_dotenv()


class ModelFactory:
    """Factory for creating LLM calls with cost tracking."""

    _instance = None
    _total_input_tokens = 0
    _total_output_tokens = 0
    _total_cost_usd = 0.0

    # Pricing per 1M tokens (approximate)
    PRICING = {
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._client = None
        return cls._instance

    @property
    def client(self):
        if self._client is None:
            api_key = os.getenv("ANTHROPIC_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_KEY not found in environment")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def call(self, model: str, prompt: str, system: str = "", max_tokens: int = 1024,
             temperature: float = 0.3, json_mode: bool = False) -> str:
        """Call an LLM model and return the text response."""
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)

        # Track usage
        usage = response.usage
        self._total_input_tokens += usage.input_tokens
        self._total_output_tokens += usage.output_tokens

        pricing = self.PRICING.get(model, {"input": 3.0, "output": 15.0})
        cost = (usage.input_tokens * pricing["input"] + usage.output_tokens * pricing["output"]) / 1_000_000
        self._total_cost_usd += cost

        text = response.content[0].text if response.content else ""

        if json_mode:
            # Try to extract JSON from response
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
