"""
utils.py
--------
Shared utility: LLMClient wraps Google Gemini's free-tier API.

demo_mode="offline" -> never calls the network. All agent classes fall back
    to deterministic, rule-based logic. Use this for development/testing
    and for grading/demo environments with no API key.
demo_mode="live"    -> calls the Gemini API (gemini-2.0-flash by default,
    a free-tier model) for the two steps that actually benefit from an LLM:
    natural-language requirement extraction and turning a worker's raw
    search results into a short natural-language recommendation blurb.

Get a free Gemini API key at https://aistudio.google.com/app/apikey
and put it in .env as GEMINI_API_KEY=...
"""

import os
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class LLMClient:
    """Thin wrapper around Gemini's generateContent endpoint.

    Kept intentionally small: one method, `complete(system, user)`, that
    every worker/critic/manager calls the same way regardless of provider.
    Swapping Gemini for another provider later only means editing this file.
    """

    def __init__(self, demo_mode=None, model=None, api_key=None):
        self.demo_mode = demo_mode or os.getenv("DEMO_MODE", "offline")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        self._client = None
        if self.demo_mode == "live":
            if not self.api_key:
                raise RuntimeError(
                    "DEMO_MODE=live but no GEMINI_API_KEY was found. "
                    "Set it in .env or export it, or switch DEMO_MODE=offline."
                )
            try:
                import google.generativeai as genai
            except ImportError as e:
                raise RuntimeError(
                    "google-generativeai is not installed. "
                    "Run: pip install google-generativeai"
                ) from e
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model)

    def complete(self, system: str, user: str, json_mode: bool = True) -> str:
        """Send one prompt to Gemini and return the raw text response.

        Callers are responsible for parsing the response (usually as JSON).
        Raises on any API failure -- callers should catch and fall back to
        offline/rule-based logic rather than crash the pipeline.
        """
        if self.demo_mode != "live":
            raise RuntimeError("complete() called while demo_mode != 'live'")

        prompt = f"{system}\n\n{user}"
        if json_mode:
            prompt += "\n\nRespond with ONLY valid JSON. No markdown fences, no preamble."

        generation_config = {"response_mime_type": "application/json"} if json_mode else {}
        response = self._client.generate_content(
            prompt,
            generation_config=generation_config,
        )
        return response.text

    def complete_json(self, system: str, user: str) -> dict:
        """Convenience wrapper: complete() + strip fences + json.loads()."""
        raw = self.complete(system, user, json_mode=True)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return json.loads(cleaned.strip())
