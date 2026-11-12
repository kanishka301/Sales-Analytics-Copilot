"""Gemini chat connection with a short timeout and one bounded retry."""
import time

from google import genai
from google.genai import types

from src.config import GEMINI_MODEL, GEMINI_TIMEOUT_MS

_client = None


def _get_client():
    """Reuse one HTTP client so each agent step does not rebuild connections."""
    global _client
    if _client is None:
        # With no SDK retry policy, each HTTP request is attempted once.
        # The small retry below is the only retry in this project.
        _client = genai.Client(
            http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS)
        )
    return _client


def llm(system: str, messages: list, retries: int = 1) -> str:
    """Send the conversation through Chat.send_message and return Gemini's text."""
    client = _get_client()
    history = [
        types.Content(
            role="model" if item["role"] == "assistant" else "user",
            parts=[types.Part(text=item["content"])],
        )
        for item in messages[:-1]
    ]
    last_message = messages[-1]["content"] if messages else ""
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0,
        response_mime_type="application/json",
        max_output_tokens=1200,
    )

    for attempt in range(min(max(int(retries), 0), 1) + 1):
        try:
            chat = client.chats.create(
                model=GEMINI_MODEL,
                config=config,
                history=history or None,
            )
            response = chat.send_message(last_message)
            answer = response.text or ""
            if not answer.strip():
                raise RuntimeError("Gemini returned an empty response.")
            return answer
        except Exception as error:
            status = getattr(error, "code", None) or getattr(error, "status_code", None)
            transient = status in (408, 429, 500, 502, 503, 504) or "timeout" in str(error).lower()
            if attempt >= min(max(int(retries), 0), 1) or not transient:
                raise
            time.sleep(1)
