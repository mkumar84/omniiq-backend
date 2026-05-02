import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

_client: Anthropic | None = None
MODEL = "claude-sonnet-4-6"


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


def call_claude(system_prompt: str, user_message: str, max_tokens: int = 512) -> str:
    response = _get_client().messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text
