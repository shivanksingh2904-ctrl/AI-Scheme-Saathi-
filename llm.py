"""Calls an OpenAI-compatible chat completions API using plain HTTP."""
from typing import Dict, List

import requests

from app import config


class LLMError(Exception):
    """A problem talking to the LLM service (safe to show to the user)."""


def generate(messages: List[Dict]) -> str:
    url = f"{config.LLM_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",  # authentication
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": config.LLM_TEMPERATURE,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=config.LLM_TIMEOUT)
    except requests.RequestException as e:
        raise LLMError("Could not reach the AI service. Please try again.") from e

    if response.status_code in (401, 403):
        raise LLMError("The AI service rejected the API key. Check your .env file.")
    if response.status_code == 429:
        raise LLMError("The AI service rate limit was hit. Please wait a bit and retry.")
    if response.status_code != 200:
        raise LLMError(f"The AI service returned an error (status {response.status_code}).")

    try:
        return response.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, ValueError) as e:
        raise LLMError("The AI service sent an unexpected response.") from e
