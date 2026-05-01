"""
MediCareGuide Ollama Transport
==============================
Handles all HTTP communication with the local Ollama inference server.

  call_ollama()           — POST a prompt + history to /api/chat, return answer text
  split_system_and_user() — split build_prompt() output into system/user roles

Model: gemma4:e4b (set in MODEL constant) — 8B dense, 131K context, fully local, CPU-friendly

Dependencies:
    pip install requests
"""

import re
import requests

OLLAMA_URL  = "http://localhost:11434/api/chat"


def strip_thinking_block(text: str) -> str:
    """Remove Gemma 4 thinking block from response before displaying to user."""
    return re.sub(r'<\|channel>thought\n.*?<channel\|>', '', text, flags=re.DOTALL).strip()
MODEL_CLOUD   = "gemma4:31b-cloud"  # fast, Ollama zero-retention cloud
MODEL_LOCAL   = "gemma4:e4b"        # fully offline, slower
TIMEOUT_CLOUD = 120
TIMEOUT_LOCAL = 600

def call_ollama(system_prompt: str, user_message: str, history: list,
                mode: str = "cloud") -> str:
    """
    Send a message to Gemma 4 via the Ollama /api/chat endpoint.

    Uses the native messages format so Ollama handles the chat template.
    The MediCareGuide system prompt + plan data go into the system role;
    conversation history and the new question go into user/assistant turns.

    mode: "cloud" uses gemma4:31b-cloud (fast, Ollama-hosted, zero-retention);
          "local" uses gemma4:e4b (fully offline, private, slower).
    """
    model   = MODEL_CLOUD if mode == "cloud" else MODEL_LOCAL
    timeout = TIMEOUT_CLOUD if mode == "cloud" else TIMEOUT_LOCAL
    print(f"[OLLAMA] model={model}  timeout={timeout}s")

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return strip_thinking_block(resp.json()["message"]["content"])
    except requests.exceptions.ConnectionError:
        return (
            "[ERROR] Could not connect to Ollama. "
            "Make sure it's running: `ollama serve`"
        )
    except requests.exceptions.Timeout:
        return f"[ERROR] Ollama request timed out (>{timeout}s). Try switching to Fast mode or a shorter prompt."
    except Exception as e:
        return f"[ERROR] {e}"


def split_system_and_user(full_prompt: str) -> "tuple[str, str]":
    """
    build_prompt() returns a single string with SYSTEM_PROMPT prepended.
    Split it so we can send system content in the system role and the
    user question in the user role — giving Ollama cleaner context boundaries.

    Everything up to (but not including) "User question:" is system context.
    The "User question: ..." line becomes the user turn.
    """
    marker = "User question:"
    idx = full_prompt.rfind(marker)
    if idx == -1:
        return full_prompt, ""

    system_part = full_prompt[:idx].strip()
    user_part = full_prompt[idx + len(marker):].strip()
    return system_part, user_part
