#!/usr/bin/env python3
"""Quick check that the Anthropic API key in .env is valid and has credits."""
import json
import os
import urllib.request
import urllib.error
from pathlib import Path

ENV_FILE = Path(__file__).parent / ".env"


def load_env() -> dict[str, str]:
    env = {}
    if not ENV_FILE.exists():
        return env
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip("\"'")
    return env


def check_anthropic(key: str) -> str:
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}],
        }).encode(),
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=10).read())
        return "OK" if d.get("type") == "message" else f"ERROR: {d.get('error', d)}"
    except urllib.error.HTTPError as e:
        d = json.loads(e.read())
        return f"ERROR: {d.get('error', d)}"
    except Exception as e:
        return f"ERROR: {e}"


def main() -> None:
    env = load_env()

    anthropic_key = env.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

    if anthropic_key:
        result = check_anthropic(anthropic_key)
        print(f"ANTHROPIC_API_KEY : {result}")
    else:
        print("ANTHROPIC_API_KEY : NOT SET")


if __name__ == "__main__":
    main()
