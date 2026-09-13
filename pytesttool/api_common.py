"""Shared helpers for the lightweight HTTP smoke-test scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib import error, request

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9003
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.json"


def load_endpoint(config_path: str | Path) -> tuple[str, int]:
    host = DEFAULT_HOST
    port = DEFAULT_PORT
    try:
        with Path(config_path).open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        host = config.get("host") or config.get("ip") or host
        port = int(config.get("analyzerPort", port))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"[WARN] Failed to read config; using {host}:{port}: {exc}")
    return host, port


def base_url(config_path: str | Path, host: str | None, port: int | None) -> str:
    config_host, config_port = load_endpoint(config_path)
    return f"http://{host or config_host}:{port or config_port}"


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    http_request = request.Request(url, data=body, headers=headers, method=method)
    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        print(f"[ERROR] HTTP {exc.code}: {text}")
        return None
    except (error.URLError, TimeoutError, OSError) as exc:
        print(f"[ERROR] Request failed: {exc}")
        return None

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        print(f"[ERROR] Server returned non-JSON response: {text}")
        return None

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
