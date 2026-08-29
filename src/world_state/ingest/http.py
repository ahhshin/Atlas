from __future__ import annotations

import time
from typing import Any

import httpx


def get_json_bytes(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    retries: int = 3,
    backoff_seconds: float = 0.5,
) -> tuple[bytes, str]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.get(url, params=params, headers=headers)
            response.raise_for_status()
            response.json()
            return response.content, str(response.url)
        except (httpx.HTTPError, ValueError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(backoff_seconds * (2**attempt))
    assert last_error is not None
    raise last_error
