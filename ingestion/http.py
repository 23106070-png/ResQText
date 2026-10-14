"""HTTP layer shared by all adapters. Uses only the standard library. Adapters receive a `fetch` callable so
tests can replace the network completely."""
from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

import config
from ingestion.models import SourceError


@dataclass
class HttpRequest:
    url: str
    method: str = "GET"
    headers: dict = field(default_factory=dict)
    body: bytes | None = None
    timeout: float = 0


@dataclass
class HttpResponse:
    status: int
    headers: dict          # lower-cased keys
    body: bytes = b""

    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


Fetcher = Callable[[HttpRequest], HttpResponse]


def default_fetcher(req: HttpRequest) -> HttpResponse:
    """Real network call. Returns the response for any HTTP status (including 304/4xx/5xx); raises SourceError only
    for timeouts and connection problems."""
    headers = {"User-Agent": config.USER_AGENT, "Accept": "*/*", **req.headers}
    request = urllib.request.Request(req.url, data=req.body, method=req.method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=req.timeout or config.HTTP_TIMEOUT) as resp:
            return HttpResponse(resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read())
    except urllib.error.HTTPError as exc:  # 304, 403, 429, 5xx ...
        body = b""
        try:
            body = exc.read()
        except Exception:
            pass
        return HttpResponse(exc.code, {k.lower(): v for k, v in (exc.headers or {}).items()}, body)
    except (socket.timeout, TimeoutError) as exc:
        raise SourceError("timeout", "The request timed out.") from exc
    except urllib.error.URLError as exc:
        if isinstance(getattr(exc, "reason", None), (socket.timeout, TimeoutError)):
            raise SourceError("timeout", "The request timed out.") from exc
        raise SourceError("network", f"Could not connect ({type(exc.reason).__name__}).") from exc
    except OSError as exc:
        raise SourceError("network", f"Network error ({type(exc).__name__}).") from exc


def check_status(resp: HttpResponse, what: str) -> None:
    """Turn a non-success status into a classified SourceError (never leaks response bodies into messages)."""
    s = resp.status
    if 200 <= s < 300:
        return
    if s == 429:
        try:
            wait = float(resp.headers.get("retry-after", ""))
        except ValueError:
            wait = None
        raise SourceError("rate_limited", f"{what}: rate limit reached (HTTP 429).", s, wait)
    if s in (401, 403):
        raise SourceError("auth", f"{what}: access refused (HTTP {s}). Check credentials/approval or whether the "
                                  "service blocks your network.", s)
    if 500 <= s < 600:
        raise SourceError("http_error", f"{what}: the service reported a server error (HTTP {s}).", s)
    raise SourceError("http_error", f"{what}: unexpected HTTP {s}.", s)


def parse_json(resp: HttpResponse, what: str):
    try:
        return json.loads(resp.text())
    except (json.JSONDecodeError, ValueError) as exc:
        raise SourceError("invalid_response", f"{what}: the response was not valid JSON.") from exc
