"""Access control for write actions (review, refresh, delete). Reads stay public; writes fail CLOSED.

local       + no secret set  -> single-user demo: writes allowed (the UI shows a visible 'local mode' banner)
any env     + secret set     -> writes require the passcode (dashboard) / X-API-Key header (API)
production  + no secret set  -> writes are DISABLED until a secret is configured
"""
from __future__ import annotations

import hmac

import config


def _mode(secret: str) -> str:
    if secret:
        return "secret"
    return "open_local" if config.APP_ENV == "local" else "locked"


def dashboard_mode() -> str:
    return _mode(config.ADMIN_PASSCODE)


def api_mode() -> str:
    return _mode(config.API_KEY)


def check_passcode(candidate: str) -> bool:
    return bool(config.ADMIN_PASSCODE) and hmac.compare_digest(candidate.encode(), config.ADMIN_PASSCODE.encode())


def check_api_key(candidate: str | None) -> bool:
    return bool(config.API_KEY and candidate) and hmac.compare_digest(candidate.encode(), config.API_KEY.encode())


def api_write_decision(candidate: str | None) -> tuple[bool, int, str]:
    """(allowed, http_status_if_denied, message)"""
    mode = api_mode()
    if mode == "open_local":
        return True, 200, ""
    if mode == "locked":
        return False, 503, "Write endpoints are disabled: set RESQTEXT_API_KEY to enable them."
    if check_api_key(candidate):
        return True, 200, ""
    return False, 401, "A valid X-API-Key header is required."
