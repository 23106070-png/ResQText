"""Fake network for tests: no test ever touches the internet."""
from __future__ import annotations

import json
from pathlib import Path

from ingestion.http import HttpRequest, HttpResponse
from ingestion.models import SourceError

FIX = Path(__file__).parent / "fixtures"


def fixture(name: str) -> bytes:
    return (FIX / name).read_bytes()


def R(status=200, body=b"", **headers) -> HttpResponse:
    if isinstance(body, (dict, list)):
        body = json.dumps(body).encode()
    if isinstance(body, str):
        body = body.encode()
    return HttpResponse(status, {k.lower().replace("_", "-"): v for k, v in headers.items()}, body)


class FakeNet:
    """routes: substring of URL -> sequence of responses/exceptions (the last one repeats)."""

    def __init__(self):
        self.routes, self.calls = [], []

    def add(self, substr: str, *responses, method: str | None = None):
        self.routes.append([substr, method, list(responses)])
        return self

    def __call__(self, req: HttpRequest):
        self.calls.append(req)
        for substr, method, seq in self.routes:
            if substr in req.url and (method is None or method == req.method):
                item = seq.pop(0) if len(seq) > 1 else seq[0]
                if isinstance(item, Exception):
                    raise item
                return item
        raise AssertionError(f"Unexpected request in test: {req.method} {req.url}")

    def urls(self, substr: str = ""):
        return [c.url for c in self.calls if substr in c.url]


def sachet_net(rss="sachet_rss.xml") -> FakeNet:
    net = FakeNet().add("rss_india.xml", R(200, fixture(rss), etag='"feed-1"'))
    net.add("identifier=1791731455698013", R(200, fixture("sachet_cap_flood.xml"), etag='"c1"'))
    net.add("identifier=1791732541708028", R(200, fixture("sachet_cap_jaipur.xml"), etag='"c2"'))
    net.add("identifier=1791720898008028", R(200, fixture("sachet_cap_hindi.xml"), etag='"c3"'))
    net.add("identifier=1000000000000001", R(200, fixture("sachet_cap_test.xml"), etag='"c4"'))
    return net


class Clock:
    def __init__(self, dt):
        self.dt = dt

    def __call__(self):
        return self.dt

    def advance(self, **kw):
        from datetime import timedelta
        self.dt = self.dt + timedelta(**kw)
