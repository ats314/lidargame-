"""The downloader has to survive a public host dropping a connection.

A 3DEP tile is hundreds of megabytes from a host with no SLA. Before this,
one `Connection reset by peer` failed the whole command -- which is how a
docs-only commit turned the viewer deploy red on `main`.
"""
from __future__ import annotations

import io
import urllib.error

import pytest

from lidarworld.data import fetch


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def test_a_reset_connection_is_retried(tmp_path, monkeypatch):
    attempts = []
    slept = []

    def flaky(request, timeout=None):
        attempts.append(1)
        if len(attempts) < 3:
            raise urllib.error.URLError(OSError(104, "Connection reset by peer"))
        return FakeResponse(b"tile-bytes")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", flaky)
    dest = fetch.download("https://example.invalid/t.laz", tmp_path / "t.laz",
                          sleep=slept.append)

    assert dest.read_bytes() == b"tile-bytes"
    assert len(attempts) == 3
    assert slept == [2, 4], "backoff must grow, not hammer"
    assert not (tmp_path / "t.laz.part").exists(), "partial must not survive"


def test_retries_are_bounded_and_the_error_surfaces(tmp_path, monkeypatch):
    def always_resets(request, timeout=None):
        raise urllib.error.URLError(OSError(104, "Connection reset by peer"))

    monkeypatch.setattr(fetch.urllib.request, "urlopen", always_resets)
    with pytest.raises(urllib.error.URLError):
        fetch.download("https://example.invalid/t.laz", tmp_path / "t.laz",
                       attempts=2, sleep=lambda _: None)
    assert not (tmp_path / "t.laz.part").exists()


def test_an_http_error_is_not_retried(tmp_path, monkeypatch):
    """A 404 is the server answering. Repeating it just repeats it."""
    attempts = []

    def not_found(request, timeout=None):
        attempts.append(1)
        raise urllib.error.HTTPError("https://example.invalid/t.laz", 404,
                                     "Not Found", {}, None)

    monkeypatch.setattr(fetch.urllib.request, "urlopen", not_found)
    with pytest.raises(urllib.error.HTTPError):
        fetch.download("https://example.invalid/t.laz", tmp_path / "t.laz",
                       sleep=lambda _: None)
    assert len(attempts) == 1


def test_a_truncated_body_is_treated_as_a_failure(tmp_path, monkeypatch):
    """Short reads are the silent version of a reset: the file lands, wrong."""
    calls = []

    def short_then_whole(request, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            response = FakeResponse(b"half")
            response.headers = {"Content-Length": "999"}
            return response
        return FakeResponse(b"whole-tile")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", short_then_whole)
    dest = fetch.download("https://example.invalid/t.laz", tmp_path / "t.laz",
                          sleep=lambda _: None)
    assert dest.read_bytes() == b"whole-tile"
    assert len(calls) == 2


def test_an_existing_file_is_not_refetched(tmp_path, monkeypatch):
    dest = tmp_path / "t.laz"
    dest.write_bytes(b"already here")

    def explode(request, timeout=None):
        raise AssertionError("must not hit the network")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", explode)
    assert fetch.download("https://example.invalid/t.laz", dest) == dest
