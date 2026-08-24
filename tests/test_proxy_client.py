from __future__ import annotations

import asyncio

import httpx
import pytest

from dressage.proxy.proxy_client import ProxyClient


def test_proxy_client_sends_default_headers_and_reads_capabilities():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/integration/capabilities":
            return httpx.Response(
                200,
                json={
                    "schema_version": "dressage.proxy.integration/v1",
                    "current_weight_version": "weights-v7",
                },
            )
        return httpx.Response(200, json={"ok": True})

    async def run_test() -> dict:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        ) as http_client:
            client = ProxyClient(
                "http://proxy.test/",
                client=http_client,
                default_headers={
                    "Authorization": "Bearer proxy-secret",
                    "X-Client": "harbor",
                },
            )
            await client.chat_completions(
                {"model": "test", "messages": []},
                session_id="session-1",
                instance_id="instance-1",
            )
            return await client.capabilities()

    capabilities = asyncio.run(run_test())

    assert capabilities["current_weight_version"] == "weights-v7"
    assert [request.url.path for request in requests] == [
        "/v1/chat/completions",
        "/integration/capabilities",
    ]
    for request in requests:
        assert request.headers["authorization"] == "Bearer proxy-secret"
        assert request.headers["x-client"] == "harbor"
    assert requests[0].headers["x-session-id"] == "session-1"
    assert requests[0].headers["x-instance-id"] == "instance-1"


def test_proxy_client_default_timeout_is_bounded(monkeypatch):
    monkeypatch.delenv("DRESSAGE_PROXY_REQUEST_TIMEOUT_SEC", raising=False)

    async def run_test() -> tuple[float | None, float | None]:
        client = ProxyClient("http://proxy.test")
        try:
            return client._client.timeout.connect, client._client.timeout.read
        finally:
            await client.close()

    connect, read = asyncio.run(run_test())

    assert connect == 10.0
    assert read == 300.0


def test_proxy_client_uses_request_timeout_from_environment(monkeypatch):
    monkeypatch.setenv("DRESSAGE_PROXY_REQUEST_TIMEOUT_SEC", "1800")

    async def run_test() -> tuple[float | None, float | None]:
        client = ProxyClient("http://proxy.test")
        try:
            return client._client.timeout.connect, client._client.timeout.read
        finally:
            await client.close()

    connect, read = asyncio.run(run_test())

    assert connect == 10.0
    assert read == 1800.0


def test_proxy_client_explicit_timeout_overrides_environment(monkeypatch):
    monkeypatch.setenv("DRESSAGE_PROXY_REQUEST_TIMEOUT_SEC", "invalid")

    async def run_test() -> tuple[float | None, float | None]:
        client = ProxyClient(
            "http://proxy.test",
            timeout=httpx.Timeout(45.0, connect=5.0),
        )
        try:
            return client._client.timeout.connect, client._client.timeout.read
        finally:
            await client.close()

    connect, read = asyncio.run(run_test())

    assert connect == 5.0
    assert read == 45.0


def test_proxy_client_injected_client_does_not_parse_timeout_environment(
    monkeypatch,
):
    monkeypatch.setenv("DRESSAGE_PROXY_REQUEST_TIMEOUT_SEC", "invalid")

    async def run_test() -> None:
        async with httpx.AsyncClient() as http_client:
            client = ProxyClient("http://proxy.test", client=http_client)
            assert client._client is http_client

    asyncio.run(run_test())


@pytest.mark.parametrize(
    "value",
    ["0", "-1", "nan", "inf", "-inf", "not-a-number"],
)
def test_proxy_client_rejects_invalid_request_timeout_environment(
    monkeypatch,
    value,
):
    monkeypatch.setenv("DRESSAGE_PROXY_REQUEST_TIMEOUT_SEC", value)

    with pytest.raises(
        ValueError,
        match="DRESSAGE_PROXY_REQUEST_TIMEOUT_SEC must be a positive finite number",
    ):
        ProxyClient("http://proxy.test")


def test_proxy_client_discards_session_context_with_delete_and_checks_status():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(503, json={"detail": "unavailable"})

    async def run_test() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        ) as http_client:
            client = ProxyClient(
                "http://proxy.test",
                client=http_client,
                default_headers={"Authorization": "Bearer proxy-secret"},
            )
            with pytest.raises(httpx.HTTPStatusError):
                await client.discard_session_context("session-1")

    asyncio.run(run_test())

    assert len(requests) == 1
    assert requests[0].method == "DELETE"
    assert requests[0].url.path == "/v1/session/context/session-1"
    assert requests[0].headers["authorization"] == "Bearer proxy-secret"
