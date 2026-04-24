from meshagent.openai.proxy import proxy


def test_get_client_reads_base_url_from_environment(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example.test/v1")
    monkeypatch.setattr(proxy, "AsyncOpenAI", _FakeAsyncOpenAI)

    client = proxy.get_client()

    assert isinstance(client, _FakeAsyncOpenAI)
    assert captured["base_url"] == "https://env.example.test/v1"
    assert captured["default_headers"] == {"User-Agent": proxy.DEFAULT_USER_AGENT}


def test_get_client_explicit_base_url_overrides_environment(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example.test/v1")
    monkeypatch.setattr(proxy, "AsyncOpenAI", _FakeAsyncOpenAI)

    client = proxy.get_client(base_url="https://explicit.example.test/v1")

    assert isinstance(client, _FakeAsyncOpenAI)
    assert captured["base_url"] == "https://explicit.example.test/v1"
    assert captured["default_headers"] == {"User-Agent": proxy.DEFAULT_USER_AGENT}


def test_get_client_uses_meshagent_defaults_when_provider_env_missing(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MESHAGENT_API_URL", "https://api.example.test")
    monkeypatch.setenv("MESHAGENT_TOKEN", "meshagent-token")
    monkeypatch.setattr(proxy, "AsyncOpenAI", _FakeAsyncOpenAI)

    client = proxy.get_client()

    assert isinstance(client, _FakeAsyncOpenAI)
    assert captured["base_url"] == "https://api.example.test/openai/v1"
    assert captured["api_key"] == "meshagent-token"
    assert captured["default_headers"] == {"User-Agent": proxy.DEFAULT_USER_AGENT}


def test_get_client_uses_configured_user_agent(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(proxy, "AsyncOpenAI", _FakeAsyncOpenAI)

    client = proxy.get_client(user_agent=" custom-app/1.0 ")

    assert isinstance(client, _FakeAsyncOpenAI)
    assert captured["default_headers"] == {"User-Agent": "custom-app/1.0"}
