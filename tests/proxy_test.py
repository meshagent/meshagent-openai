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
