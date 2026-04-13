from agent.smart_model_routing import choose_cheap_model_route


_BASE_CONFIG = {
    "enabled": True,
    "cheap_model": {
        "provider": "openrouter",
        "model": "google/gemini-2.5-flash",
    },
}


def test_returns_none_when_disabled():
    cfg = {**_BASE_CONFIG, "enabled": False}
    assert choose_cheap_model_route("what time is it in tokyo?", cfg) is None


def test_routes_short_simple_prompt():
    result = choose_cheap_model_route("what time is it in tokyo?", _BASE_CONFIG)
    assert result is not None
    assert result["provider"] == "openrouter"
    assert result["model"] == "google/gemini-2.5-flash"
    assert result["routing_reason"] == "simple_turn"


def test_skips_long_prompt():
    prompt = "please summarize this carefully " * 20
    assert choose_cheap_model_route(prompt, _BASE_CONFIG) is None


def test_skips_code_like_prompt():
    prompt = "debug this traceback: ```python\nraise ValueError('bad')\n```"
    assert choose_cheap_model_route(prompt, _BASE_CONFIG) is None


def test_skips_tool_heavy_prompt_keywords():
    prompt = "implement a patch for this docker error"
    assert choose_cheap_model_route(prompt, _BASE_CONFIG) is None


def test_resolve_turn_route_falls_back_to_primary_when_route_runtime_cannot_be_resolved(monkeypatch):
    from agent.smart_model_routing import resolve_turn_route

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad route")),
    )
    result = resolve_turn_route(
        "what time is it in tokyo?",
        _BASE_CONFIG,
        {
            "model": "anthropic/claude-sonnet-4",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
            "api_key": "sk-primary",
        },
    )
    assert result["model"] == "anthropic/claude-sonnet-4"
    assert result["runtime"]["provider"] == "openrouter"
    assert result["label"] is None


def test_resolve_turn_route_preserves_extra_headers_from_primary(monkeypatch):
    """resolve_turn_route should preserve extra_headers from the primary runtime."""
    from agent.smart_model_routing import resolve_turn_route

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad route")),
    )
    result = resolve_turn_route(
        "what time is it in tokyo?",
        _BASE_CONFIG,
        {
            "model": "anthropic/claude-sonnet-4",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
            "api_key": "sk-primary",
            "extra_headers": {"x-host-head": "my-host-value"},
        },
    )
    assert result["model"] == "anthropic/claude-sonnet-4"
    assert result["runtime"]["provider"] == "openrouter"
    assert result["runtime"]["extra_headers"] == {"x-host-head": "my-host-value"}


def test_resolve_turn_route_preserves_extra_headers_in_signature(monkeypatch):
    """resolve_turn_route should include extra_headers in the signature tuple."""
    from agent.smart_model_routing import resolve_turn_route

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad route")),
    )
    result = resolve_turn_route(
        "what time is it in tokyo?",
        _BASE_CONFIG,
        {
            "model": "anthropic/claude-sonnet-4",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
            "api_key": "sk-primary",
            "extra_headers": {"x-host-head": "sig-test"},
        },
    )
    # signature is a tuple, extra_headers is NOT in the signature
    assert "extra_headers" not in result["signature"]


def test_resolve_turn_route_with_routed_runtime_includes_extra_headers(monkeypatch):
    """resolve_turn_route should include extra_headers from the resolved routed runtime."""
    from agent.smart_model_routing import resolve_turn_route

    def mock_resolve(**kwargs):
        return {
            "provider": "custom",
            "base_url": "https://custom.host/v1",
            "api_key": "route-key",
            "api_mode": "chat_completions",
            "extra_headers": {"x-host-head": "routed-host-value"},
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        mock_resolve,
    )
    result = resolve_turn_route(
        "what time is it in tokyo?",
        _BASE_CONFIG,
        {
            "model": "anthropic/claude-sonnet-4",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
            "api_key": "sk-primary",
            "extra_headers": {"x-host-head": "primary-value"},
        },
    )
    # When routing succeeds, the routed runtime's extra_headers should be used
    assert result["model"] == "google/gemini-2.5-flash"
    assert result["runtime"]["extra_headers"] == {"x-host-head": "routed-host-value"}
    assert result["label"] is not None
