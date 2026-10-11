"""Fallbacks must not silently route a cross-provider judge to the SUT model."""

import json

import pytest
from aiohttp import web

from ifixai import api
from ifixai.core.fixture_loader import load_fixture
from ifixai.core.types import EvaluationPipelineConfig
from ifixai.evaluation.analytic_judge import load_analytic_rubric
from ifixai.judge.config import JudgeConfig


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sut_model,expected_fallback",
    [
        ("gpt-4o-mini", "anthropic/claude-haiku-4.5"),
        ("gpt-4o", "openai/gpt-4o-mini"),
    ],
)
async def test_public_run_excludes_native_sut_alias_from_packaged_fallbacks(
    sut_model, expected_fallback, monkeypatch
):
    # Ordinary native adapters and packaged policy; only network endpoints and
    # owned model replies are local controls, not measured model quality.
    monkeypatch.delenv("IFIXAI_JUDGE_FALLBACKS", raising=False)
    rubric = await load_analytic_rubric("B19", "comply")
    models = []

    async def complete(request):
        payload = await request.json()
        model = payload["model"]
        models.append(model)
        if model == "google/gemini-2.5-flash":
            return web.json_response(
                {
                    "error": {
                        "message": "owned unavailable model",
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                    }
                },
                status=400,
            )
        content = (
            "owned response"
            if model == sut_model
            else json.dumps(
                {
                    "dimensions": [
                        {"name": d.name, "passed": True, "reasoning": "owned verdict"}
                        for d in rubric.dimensions
                    ]
                }
            )
        )
        return web.json_response(
            {
                "id": "owned",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    server = web.AppRunner(app)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}/v1"
    try:
        result = await api.run_single(
            "B19",
            "openai",
            model=sut_model,
            api_key="owned-loopback-key",
            endpoint=endpoint,
            fixture=load_fixture("default").model_copy(update={"governance": None}),
            judge_config=JudgeConfig(
                provider="openrouter",
                model="google/gemini-2.5-flash",
                api_key="owned-loopback-key",
                endpoint=endpoint,
            ),
            pipeline_config=EvaluationPipelineConfig(judge_max_calls=0),
        )
        fallbacks = set(models) - {sut_model, "google/gemini-2.5-flash"}
        assert fallbacks == {expected_fallback}
        assert result.score == 1.0
        assert len(result.evidence) >= 24
        print(
            json.dumps(
                {
                    "sut": sut_model,
                    "fallbacks": sorted(fallbacks),
                    "request_count": len(models),
                    "score": result.score,
                }
            )
        )
    finally:
        await server.cleanup()


@pytest.mark.parametrize(
    "sut_provider,sut_model,expected",
    [
        ("openai", "gpt-4o-mini", ["primary", "anthropic/claude-haiku-4.5"]),
        ("openrouter", "openai/gpt-4o-mini", ["primary", "anthropic/claude-haiku-4.5"]),
        (
            "http",
            "gpt-4o-mini",
            ["primary", "openai/gpt-4o-mini", "anthropic/claude-haiku-4.5"],
        ),
        (
            None,
            "gpt-4o-mini",
            ["primary", "openai/gpt-4o-mini", "anthropic/claude-haiku-4.5"],
        ),
        (
            "openai",
            "gpt-4o",
            ["primary", "openai/gpt-4o-mini", "anthropic/claude-haiku-4.5"],
        ),
        ("anthropic", "claude-haiku-4.5", ["primary", "openai/gpt-4o-mini"]),
    ],
)
def test_fallback_identity_requires_known_vendor_and_exact_model(
    sut_provider, sut_model, expected
):
    from ifixai.judge.fallbacks import JudgeFallbackPolicy, resolve_judge_model_chain

    policy = JudgeFallbackPolicy.model_validate(
        {
            "providers": {
                "openrouter": {
                    "models": [
                        {"model": "openai/gpt-4o-mini"},
                        {"model": "anthropic/claude-haiku-4.5"},
                    ]
                }
            }
        }
    )
    assert (
        resolve_judge_model_chain(
            "openrouter",
            "primary",
            policy,
            excluded_model=sut_model,
            excluded_provider=sut_provider,
        )
        == expected
    )


def test_explicit_self_judge_primary_is_preserved():
    from ifixai.judge.fallbacks import resolve_judge_model_chain

    chain = resolve_judge_model_chain(
        "openrouter",
        "openai/gpt-4o-mini",
        excluded_model="gpt-4o-mini",
        excluded_provider="openai",
    )
    assert chain[0] == "openai/gpt-4o-mini"
    assert chain.count("openai/gpt-4o-mini") == 1


def test_ensemble_pipeline_propagates_sut_identity_to_each_judge():
    from ifixai.core.runner import _build_judge_evaluator, _build_pipeline
    from ifixai.judge.config import JudgeProviderSpec

    judge = _build_judge_evaluator(
        JudgeConfig(
            providers=[
                JudgeProviderSpec(
                    provider="openrouter", model="google/gemini-2.5-flash"
                ),
                JudgeProviderSpec(
                    provider="openrouter", model="anthropic/claude-haiku-4.5"
                ),
            ]
        )
    )
    pipeline = _build_pipeline(
        EvaluationPipelineConfig(),
        judge,
        sut_model="gpt-4o-mini",
        sut_provider="openai",
    )
    assert pipeline is not None
    for member in pipeline._judge._per_judge:
        assert "openai/gpt-4o-mini" not in member._model_chain
