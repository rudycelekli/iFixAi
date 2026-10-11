"""Completed audits must disclose failures of the independent consistency check."""

import json

import aiohttp
import pytest
from aiohttp import web

from ifixai.core import runner
from ifixai.core.fixture_loader import load_fixture
from ifixai.core.types import (
    ProviderCapabilities,
    ProviderConfig,
    TestResult,
    TestStatus,
)
from ifixai.harness.registry import ALL_SPECS
from ifixai.providers.base import ChatProvider
from ifixai.reporting.scorecard import (
    generate_json_report,
    generate_markdown_report,
    render_consistency_warnings,
)


class HookProvider(ChatProvider):
    def __init__(self, session, endpoint):
        self.session = session
        self.endpoint = endpoint

    async def send_message(self, messages, config):
        raise AssertionError("completed checkpoints must remain reusable")

    async def get_roles(self, config):
        async with self.session.get(self.endpoint) as response:
            response.raise_for_status()
            return []

    async def get_audit_trail(self, query, config):
        return []

    async def get_permission_matrix(self, config):
        return []


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["run_all", "run_strategic", "run_selected"])
@pytest.mark.parametrize("status", [503, 200])
async def test_hook_outage_disclosed_without_discarding_completed_results(
    entrypoint, status
):
    app = web.Application()
    calls = []

    async def roles(request):
        calls.append(request.path)
        return web.json_response([], status=status)

    app.router.add_get("/roles", roles)
    server = web.AppRunner(app)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    cached = {
        spec.test_id: TestResult(
            test_id=spec.test_id,
            spec=spec,
            name=spec.name,
            category=spec.category,
            score=1.0,
            threshold=spec.threshold,
            passed=True,
            passing=True,
            status=TestStatus.PASS,
        )
        for spec in ALL_SPECS
    }
    try:
        async with aiohttp.ClientSession() as session:
            provider = HookProvider(session, f"http://127.0.0.1:{port}/roles")
            kwargs = dict(
                provider=provider,
                config=ProviderConfig(provider="custom"),
                fixture=load_fixture("default"),
                capabilities=ProviderCapabilities(),
                cached_results=cached,
            )
            if entrypoint == "run_selected":
                kwargs["test_ids"] = {"B03"}
            result = await getattr(runner, entrypoint)(**kwargs)
            assert calls == ["/roles"]
            assert result.test_results
            assert all(item.score == 1.0 for item in result.test_results)
            report = generate_json_report(result)
            if status == 503:
                assert any(
                    "consistency" in warning.lower() and "incomplete" in warning.lower()
                    for warning in result.validation_warnings
                )
                assert (
                    "consistency" in report.lower() and "incomplete" in report.lower()
                )
                markdown = generate_markdown_report(result)
                assert "verification incomplete" in markdown.lower()
                assert "scores are capped" not in render_consistency_warnings(result)
                assert not result.score_capped
                if entrypoint == "run_selected":
                    print(
                        json.dumps(
                            {
                                "test_id": "B03",
                                "score": result.test_results[0].score,
                                "status": result.test_results[0].status.value,
                                "validation_warnings": result.validation_warnings,
                                "hook_http_status": status,
                            }
                        )
                    )
            else:
                assert result.validation_warnings == []
    finally:
        await server.cleanup()
