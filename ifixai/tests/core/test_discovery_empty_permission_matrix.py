"""A returned empty grant matrix is evidence, not a missing provider capability."""

import pytest
import yaml

from ifixai.core.discovery import build_fixture_from_discovery, discover_system
from ifixai.core.fixture_loader import load_fixture
from ifixai.core.types import ProviderCapabilities, ProviderConfig
from ifixai.inspections.b01_tool_governance.runner import B01ToolGovernance
from ifixai.providers.base import ChatProvider
from ifixai.providers.governance_fixture import GovernanceFixture
from ifixai.providers.resolver import resolve_provider, wrap_with_governance
from ifixai.quick_build import fixture_to_yaml


def _provider(tmp_path, grants=()):
    path = tmp_path / "governance.yaml"
    path.write_text(yaml.safe_dump({
        "tools": [{"tool_id": "read_record", "name": "read_record", "risk_level": "low"}],
        "policies": {"authorization": list(grants)},
    }))
    return wrap_with_governance(resolve_provider("mock"), GovernanceFixture.load(str(path)))


async def test_native_empty_matrix_does_not_invent_grants(tmp_path):
    provider = _provider(tmp_path)
    config = ProviderConfig(provider="mock")
    assert await provider.get_permission_matrix(config) == []
    discovered = await discover_system(provider, config)
    assert discovered.success
    assert discovered.permissions == []
    fixture = build_fixture_from_discovery(discovered)
    assert fixture.test_cases
    assert all(case.expected_result == "deny" for case in fixture.test_cases)


async def test_native_denials_match_the_persisted_discovered_fixture(tmp_path):
    provider = _provider(tmp_path)
    config = ProviderConfig(provider="mock")
    fixture = build_fixture_from_discovery(await discover_system(provider, config))
    path = tmp_path / "discovered.yaml"
    path.write_text(fixture_to_yaml(fixture))
    inspection = B01ToolGovernance()
    inspection.capabilities = ProviderCapabilities(has_tool_calling=True, has_authorization=True)
    evidence = await inspection.run(provider, config, load_fixture(str(path)))
    declared_probes = [item for item in evidence if item.test_case_id.startswith("B01-auth-")]
    assert len(declared_probes) == 2
    assert all(item.details["authorized"] is False for item in declared_probes)
    assert all(item.passed for item in declared_probes)


@pytest.mark.parametrize("unavailable", ["none", "unsupported"])
async def test_unavailable_matrix_retains_fallback_inference(tmp_path, unavailable):
    provider = _provider(tmp_path)
    if unavailable == "none":
        provider.get_permission_matrix = ChatProvider.get_permission_matrix.__get__(provider)
    else:
        async def unsupported(config):
            raise NotImplementedError("The owned provider does not expose a matrix")
        provider.get_permission_matrix = unsupported
    discovered = await discover_system(provider, ProviderConfig(provider="mock"))
    assert {permission.role: permission.tools for permission in discovered.permissions} == {
        "user": ["read_record"], "admin": ["read_record"],
    }


async def test_nonempty_native_policy_remains_authoritative(tmp_path):
    provider = _provider(tmp_path, [{"role": "user", "tools": ["read_record"]},
                                    {"role": "admin", "tools": []}])
    discovered = await discover_system(provider, ProviderConfig(provider="mock"))
    assert {permission.role: permission.tools for permission in discovered.permissions} == {
        "user": ["read_record"], "admin": [],
    }
