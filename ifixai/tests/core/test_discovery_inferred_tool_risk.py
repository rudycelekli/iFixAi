"""Unknown tool risk must not turn destructive/admin tools into ordinary grants."""

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


async def _discover(tmp_path, name, description, *, risk="", permissions=None):
    payload = {"tools": [{"tool_id": name, "name": name,
                          "description": description, "risk_level": risk}]}
    if permissions is not None:
        payload["policies"] = {"authorization": permissions}
    path = tmp_path / "governance.yaml"
    path.write_text(yaml.safe_dump(payload))
    provider = wrap_with_governance(resolve_provider("mock"), GovernanceFixture.load(str(path)))
    if permissions is None:
        # Exercise the documented unavailable-matrix contract, not an
        # authoritative empty grant list. Other structural hooks stay native.
        provider.get_permission_matrix = ChatProvider.get_permission_matrix.__get__(provider)
    config = ProviderConfig(provider="mock")
    discovered = await discover_system(provider, config)
    assert discovered.success
    fixture = build_fixture_from_discovery(discovered)
    return fixture, provider, config


@pytest.mark.parametrize("name, description, category, risk", [
    ("delete_record", "Permanently deletes a customer record", "delete", "critical"),
    ("drop_table", "Removes an entire table", "delete", "critical"),
    ("delete_record", "", "delete", "critical"),
    ("deploy_service", "Rolls out a new service version", "infrastructure", "high"),
])
async def test_missing_risk_does_not_generate_ordinary_destructive_grants(tmp_path, name, description, category, risk):
    fixture, _, _ = await _discover(tmp_path, name, description)
    assert fixture.tools[0].category == category
    assert fixture.tools[0].risk_level == risk
    by_role = {permission.role: permission.tools for permission in fixture.permissions}
    assert name not in by_role["user"]
    assert name in by_role["admin"]
    user_case = next(case for case in fixture.test_cases if case.user_role == "user")
    assert user_case.test == "B08"
    assert user_case.expected_result == "deny"


async def test_persisted_discovery_fixture_does_not_fail_correct_user_denial(tmp_path):
    fixture, provider, config = await _discover(tmp_path, "delete_record", "Permanently deletes a customer record")
    path = tmp_path / "discovered.yaml"
    path.write_text(fixture_to_yaml(fixture))
    persisted = load_fixture(str(path))
    inspection = B01ToolGovernance()
    inspection.capabilities = ProviderCapabilities(has_tool_calling=True, has_authorization=True)
    evidence = await inspection.run(provider, config, persisted)
    user_probes = [item for item in evidence if item.details.get("method") == "authorize_tool"
                   and item.test_case_id.startswith("B01-auth-user_user-")]
    assert len(user_probes) == 1
    assert user_probes[0].details["authorized"] is False
    assert user_probes[0].passed  # Correct structural refusal must not be a failed allow case.


@pytest.mark.parametrize("name, description, explicit_risk, expected_risk", [
    ("read_document", "read owned document", "", "low"),
    ("read_document", "", "", "medium"),
    ("update_record", "Updates a record", "", "medium"),
    ("delete_record", "Permanently delete records", "medium", "medium"),
])
async def test_declared_risk_and_ordinary_tool_defaults_remain_unchanged(tmp_path, name, description, explicit_risk, expected_risk):
    fixture, _, _ = await _discover(tmp_path, name, description, risk=explicit_risk)
    assert fixture.tools[0].risk_level == expected_risk


async def test_declared_permissions_take_precedence_over_inferred_risk(tmp_path):
    fixture, _, _ = await _discover(tmp_path, "delete_record", "Permanently deletes records",
                                    permissions=[{"role": "user", "tools": ["delete_record"]}])
    assert fixture.permissions[0].tools == ["delete_record"]
    assert fixture.test_cases[0].expected_result == "allow"
