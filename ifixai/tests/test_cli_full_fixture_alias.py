"""Full mode must apply its fixture policy to resolved fixture identities."""

import importlib
from pathlib import Path

import pytest
from click.testing import CliRunner

from ifixai.cli.main import ifixai_cli
from ifixai.core.connection import ConnectionTestResult


@pytest.mark.parametrize("fixture_kind", ["builtin-name", "absolute-file", "symlink", "custom-name", "custom-file"])
def test_full_mode_resolves_fixture_before_enforcing_default_ban(tmp_path, monkeypatch, fixture_kind):
    module = importlib.import_module("ifixai.cli.run")
    fixture_root = Path(module.__file__).parents[1] / "fixtures"
    default = fixture_root / "default/fixture.yaml"
    custom = fixture_root / "examples/customer_support.yaml"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IFIXAI_TELEMETRY", "0")
    executions = []

    async def probe(_provider, _config):
        return ConnectionTestResult(success=True)

    async def execute(**kwargs):
        executions.append(kwargs)
        return None

    monkeypatch.setattr(module, "_test_conn", probe)
    monkeypatch.setattr(module, "execute_tests", execute)
    if fixture_kind == "builtin-name":
        fixture = "default"
    elif fixture_kind == "absolute-file":
        fixture = str(default)
    elif fixture_kind == "symlink":
        link = tmp_path / "default-link.yaml"
        link.symlink_to(default)
        fixture = str(link)
    elif fixture_kind == "custom-name":
        fixture = "customer_support"
    else:
        fixture = str(custom)
    result = CliRunner().invoke(ifixai_cli, ["run", "--provider", "mock", "--fixture", fixture, "--mode", "full", "--eval-mode", "full", "--judge-provider", "mock", "--judge-provider", "http", "--judge-api-key", "", "--judge-api-key", "", "--test", "B01", "--no-promo", "--no-telemetry", "--no-parallel", "--output", str(tmp_path / "reports"), "--reliability-out", str(tmp_path / "runs")])
    if fixture_kind.startswith("custom"):
        assert executions, result.output
        assert "requires a hand-built fixture" not in result.output
    else:
        assert not executions, result.output
        assert result.exit_code == 1, result.output
        assert "requires a hand-built fixture" in result.output
        assert not (tmp_path / "runs").exists()
