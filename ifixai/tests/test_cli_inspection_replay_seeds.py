"""Interrupted analytic CLI runs must resume the same inspection corpus."""

import importlib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from ifixai.cli.main import ifixai_cli
from ifixai.inspections.v09_prompt_override_resistance import runner as v09
from ifixai.inspections.v09_prompt_override_resistance.runner_rotation import (
    manifest_digest,
)


@pytest.mark.parametrize("legacy", [False, True])
def test_native_v09_interrupted_resume_preserves_rotation(tmp_path, monkeypatch, legacy):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IFIXAI_TELEMETRY", "0")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    # Make the old random config's drift reproducible, rather than a flaky
    # assertion that two random seeds happen to select different rotations.
    cli_run = importlib.import_module("ifixai.cli.run")
    original_manifest = cli_run.build_manifest
    if legacy:
        def legacy_manifest(**kwargs):
            kwargs["seed"] = None
            return original_manifest(**kwargs)
        monkeypatch.setattr(cli_run, "build_manifest", legacy_manifest)
    original_rotation = v09.rotation_offsets
    original_specs = v09.select_probe_specs
    rotations = []
    seeds = []

    def record_rotation(seed):
        seeds.append(seed)
        rotation = original_rotation(seed)
        rotations.append(manifest_digest(rotation))
        return rotation

    def interrupt_once(users, rotation):
        specs = original_specs(users, rotation)
        if len(rotations) == 1:
            raise KeyboardInterrupt
        return specs

    monkeypatch.setattr(v09, "rotation_offsets", record_rotation)
    monkeypatch.setattr(v09, "select_probe_specs", interrupt_once)
    # Pydantic captures default_factory functions when the class is defined.
    # Supply deterministic successive seeds through the real factory's RNG.
    draws = iter(range(1, 10000))
    monkeypatch.setattr("ifixai.core.types.secrets.randbelow", lambda _: next(draws))
    fixture = Path(__file__).parents[1] / "fixtures/examples/customer_support.yaml"
    args = ["run", "--provider", "mock", "--model", "owned-mock",
            "--fixture", str(fixture), "--test", "B01", "--test", "V09",
            "--eval-mode", "single", "--judge-provider", "mock",
            "--judge-api-key", "owned-unused", "--judge-budget", "0",
            "--output", str(tmp_path / "reports"),
            "--reliability-out", str(tmp_path / "runs"),
            "--no-parallel", "--no-promo", "--no-telemetry"]
    first = CliRunner().invoke(ifixai_cli, args)
    assert first.exit_code == 130, first.output
    manifest_path, = (tmp_path / "runs").glob("*/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    checkpoint = json.loads(manifest_path.with_name("checkpoint.json").read_text())
    assert "B01" in checkpoint
    assert "V09" not in checkpoint
    monkeypatch.setattr(cli_run, "build_manifest", original_manifest)
    resumed = CliRunner().invoke(ifixai_cli, [*args, "--resume", manifest["run_id"]])
    if legacy:
        assert resumed.exit_code == 1, resumed.output
        assert "original probe contexts cannot be restored" in resumed.output
        assert len(seeds) == 1
        assert json.loads(manifest_path.with_name("checkpoint.json").read_text()) == checkpoint
        return
    assert resumed.exit_code in (0, 2), resumed.output
    assert "[reused] B01" in resumed.output
    assert len(seeds) == 2
    assert seeds[0] == seeds[1], (seeds, rotations)
    assert rotations[0] == rotations[1]
    assert json.loads(manifest_path.read_text())["run_id"] == manifest["run_id"]
    checkpoint = json.loads(manifest_path.with_name("checkpoint.json").read_text())
    assert "V09" in checkpoint
    assert checkpoint["V09"]["variant_seed"] == seeds[0]


@pytest.mark.parametrize("mode", ["single", "deterministic"])
def test_native_legacy_b_only_resume_remains_compatible(tmp_path, monkeypatch, mode):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IFIXAI_TELEMETRY", "0")
    cli_run = importlib.import_module("ifixai.cli.run")
    original_manifest = cli_run.build_manifest

    def legacy_manifest(**kwargs):
        kwargs["seed"] = None
        return original_manifest(**kwargs)

    monkeypatch.setattr(cli_run, "build_manifest", legacy_manifest)
    configs = []
    original_execute = cli_run.execute_tests

    async def capture_execute(*args, **kwargs):
        configs.append(kwargs.get("pipeline_config"))
        return await original_execute(*args, **kwargs)

    monkeypatch.setattr(cli_run, "execute_tests", capture_execute)
    args = ["run", "--provider", "mock", "--fixture", "customer_support",
            "--test", "B01", "--eval-mode", mode, "--no-promo", "--no-telemetry",
            "--no-parallel", "--output", str(tmp_path / "reports"),
            "--reliability-out", str(tmp_path / "runs")]
    if mode == "single":
        args += ["--judge-provider", "mock"]
    first = CliRunner().invoke(ifixai_cli, args)
    assert first.exit_code == 2, first.output
    path, = (tmp_path / "runs").glob("*/manifest.json")
    manifest = json.loads(path.read_text())
    monkeypatch.setattr(cli_run, "build_manifest", original_manifest)
    resumed = CliRunner().invoke(ifixai_cli, [*args, "--resume", manifest["run_id"]])
    assert resumed.exit_code == 2, resumed.output
    assert "[reused] B01" in resumed.output
    assert json.loads(path.read_text())["seed"] is None
    if mode == "deterministic":
        assert configs == [None, None]


def test_derived_seeds_cover_all_unpersisted_inspections():
    from ifixai.cli.inspection_seeds import ANALYTIC_SEED_FIELDS, inspection_seeds
    from ifixai.core.types import EvaluationPipelineConfig

    seeds = inspection_seeds(17)
    config = EvaluationPipelineConfig(**seeds)
    assert len(seeds) == 22
    assert seeds == inspection_seeds(17)
    assert seeds != inspection_seeds(18)
    assert len(set(seeds.values())) == len(seeds)
    assert config.b12_seed_pinned is False
    assert config.v09_seed_pinned is False
    for field in ANALYTIC_SEED_FIELDS:
        assert getattr(config, field) == seeds[field]
        assert 0 <= seeds[field] < 2**31
