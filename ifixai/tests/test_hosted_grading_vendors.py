"""A hosting adapter does not establish the independence of its model's creator."""

from types import SimpleNamespace

import pytest

from ifixai.core.types import ProviderConfig, TestResult, TestStatus
from ifixai.inspections.v05_grader_independence.runner_floors import (
    grader_independence_details,
    independence_floor_corrected,
)
from ifixai.judge.config import JudgeConfig
from ifixai.reporting.scorecard import (
    SELF_JUDGE_BIAS_ADVISORY,
    grading_vendor,
    scorecard_warnings,
)


@pytest.mark.parametrize("model", [
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "us.anthropic.claude-3-haiku-20240307-v1:0",
    "eu.anthropic.claude-3-haiku-20240307-v1:0",
    "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-3-haiku-20240307-v1:0",
])
def test_bedrock_same_creator_retains_bias_warning_and_v05_floor(model):
    sut = ProviderConfig(provider="bedrock", model=model)
    judge = ProviderConfig(provider="anthropic", model="claude-sonnet-4-5")
    pipeline = SimpleNamespace(classifier_pair=lambda: {"config": judge})
    details = grader_independence_details(sut, pipeline)
    result = independence_floor_corrected(
        TestResult(test_id="V05", status=TestStatus.PASS, passing=True, passed=True, score=1.0),
        details,
    )
    assert details["verdict"] == "same_vendor"
    assert result.status == TestStatus.INCONCLUSIVE
    assert not result.passing
    warnings = scorecard_warnings(JudgeConfig(provider="anthropic", model=judge.model), sut.provider, sut.model)
    assert SELF_JUDGE_BIAS_ADVISORY in warnings


@pytest.mark.parametrize("provider, model, vendor", [
    ("bedrock", "meta.llama3-70b-instruct-v1:0", "meta"),
    ("bedrock", "amazon.nova-pro-v1:0", "amazon"),
    ("huggingface", "meta-llama/Llama-3.3-70B-Instruct", "meta"),
    ("huggingface", "mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
])
def test_hosted_models_resolve_declared_creator(provider, model, vendor):
    assert grading_vendor(provider, model) == vendor


def test_different_bedrock_creators_are_independent():
    assert SELF_JUDGE_BIAS_ADVISORY not in scorecard_warnings(
        JudgeConfig(provider="bedrock", model="meta.llama3-70b-instruct-v1:0"),
        "bedrock", "anthropic.claude-3-sonnet-20240229-v1:0",
    )


@pytest.mark.parametrize("provider, model, vendor", [
    ("bedrock", None, "bedrock"),
    ("bedrock", "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/opaque", "bedrock"),
    ("bedrock", "owned-custom-model", "bedrock"),
    ("huggingface", None, "huggingface"),
    ("huggingface", "owned-local-model", "huggingface"),
    ("azure", "owned-claude-deployment", "openai"),
    ("openrouter", "anthropic/claude-sonnet-4-5", "anthropic"),
])
def test_unresolved_hosts_and_existing_routes_keep_identity(provider, model, vendor):
    assert grading_vendor(provider, model) == vendor
