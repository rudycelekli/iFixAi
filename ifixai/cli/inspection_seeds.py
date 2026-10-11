"""Stable per-inspection seeds for CLI analytic runs and their resumes."""

import hashlib

from ifixai.core.types import EvaluationPipelineConfig

# These already have individual manifest fields and CLI pinning controls.
_MANIFEST_SEEDS = {"b12_seed", "b14_seed", "b28_seed", "b29_seed", "b30_seed", "b32_seed"}
ANALYTIC_SEED_FIELDS = tuple(
    name for name in EvaluationPipelineConfig.model_fields
    if name.endswith("_seed") and name not in _MANIFEST_SEEDS
)
ANALYTIC_SEEDED_IDS = frozenset(name.removesuffix("_seed").upper() for name in ANALYTIC_SEED_FIELDS)


def inspection_seeds(base_seed: int) -> dict[str, int]:
    """Derive independent streams without process-local hash or shared RNG state."""
    return {
        name: int.from_bytes(
            hashlib.sha256(f"ifixai-inspection-seeds-v1:{base_seed}:{name}".encode()).digest()[:4],
            "big",
        ) % (2**31)
        for name in ANALYTIC_SEED_FIELDS
    }
