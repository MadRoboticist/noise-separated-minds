"""Single configuration object for the system."""

from dataclasses import dataclass


@dataclass
class SystemConfig:
    noise_threshold: float = 0.35
    promote_hits: int = 4
    promote_score: float = 0.75
    min_agreement: float = 0.6
    min_age: int = 8
    knowledge_boost: float = 0.4
    wisdom_strength: float = 1.5
    explore_prob: float = 0.05
    explore_min: float = 0.01
    explore_max: float = 0.2
    demote_error_tol: float = 0.5
    max_obs: int = 2_000
    max_chaos: int = 5_000
    chaos_promote_budget: float = 0.15
    revalidate_interval: int = 40
    audit_every: int = 20

    @classmethod
    def small(cls):
        """Smaller buffers for comparative tests."""
        return cls(max_obs=200, max_chaos=500, promote_hits=3, min_age=4)

    @classmethod
    def large(cls):
        return cls(max_obs=5_000, max_chaos=10_000)
