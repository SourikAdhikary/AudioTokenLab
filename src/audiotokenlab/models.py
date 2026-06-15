from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AudioClip:
    """A normalized mono audio clip."""

    clip_id: str
    samples: tuple[float, ...]
    sample_rate: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return len(self.samples) / self.sample_rate


@dataclass(frozen=True)
class TokenBundle:
    """Discrete audio-token representation emitted by a tokenizer."""

    clip_id: str
    tokenizer: str
    tokens: tuple[int, ...]
    frame_rate: float
    codebook_count: int
    sample_rate: int
    duration_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    @property
    def tokens_per_second(self) -> float:
        if self.duration_seconds <= 0:
            return 0.0
        return self.token_count / self.duration_seconds


@dataclass(frozen=True)
class CompressionResult:
    """Result of applying a compression strategy to tokenized audio."""

    strategy: str
    original: TokenBundle
    compressed: TokenBundle
    reconstructed: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_reduction_ratio(self) -> float:
        if self.original.token_count == 0:
            return 0.0
        removed = self.original.token_count - self.compressed.token_count
        return removed / self.original.token_count


@dataclass(frozen=True)
class ProfileConfig:
    """Runtime configuration for a profiling run."""

    run_id: str
    output_dir: Path
    dataset: dict[str, Any]
    tokenizer: dict[str, Any]
    strategies: list[dict[str, Any]]
    kv_cache: dict[str, Any]


@dataclass(frozen=True)
class MetricRow:
    """Flat row written to CSV and report output."""

    clip_id: str
    strategy: str
    duration_seconds: float
    original_tokens: int
    compressed_tokens: int
    token_reduction_ratio: float
    original_tokens_per_second: float
    compressed_tokens_per_second: float
    encode_ms: float
    decode_ms: float
    real_time_factor: float
    estimated_kv_cache_mb: float
    estimated_kv_cache_savings_mb: float
    reconstruction_mse: float
    duration_drift_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "strategy": self.strategy,
            "duration_seconds": self.duration_seconds,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "token_reduction_ratio": self.token_reduction_ratio,
            "original_tokens_per_second": self.original_tokens_per_second,
            "compressed_tokens_per_second": self.compressed_tokens_per_second,
            "encode_ms": self.encode_ms,
            "decode_ms": self.decode_ms,
            "real_time_factor": self.real_time_factor,
            "estimated_kv_cache_mb": self.estimated_kv_cache_mb,
            "estimated_kv_cache_savings_mb": self.estimated_kv_cache_savings_mb,
            "reconstruction_mse": self.reconstruction_mse,
            "duration_drift_ms": self.duration_drift_ms,
        }

