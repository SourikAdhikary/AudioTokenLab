from __future__ import annotations

import math
import time
from pathlib import Path

from audiotokenlab.audio_io import write_wav
from audiotokenlab.compression import compress_tokens
from audiotokenlab.models import AudioClip, CompressionResult, MetricRow
from audiotokenlab.tokenizers.base import AudioTokenizer


def profile_clip(
    clip: AudioClip,
    tokenizer: AudioTokenizer,
    strategies: list[dict],
    kv_cache: dict,
    sample_dir: Path | None = None,
) -> list[MetricRow]:
    encode_start = time.perf_counter()
    original = tokenizer.encode(clip)
    encode_ms = (time.perf_counter() - encode_start) * 1000.0

    rows: list[MetricRow] = []
    for strategy in strategies:
        compressed = compress_tokens(original, strategy)
        decode_start = time.perf_counter()
        reconstructed = tokenizer.decode(compressed)
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
        result = CompressionResult(
            strategy=str(strategy.get("label", strategy.get("name", "baseline"))),
            original=original,
            compressed=compressed,
            reconstructed=reconstructed,
            metadata=dict(strategy),
        )
        if sample_dir is not None:
            write_wav(
                sample_dir / f"{clip.clip_id}__{result.strategy}.wav",
                reconstructed,
                clip.sample_rate,
            )
        rows.append(
            build_metric_row(
                clip=clip,
                result=result,
                encode_ms=encode_ms,
                decode_ms=decode_ms,
                kv_cache=kv_cache,
            )
        )
    return rows


def build_metric_row(
    clip: AudioClip,
    result: CompressionResult,
    encode_ms: float,
    decode_ms: float,
    kv_cache: dict,
) -> MetricRow:
    original_kv_mb = estimate_kv_cache_mb(result.original.token_count, kv_cache)
    compressed_kv_mb = estimate_kv_cache_mb(result.compressed.token_count, kv_cache)
    duration_ms = max(1e-9, result.original.duration_seconds * 1000.0)
    return MetricRow(
        clip_id=clip.clip_id,
        strategy=result.strategy,
        duration_seconds=clip.duration_seconds,
        original_tokens=result.original.token_count,
        compressed_tokens=result.compressed.token_count,
        token_reduction_ratio=result.token_reduction_ratio,
        original_tokens_per_second=result.original.tokens_per_second,
        compressed_tokens_per_second=result.compressed.tokens_per_second,
        encode_ms=encode_ms,
        decode_ms=decode_ms,
        real_time_factor=decode_ms / duration_ms,
        estimated_kv_cache_mb=compressed_kv_mb,
        estimated_kv_cache_savings_mb=original_kv_mb - compressed_kv_mb,
        reconstruction_mse=mean_squared_error(clip.samples, result.reconstructed),
        reconstruction_mae=mean_absolute_error(clip.samples, result.reconstructed),
        reconstruction_snr_db=signal_to_noise_db(clip.samples, result.reconstructed),
        duration_drift_ms=abs(len(result.reconstructed) - len(clip.samples))
        / max(1, clip.sample_rate)
        * 1000.0,
    )


def estimate_kv_cache_mb(token_count: int, kv_cache: dict) -> float:
    layers = int(kv_cache.get("layers", 24))
    hidden_size = int(kv_cache.get("hidden_size", 1024))
    bytes_per_element = int(kv_cache.get("bytes_per_element", 2))
    kv_bytes = token_count * layers * hidden_size * 2 * bytes_per_element
    return kv_bytes / (1024.0 * 1024.0)


def mean_squared_error(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    count = min(len(left), len(right))
    if count == 0:
        return 0.0
    total = 0.0
    for index in range(count):
        diff = left[index] - right[index]
        total += diff * diff
    return total / count


def mean_absolute_error(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    count = min(len(left), len(right))
    if count == 0:
        return 0.0
    total = 0.0
    for index in range(count):
        total += abs(left[index] - right[index])
    return total / count


def signal_to_noise_db(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    count = min(len(left), len(right))
    if count == 0:
        return 0.0

    signal_power = 0.0
    noise_power = 0.0
    for index in range(count):
        signal_power += left[index] * left[index]
        diff = left[index] - right[index]
        noise_power += diff * diff

    if noise_power == 0.0:
        return 99.0
    if signal_power == 0.0:
        return 0.0
    return 10.0 * math.log10(signal_power / noise_power)
