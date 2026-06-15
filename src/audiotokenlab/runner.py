from __future__ import annotations

from audiotokenlab.config import load_config
from audiotokenlab.datasets import load_dataset
from audiotokenlab.models import MetricRow
from audiotokenlab.profiling import profile_clip
from audiotokenlab.reporting import write_run_artifacts
from audiotokenlab.tokenizers import build_tokenizer


def run_profile(config_path: str) -> list[MetricRow]:
    config = load_config(config_path)
    clips = load_dataset(config.dataset)
    tokenizer = build_tokenizer(config.tokenizer)

    rows: list[MetricRow] = []
    sample_dir = config.output_dir / "samples"
    for clip in clips:
        rows.extend(
            profile_clip(
                clip,
                tokenizer,
                config.strategies,
                config.kv_cache,
                sample_dir=sample_dir,
            )
        )

    write_run_artifacts(config, rows)
    return rows
