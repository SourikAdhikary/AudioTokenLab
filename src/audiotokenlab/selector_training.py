from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


SELECTOR_TEMPLATES = {
    "acoustic_salience": {
        "bias": -0.05,
        "energy": 0.0,
        "onset": 0.0,
        "transition": 2.0,
        "speech_activity": 0.0,
        "center": 0.0,
    },
    "energy_salience": {
        "bias": -0.05,
        "energy": 2.0,
        "onset": 2.0,
        "transition": 1.0,
        "speech_activity": 0.0,
        "center": 0.0,
    },
    "vad_salience": {
        "bias": -0.05,
        "energy": 1.0,
        "onset": 2.0,
        "transition": 1.0,
        "speech_activity": 2.0,
        "center": 0.0,
        "noise_floor_ratio": 1.8,
        "absolute_threshold": 0.04,
        "min_speech_frames": 2.0,
        "hangover_frames": 1.0,
    },
    "linear_selector_v1": {
        "bias": -0.05,
        "energy": 2.2,
        "onset": 1.4,
        "transition": 1.0,
        "speech_activity": 1.8,
        "center": 0.15,
        "noise_floor_ratio": 1.8,
        "absolute_threshold": 0.04,
        "min_speech_frames": 1.0,
        "hangover_frames": 1.0,
    },
}


def train_selector_from_artifacts(
    run_dir: Path,
    output_path: Path,
    target_reduction: float = 0.5,
    wer_weight: float = 1.0,
    cer_weight: float = 0.25,
    speaker_weight: float = 0.35,
    reduction_weight: float = 0.25,
    temperature: float = 0.08,
) -> dict[str, Any]:
    metrics_rows = _read_csv_dicts(run_dir / "metrics.csv")
    asr_rows = _read_csv_dicts(run_dir / "asr_metrics.csv")
    speaker_rows = _read_csv_dicts(run_dir / "speaker_metrics.csv")
    joined = _join_quality_rows(metrics_rows, asr_rows, speaker_rows)
    selector_rows = [
        row
        for row in joined
        if row["strategy"] in SELECTOR_TEMPLATES
    ]
    if not selector_rows:
        raise ValueError(f"No selector strategy rows found in {run_dir}")

    scored = []
    for row in selector_rows:
        score = (
            wer_weight * row["wer"]
            + cer_weight * row["cer"]
            + speaker_weight * (1.0 - row["speaker_similarity"])
            + reduction_weight * abs(row["token_reduction_ratio"] - target_reduction)
        )
        scored.append({**row, "selector_loss": score})

    credit_by_strategy = _strategy_credit(scored, temperature=temperature)
    weights = _blend_selector_templates(credit_by_strategy)
    strategy = {
        "name": "learned_selector",
        "label": "trained_selector_v1",
        "factor": 2,
        "weights": weights,
    }
    summary = {
        "training_source": str(run_dir),
        "row_count": len(scored),
        "clip_count": len({row["clip_id"] for row in scored}),
        "objective": {
            "target_reduction": target_reduction,
            "wer_weight": wer_weight,
            "cer_weight": cer_weight,
            "speaker_weight": speaker_weight,
            "reduction_weight": reduction_weight,
            "temperature": temperature,
        },
        "strategy_credit": credit_by_strategy,
        "strategy_summary": _summarize_scored_rows(scored),
        "trained_strategy": strategy,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _join_quality_rows(
    metrics_rows: list[dict[str, str]],
    asr_rows: list[dict[str, str]],
    speaker_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    metrics_by_key = {
        (row.get("clip_id", ""), row.get("strategy", "")): row
        for row in metrics_rows
    }
    speaker_by_key = {
        (row.get("clip_id", ""), row.get("strategy", "")): row
        for row in speaker_rows
    }
    joined = []
    for asr in asr_rows:
        key = (asr.get("clip_id", ""), asr.get("strategy", ""))
        metrics = metrics_by_key.get(key, {})
        speaker = speaker_by_key.get(key, {})
        joined.append(
            {
                "clip_id": key[0],
                "strategy": key[1],
                "wer": _to_float(asr.get("wer")),
                "cer": _to_float(asr.get("cer")),
                "speaker_similarity": _to_float(speaker.get("speaker_similarity")),
                "token_reduction_ratio": _to_float(metrics.get("token_reduction_ratio")),
                "reconstruction_snr_db": _to_float(metrics.get("reconstruction_snr_db")),
            }
        )
    return joined


def _strategy_credit(
    rows: list[dict[str, Any]],
    temperature: float,
) -> dict[str, float]:
    by_clip: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_clip.setdefault(row["clip_id"], []).append(row)

    credit: dict[str, float] = {strategy: 0.0 for strategy in SELECTOR_TEMPLATES}
    for clip_rows in by_clip.values():
        best_loss = min(row["selector_loss"] for row in clip_rows)
        raw = [
            math.exp(-(row["selector_loss"] - best_loss) / max(1e-6, temperature))
            for row in clip_rows
        ]
        total = sum(raw) or 1.0
        for row, value in zip(clip_rows, raw, strict=True):
            credit[row["strategy"]] += value / total

    total_credit = sum(credit.values()) or 1.0
    return {
        strategy: value / total_credit
        for strategy, value in sorted(credit.items())
    }


def _blend_selector_templates(credit_by_strategy: dict[str, float]) -> dict[str, float]:
    keys = ["bias", "center", "energy", "onset", "speech_activity", "transition"]
    blended = {}
    for key in keys:
        value = 0.0
        for strategy, credit in credit_by_strategy.items():
            value += credit * SELECTOR_TEMPLATES[strategy].get(key, 0.0)
        blended[key] = round(value, 6)
    blended.update(
        {
            "noise_floor_ratio": 1.8,
            "absolute_threshold": 0.04,
            "min_speech_frames": 1.0,
            "hangover_frames": 1.0,
        }
    )
    return blended


def _summarize_scored_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["strategy"], []).append(row)
    summary = {}
    for strategy, strategy_rows in sorted(grouped.items()):
        summary[strategy] = {
            "row_count": float(len(strategy_rows)),
            "mean_selector_loss": _mean(strategy_rows, "selector_loss"),
            "mean_wer": _mean(strategy_rows, "wer"),
            "mean_cer": _mean(strategy_rows, "cer"),
            "mean_speaker_similarity": _mean(strategy_rows, "speaker_similarity"),
            "mean_token_reduction_ratio": _mean(strategy_rows, "token_reduction_ratio"),
        }
    return summary


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)


def _to_float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)
