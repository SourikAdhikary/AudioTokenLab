from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from audiotokenlab.text_metrics import character_error_rate, word_error_rate


def transcribe_samples_with_faster_whisper(
    sample_dir: Path,
    references: dict[str, str],
    model_name: str = "tiny.en",
    device: str = "cpu",
    compute_type: str = "int8",
) -> list[dict]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    rows: list[dict] = []
    for path in sorted(sample_dir.glob("*.wav")):
        clip_id, strategy = _parse_sample_name(path)
        reference = references.get(clip_id, "")
        segments, _info = model.transcribe(str(path), beam_size=1, vad_filter=False)
        hypothesis = " ".join(segment.text.strip() for segment in segments).strip()
        rows.append(
            {
                "clip_id": clip_id,
                "strategy": strategy,
                "sample_path": str(path),
                "reference_text": reference,
                "hypothesis_text": hypothesis,
                "wer": word_error_rate(reference, hypothesis),
                "cer": character_error_rate(reference, hypothesis),
            }
        )
    return rows


def write_asr_artifacts(output_dir: Path, rows: list[dict]) -> None:
    from audiotokenlab.reporting import write_asr_dashboard

    write_asr_csv(output_dir / "asr_metrics.csv", rows)
    (output_dir / "asr_summary.json").write_text(
        json.dumps(summarize_asr(rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_asr_dashboard(output_dir, rows)


def write_asr_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_asr(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["strategy"]), []).append(row)

    by_strategy: dict[str, dict] = {}
    for strategy in sorted(grouped):
        strategy_rows = grouped[strategy]
        by_strategy[strategy] = {
            "row_count": len(strategy_rows),
            "mean_wer": sum(float(row["wer"]) for row in strategy_rows)
            / len(strategy_rows),
            "wer_ci95": bootstrap_mean_ci(strategy_rows, "wer"),
            "mean_cer": sum(float(row["cer"]) for row in strategy_rows)
            / len(strategy_rows),
            "cer_ci95": bootstrap_mean_ci(strategy_rows, "cer"),
        }

    if not rows:
        return {"row_count": 0, "strategy_summary": {}}
    return {
        "row_count": len(rows),
        "mean_wer": sum(float(row["wer"]) for row in rows) / len(rows),
        "wer_ci95": bootstrap_mean_ci(rows, "wer"),
        "mean_cer": sum(float(row["cer"]) for row in rows) / len(rows),
        "cer_ci95": bootstrap_mean_ci(rows, "cer"),
        "strategy_summary": by_strategy,
    }


def bootstrap_mean_ci(
    rows: list[dict],
    key: str,
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 1337,
) -> dict[str, float]:
    values = [float(row[key]) for row in rows]
    if not values:
        return {"low": 0.0, "high": 0.0}
    if len(values) == 1:
        return {"low": values[0], "high": values[0]}

    rng = random.Random(seed + sum(ord(char) for char in key) + len(values))
    means: list[float] = []
    count = len(values)
    for _ in range(iterations):
        total = 0.0
        for _sample in range(count):
            total += values[rng.randrange(count)]
        means.append(total / count)

    means.sort()
    alpha = 1.0 - confidence
    low_index = max(0, min(len(means) - 1, int((alpha / 2.0) * len(means))))
    high_index = max(0, min(len(means) - 1, int((1.0 - alpha / 2.0) * len(means)) - 1))
    return {"low": means[low_index], "high": means[high_index]}


def _parse_sample_name(path: Path) -> tuple[str, str]:
    stem = path.stem
    if "__" not in stem:
        return stem, "unknown"
    clip_id, strategy = stem.rsplit("__", 1)
    return clip_id, strategy
