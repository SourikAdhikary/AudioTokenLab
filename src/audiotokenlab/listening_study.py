from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any


def write_listening_study_artifacts(
    output_dir: Path,
    max_items: int = 24,
    seed: int = 13,
) -> dict[str, Any]:
    asr_rows = _read_csv_dicts(output_dir / "asr_metrics.csv")
    speaker_rows = _read_csv_dicts(output_dir / "speaker_metrics.csv")
    if not asr_rows:
        summary = {"item_count": 0, "seed": seed, "reason": "missing_asr_metrics"}
        (output_dir / "listening_study.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return summary

    speaker_by_key = {
        (row.get("clip_id", ""), row.get("strategy", "")): row
        for row in speaker_rows
    }
    selected = _select_study_rows(asr_rows, max_items=max_items, seed=seed)
    study_rows = []
    for index, row in enumerate(selected, 1):
        speaker = speaker_by_key.get((row.get("clip_id", ""), row.get("strategy", "")), {})
        study_rows.append(
            {
                "stimulus_id": f"atl_{index:04d}",
                "clip_id": row.get("clip_id", ""),
                "strategy": row.get("strategy", ""),
                "sample_path": _relative_sample_path(row.get("sample_path", "")),
                "reference_text": row.get("reference_text", ""),
                "hypothesis_text": row.get("hypothesis_text", ""),
                "wer": _to_float(row.get("wer")),
                "cer": _to_float(row.get("cer")),
                "speaker_similarity": _to_float(speaker.get("speaker_similarity")),
                "mos_1_5": "",
                "intelligibility_1_5": "",
                "speaker_match_1_5": "",
                "artifact_notes": "",
            }
        )

    _write_study_csv(output_dir / "listening_study.csv", study_rows)
    _write_study_markdown(output_dir / "listening_study.md", study_rows)
    summary = {
        "item_count": len(study_rows),
        "seed": seed,
        "strategies": sorted({row["strategy"] for row in study_rows}),
        "output_files": [
            "listening_study.csv",
            "listening_study.md",
            "listening_study.json",
        ],
    }
    (output_dir / "listening_study.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _select_study_rows(
    rows: list[dict[str, str]],
    max_items: int,
    seed: int,
) -> list[dict[str, str]]:
    baseline_by_clip = {
        row.get("clip_id", ""): row
        for row in rows
        if row.get("strategy") == "baseline"
    }
    candidates = [
        row for row in rows if row.get("strategy") != "baseline"
    ]
    candidates.sort(
        key=lambda row: (
            _to_float(row.get("wer")) - _to_float(baseline_by_clip.get(row.get("clip_id", ""), {}).get("wer")),
            _to_float(row.get("wer")),
        ),
        reverse=True,
    )
    hard_cases = candidates[: max(1, max_items // 2)]
    balanced = _balanced_strategy_sample(candidates, max_items - len(hard_cases), seed)
    selected_by_key = {
        (row.get("clip_id", ""), row.get("strategy", "")): row
        for row in [*hard_cases, *balanced]
    }
    selected = list(selected_by_key.values())[:max_items]
    random.Random(seed).shuffle(selected)
    return selected


def _balanced_strategy_sample(
    rows: list[dict[str, str]],
    count: int,
    seed: int,
) -> list[dict[str, str]]:
    if count <= 0:
        return []
    rng = random.Random(seed)
    by_strategy: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_strategy.setdefault(row.get("strategy", ""), []).append(row)
    for strategy_rows in by_strategy.values():
        rng.shuffle(strategy_rows)
    selected: list[dict[str, str]] = []
    while len(selected) < count and by_strategy:
        for strategy in sorted(list(by_strategy)):
            strategy_rows = by_strategy[strategy]
            if not strategy_rows:
                by_strategy.pop(strategy)
                continue
            selected.append(strategy_rows.pop())
            if len(selected) >= count:
                break
    return selected


def _write_study_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "stimulus_id",
        "clip_id",
        "strategy",
        "sample_path",
        "reference_text",
        "hypothesis_text",
        "wer",
        "cer",
        "speaker_similarity",
        "mos_1_5",
        "intelligibility_1_5",
        "speaker_match_1_5",
        "artifact_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_study_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Subjective Listening Study",
        "",
        "Rate each anonymized sample before looking at strategy labels.",
        "",
        "Suggested scales: MOS 1-5, intelligibility 1-5, speaker match 1-5.",
        "",
        "| Stimulus | Sample | Strategy | WER | Speaker Sim | Notes |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['stimulus_id']}` | "
            f"`{row['sample_path']}` | "
            f"`{row['strategy']}` | "
            f"{row['wer']:.2%} | "
            f"{row['speaker_similarity']:.3f} |  |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _relative_sample_path(value: str) -> str:
    path = Path(value)
    if len(path.parts) >= 2 and path.parts[-2] == "samples":
        return f"samples/{path.name}"
    return path.name if path.name else value


def _to_float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)
