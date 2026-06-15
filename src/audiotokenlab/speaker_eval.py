from __future__ import annotations

import csv
import json
from pathlib import Path


def evaluate_speaker_similarity_with_speechbrain(
    sample_dir: Path,
    device: str = "cpu",
    model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
) -> list[dict]:
    import torch

    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier

    classifier = EncoderClassifier.from_hparams(
        source=model_source,
        savedir=str(Path("/tmp") / "audiotokenlab_speaker_model"),
        run_opts={"device": device},
    )
    by_clip: dict[str, dict[str, Path]] = {}
    for path in sorted(sample_dir.glob("*.wav")):
        clip_id, strategy = _parse_sample_name(path)
        by_clip.setdefault(clip_id, {})[strategy] = path

    rows: list[dict] = []
    for clip_id in sorted(by_clip):
        strategy_paths = by_clip[clip_id]
        baseline_path = strategy_paths.get("baseline")
        if baseline_path is None:
            continue
        baseline_embedding = _speaker_embedding(classifier, baseline_path, torch)
        for strategy in sorted(strategy_paths):
            path = strategy_paths[strategy]
            embedding = _speaker_embedding(classifier, path, torch)
            similarity = torch.nn.functional.cosine_similarity(
                baseline_embedding.reshape(1, -1),
                embedding.reshape(1, -1),
            ).detach().cpu()
            similarity_value = max(-1.0, min(1.0, float(similarity.item())))
            rows.append(
                {
                    "clip_id": clip_id,
                    "strategy": strategy,
                    "sample_path": str(path),
                    "reference_strategy": "baseline",
                    "speaker_similarity": similarity_value,
                    "model_source": model_source,
                }
            )
    return rows


def write_speaker_artifacts(output_dir: Path, rows: list[dict]) -> None:
    from audiotokenlab.reporting import write_asr_dashboard

    write_speaker_csv(output_dir / "speaker_metrics.csv", rows)
    (output_dir / "speaker_summary.json").write_text(
        json.dumps(summarize_speaker(rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    asr_rows = _read_csv_dicts(output_dir / "asr_metrics.csv")
    if asr_rows:
        write_asr_dashboard(output_dir, asr_rows)


def write_speaker_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_speaker(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["strategy"]), []).append(row)

    by_strategy: dict[str, dict] = {}
    for strategy in sorted(grouped):
        strategy_rows = grouped[strategy]
        by_strategy[strategy] = {
            "row_count": len(strategy_rows),
            "mean_speaker_similarity": sum(
                float(row["speaker_similarity"]) for row in strategy_rows
            )
            / len(strategy_rows),
        }

    if not rows:
        return {"row_count": 0, "strategy_summary": {}}
    return {
        "row_count": len(rows),
        "mean_speaker_similarity": sum(
            float(row["speaker_similarity"]) for row in rows
        )
        / len(rows),
        "strategy_summary": by_strategy,
    }


def _speaker_embedding(classifier: object, path: Path, torch_module: object) -> object:
    signal = classifier.load_audio(str(path))
    if signal.ndim == 1:
        signal = signal.unsqueeze(0)
    device = getattr(classifier, "device", None)
    if device is not None:
        signal = signal.to(device)
    with torch_module.no_grad():
        return classifier.encode_batch(signal).squeeze()


def _parse_sample_name(path: Path) -> tuple[str, str]:
    stem = path.stem
    if "__" not in stem:
        return stem, "unknown"
    clip_id, strategy = stem.rsplit("__", 1)
    return clip_id, strategy


def _read_csv_dicts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
