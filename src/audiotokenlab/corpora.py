from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audiotokenlab.audio_io import write_wav


DEFAULT_HF_SPEECH_SOURCES = [
    {
        "name": "minds14_en_us",
        "dataset": "PolyAI/minds14",
        "config": "en-US",
        "split": "train",
        "audio_column": "audio",
        "text_column": "transcription",
    },
    {
        "name": "fleurs_en_us",
        "dataset": "google/fleurs",
        "config": "en_us",
        "split": "validation",
        "audio_column": "audio",
        "text_column": "transcription",
    },
]


def prepare_huggingface_speech_slice(
    output_dir: Path,
    sources: list[dict[str, Any]] | None = None,
    max_clips_per_source: int = 8,
    sample_rate: int = 24000,
) -> Path:
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise ImportError(
            "Hugging Face dataset support requires `pip install -e '.[datasets]'`."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = output_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_clips: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for source in sources or DEFAULT_HF_SPEECH_SOURCES:
        source_name = str(source["name"])
        dataset_name = str(source["dataset"])
        config_name = source.get("config")
        split = str(source.get("split", "validation"))
        audio_column = str(source.get("audio_column", "audio"))
        text_column = str(source.get("text_column", "text"))
        try:
            dataset = load_dataset(
                dataset_name,
                config_name,
                split=split,
                streaming=bool(source.get("streaming", True)),
                trust_remote_code=bool(source.get("trust_remote_code", False)),
            )
            dataset = dataset.cast_column(audio_column, Audio(sampling_rate=sample_rate))
            selected = 0
            for index, item in enumerate(dataset):
                audio = item.get(audio_column)
                transcript = str(item.get(text_column, "")).strip()
                if not audio or not transcript:
                    continue
                samples, decoded_sample_rate = _decode_audio(audio, sample_rate)
                if not samples:
                    continue
                clip_id = _safe_clip_id(source_name, selected)
                wav_path = wav_dir / f"{clip_id}.wav"
                write_wav(wav_path, samples, decoded_sample_rate)
                manifest_clips.append(
                    {
                        "clip_id": clip_id,
                        "path": str(wav_path),
                        "transcript": transcript,
                        "source": source_name,
                        "hf_dataset": dataset_name,
                        "hf_config": config_name or "",
                        "hf_split": split,
                        "source_index": index,
                        "license": str(source.get("license", "see upstream dataset card")),
                    }
                )
                selected += 1
                if selected >= max_clips_per_source:
                    break
        except Exception as exc:
            failures.append(
                {
                    "source": source_name,
                    "dataset": dataset_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if not manifest_clips:
        failure_path = output_dir / "source_failures.json"
        failure_path.write_text(
            json.dumps(failures, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        raise ValueError(
            "No Hugging Face speech clips were prepared. "
            f"Failures were written to {failure_path}: {failures}"
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "root": str(wav_dir),
                "clips": manifest_clips,
                "source_failures": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def merge_wav_manifests(
    manifest_paths: list[Path],
    output_path: Path,
    root: Path | None = None,
) -> Path:
    clips: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_root = Path(manifest.get("root", manifest_path.parent))
        if not manifest_root.is_absolute():
            manifest_root = (manifest_path.parent / manifest_root).resolve()
        for item in manifest.get("clips", []):
            merged = dict(item)
            audio_path = Path(str(merged["path"]))
            if not audio_path.is_absolute():
                audio_path = manifest_root / audio_path
            merged["path"] = str(audio_path)
            clips.append(merged)

    if not clips:
        raise ValueError("Cannot merge empty manifests")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"root": str(root or output_path.parent), "clips": clips}, indent=2),
        encoding="utf-8",
    )
    return output_path


def _decode_audio(audio: Any, fallback_sample_rate: int) -> tuple[tuple[float, ...], int]:
    if hasattr(audio, "get_all_samples"):
        decoded = audio.get_all_samples()
        sample_rate = int(getattr(decoded, "sample_rate", fallback_sample_rate))
        return _samples_from_array(getattr(decoded, "data", ())), sample_rate
    if not isinstance(audio, dict):
        return (), fallback_sample_rate
    array = audio.get("array")
    if array is None:
        return (), int(audio.get("sampling_rate", fallback_sample_rate))
    return _samples_from_array(array), int(audio.get("sampling_rate", fallback_sample_rate))


def _samples_from_array(array: Any) -> tuple[float, ...]:
    if hasattr(array, "tolist"):
        array = array.tolist()
    if not array:
        return ()
    if isinstance(array[0], list):
        channel_count = len(array)
        sample_count = min(len(channel) for channel in array) if channel_count else 0
        mono = []
        for sample_index in range(sample_count):
            mono.append(
                sum(float(array[channel][sample_index]) for channel in range(channel_count))
                / channel_count
            )
        array = mono
    return tuple(max(-1.0, min(1.0, float(value))) for value in array)


def _safe_clip_id(source_name: str, index: int) -> str:
    safe_source = "".join(
        character if character.isalnum() else "_"
        for character in source_name.lower()
    ).strip("_")
    return f"{safe_source}_{index:04d}"
