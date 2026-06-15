from __future__ import annotations

import math
import wave
from pathlib import Path

from audiotokenlab.models import AudioClip


def load_dataset(spec: dict) -> list[AudioClip]:
    dataset_type = spec.get("type", "synthetic")
    if dataset_type == "synthetic":
        return _load_synthetic_dataset(spec)
    if dataset_type == "wav_dir":
        return _load_wav_dir(spec)
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def _load_synthetic_dataset(spec: dict) -> list[AudioClip]:
    count = int(spec.get("count", 3))
    sample_rate = int(spec.get("sample_rate", 16000))
    duration_seconds = float(spec.get("duration_seconds", 1.0))
    base_frequency = float(spec.get("base_frequency", 220.0))
    clips: list[AudioClip] = []

    for index in range(count):
        frequency = base_frequency * (1.0 + index * 0.125)
        samples = _sine_with_envelope(
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            frequency=frequency,
        )
        clips.append(
            AudioClip(
                clip_id=f"synthetic_{index:03d}",
                samples=tuple(samples),
                sample_rate=sample_rate,
                source="synthetic",
                metadata={"frequency": frequency},
            )
        )
    return clips


def _sine_with_envelope(
    sample_rate: int,
    duration_seconds: float,
    frequency: float,
) -> list[float]:
    total_samples = max(1, int(sample_rate * duration_seconds))
    samples: list[float] = []
    for sample_index in range(total_samples):
        t = sample_index / sample_rate
        carrier = math.sin(2.0 * math.pi * frequency * t)
        harmonic = 0.25 * math.sin(2.0 * math.pi * frequency * 2.0 * t)
        envelope = 0.6 + 0.4 * math.sin(2.0 * math.pi * 3.0 * t)
        samples.append(max(-1.0, min(1.0, (carrier + harmonic) * envelope * 0.65)))
    return samples


def _load_wav_dir(spec: dict) -> list[AudioClip]:
    root = Path(spec["path"])
    max_clips = int(spec.get("max_clips", 1000))
    clips: list[AudioClip] = []
    for path in sorted(root.glob("*.wav"))[:max_clips]:
        clips.append(_read_wav(path))
    if not clips:
        raise ValueError(f"No .wav files found in {root}")
    return clips


def _read_wav(path: Path) -> AudioClip:
    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        sample_rate = reader.getframerate()
        sample_width = reader.getsampwidth()
        frame_count = reader.getnframes()
        raw = reader.readframes(frame_count)

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported in v1: {path}")

    samples: list[float] = []
    for offset in range(0, len(raw), sample_width * channels):
        channel_values: list[int] = []
        for channel in range(channels):
            start = offset + channel * sample_width
            value = int.from_bytes(raw[start : start + sample_width], "little", signed=True)
            channel_values.append(value)
        mono = sum(channel_values) / len(channel_values)
        samples.append(mono / 32768.0)

    return AudioClip(
        clip_id=path.stem,
        samples=tuple(samples),
        sample_rate=sample_rate,
        source=str(path),
        metadata={"channels": channels, "sample_width": sample_width},
    )

