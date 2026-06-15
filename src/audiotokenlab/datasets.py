from __future__ import annotations

import json
import math
import wave
from pathlib import Path

from audiotokenlab.models import AudioClip


def load_dataset(spec: dict) -> list[AudioClip]:
    dataset_type = spec.get("type", "synthetic")
    if dataset_type == "synthetic":
        return _load_synthetic_dataset(spec)
    if dataset_type == "synthetic_quiet":
        return _load_synthetic_quiet_dataset(spec)
    if dataset_type == "wav_dir":
        return _load_wav_dir(spec)
    if dataset_type == "wav_manifest":
        return _load_wav_manifest(spec)
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


def _load_synthetic_quiet_dataset(spec: dict) -> list[AudioClip]:
    count = int(spec.get("count", 3))
    sample_rate = int(spec.get("sample_rate", 16000))
    duration_seconds = float(spec.get("duration_seconds", 1.5))
    base_frequency = float(spec.get("base_frequency", 220.0))
    speech_seconds = float(spec.get("speech_seconds", 0.28))
    quiet_seconds = float(spec.get("quiet_seconds", 0.22))
    clips: list[AudioClip] = []

    for index in range(count):
        frequency = base_frequency * (1.0 + index * 0.1)
        samples = _alternating_tone_and_quiet(
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            frequency=frequency,
            speech_seconds=speech_seconds,
            quiet_seconds=quiet_seconds,
        )
        clips.append(
            AudioClip(
                clip_id=f"synthetic_quiet_{index:03d}",
                samples=tuple(samples),
                sample_rate=sample_rate,
                source="synthetic_quiet",
                metadata={
                    "frequency": frequency,
                    "speech_seconds": speech_seconds,
                    "quiet_seconds": quiet_seconds,
                },
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


def _alternating_tone_and_quiet(
    sample_rate: int,
    duration_seconds: float,
    frequency: float,
    speech_seconds: float,
    quiet_seconds: float,
) -> list[float]:
    total_samples = max(1, int(sample_rate * duration_seconds))
    speech_samples = max(1, int(sample_rate * speech_seconds))
    quiet_samples = max(1, int(sample_rate * quiet_seconds))
    cycle_samples = speech_samples + quiet_samples
    samples: list[float] = []

    for sample_index in range(total_samples):
        cycle_index = sample_index % cycle_samples
        if cycle_index >= speech_samples:
            samples.append(0.0)
            continue
        t = sample_index / sample_rate
        local_phase = cycle_index / speech_samples
        ramp = min(1.0, local_phase * 10.0, (1.0 - local_phase) * 10.0)
        carrier = math.sin(2.0 * math.pi * frequency * t)
        harmonic = 0.2 * math.sin(2.0 * math.pi * frequency * 2.0 * t)
        samples.append(max(-1.0, min(1.0, (carrier + harmonic) * ramp * 0.7)))
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


def _load_wav_manifest(spec: dict) -> list[AudioClip]:
    manifest_path = Path(spec["path"])
    if not manifest_path.is_absolute():
        manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(manifest.get("root", manifest_path.parent))
    if not root.is_absolute():
        root = (manifest_path.parent / root).resolve()

    clips: list[AudioClip] = []
    for item in manifest.get("clips", []):
        audio_path = Path(item["path"])
        if not audio_path.is_absolute():
            audio_path = root / audio_path
        clip = _read_wav(audio_path)
        metadata = dict(clip.metadata)
        metadata.update({key: value for key, value in item.items() if key != "path"})
        clips.append(
            AudioClip(
                clip_id=str(item.get("clip_id", clip.clip_id)),
                samples=clip.samples,
                sample_rate=clip.sample_rate,
                source=str(audio_path),
                metadata=metadata,
            )
        )
    if not clips:
        raise ValueError(f"No clips found in manifest {manifest_path}")
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
