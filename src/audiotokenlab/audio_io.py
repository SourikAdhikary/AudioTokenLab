from __future__ import annotations

import wave
from pathlib import Path


def write_wav(path: Path, samples: tuple[float, ...], sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(_to_pcm16(samples))


def _to_pcm16(samples: tuple[float, ...]) -> bytes:
    data = bytearray()
    for sample in samples:
        clipped = max(-1.0, min(1.0, sample))
        value = int(round(clipped * 32767.0))
        data.extend(value.to_bytes(2, "little", signed=True))
    return bytes(data)

