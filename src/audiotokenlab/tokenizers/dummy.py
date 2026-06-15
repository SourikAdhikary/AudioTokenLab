from __future__ import annotations

from audiotokenlab.models import AudioClip, TokenBundle
from audiotokenlab.tokenizers.base import AudioTokenizer


class DummyTokenizer(AudioTokenizer):
    """Deterministic amplitude tokenizer used to validate the pipeline.

    This is not an audio codec. It approximates the shape of a codec backend:
    waveform frames become discrete integer tokens, and tokens can be decoded
    back into a coarse waveform for metric smoke tests.
    """

    name = "dummy"

    def __init__(self, frame_size: int = 320, codebook_size: int = 256) -> None:
        if frame_size <= 0:
            raise ValueError("frame_size must be positive")
        if codebook_size < 2:
            raise ValueError("codebook_size must be at least 2")
        self.frame_size = frame_size
        self.codebook_size = codebook_size

    def encode(self, clip: AudioClip) -> TokenBundle:
        tokens: list[int] = []
        for start in range(0, len(clip.samples), self.frame_size):
            frame = clip.samples[start : start + self.frame_size]
            if not frame:
                continue
            mean_abs = sum(abs(value) for value in frame) / len(frame)
            token = round(mean_abs * (self.codebook_size - 1))
            tokens.append(max(0, min(self.codebook_size - 1, token)))

        frame_rate = clip.sample_rate / self.frame_size
        return TokenBundle(
            clip_id=clip.clip_id,
            tokenizer=self.name,
            tokens=tuple(tokens),
            frame_rate=frame_rate,
            codebook_count=1,
            sample_rate=clip.sample_rate,
            duration_seconds=clip.duration_seconds,
            metadata={
                "frame_size": self.frame_size,
                "codebook_size": self.codebook_size,
            },
        )

    def decode(self, bundle: TokenBundle) -> tuple[float, ...]:
        frame_size = int(bundle.metadata.get("frame_size", self.frame_size))
        codebook_size = int(bundle.metadata.get("codebook_size", self.codebook_size))
        samples: list[float] = []
        for token in bundle.tokens:
            amplitude = (token / max(1, codebook_size - 1)) * 2.0 - 1.0
            samples.extend([amplitude] * frame_size)

        target_len = int(round(bundle.duration_seconds * bundle.sample_rate))
        if len(samples) < target_len:
            samples.extend([0.0] * (target_len - len(samples)))
        return tuple(samples[:target_len])

