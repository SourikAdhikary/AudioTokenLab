from __future__ import annotations

import math

from audiotokenlab.models import AudioClip, TokenBundle
from audiotokenlab.tokenizers.base import AudioTokenizer


class MuLawTokenizer(AudioTokenizer):
    """Classic mu-law scalar quantizer exposed as an audio-token baseline.

    This is not a neural codec. It is a real dependency-free audio quantizer
    that lets the benchmark produce listenable reconstruction artifacts before
    adding EnCodec/Mimi-style backends.
    """

    name = "mulaw"

    def __init__(self, quantization_channels: int = 256, hop_size: int = 1) -> None:
        if quantization_channels < 2:
            raise ValueError("quantization_channels must be at least 2")
        if hop_size <= 0:
            raise ValueError("hop_size must be positive")
        self.quantization_channels = quantization_channels
        self.hop_size = hop_size
        self.mu = quantization_channels - 1

    def encode(self, clip: AudioClip) -> TokenBundle:
        tokens = tuple(
            self._encode_sample(clip.samples[index])
            for index in range(0, len(clip.samples), self.hop_size)
        )
        frame_rate = clip.sample_rate / self.hop_size
        return TokenBundle(
            clip_id=clip.clip_id,
            tokenizer=self.name,
            tokens=tokens,
            frame_rate=frame_rate,
            codebook_count=1,
            sample_rate=clip.sample_rate,
            duration_seconds=clip.duration_seconds,
            metadata={
                "quantization_channels": self.quantization_channels,
                "hop_size": self.hop_size,
            },
        )

    def decode(self, bundle: TokenBundle) -> tuple[float, ...]:
        hop_size = int(bundle.metadata.get("hop_size", self.hop_size))
        samples: list[float] = []
        for token in bundle.tokens:
            sample = self._decode_token(token)
            samples.extend([sample] * hop_size)

        target_len = int(round(bundle.duration_seconds * bundle.sample_rate))
        if len(samples) < target_len:
            samples.extend([0.0] * (target_len - len(samples)))
        return tuple(samples[:target_len])

    def _encode_sample(self, sample: float) -> int:
        clipped = max(-1.0, min(1.0, sample))
        sign = 1.0 if clipped >= 0.0 else -1.0
        magnitude = math.log1p(self.mu * abs(clipped)) / math.log1p(self.mu)
        encoded = sign * magnitude
        normalized = (encoded + 1.0) / 2.0
        return max(
            0,
            min(self.quantization_channels - 1, round(normalized * self.mu)),
        )

    def _decode_token(self, token: int) -> float:
        token = max(0, min(self.quantization_channels - 1, token))
        encoded = 2.0 * (token / self.mu) - 1.0
        sign = 1.0 if encoded >= 0.0 else -1.0
        magnitude = (math.expm1(abs(encoded) * math.log1p(self.mu))) / self.mu
        return sign * magnitude

