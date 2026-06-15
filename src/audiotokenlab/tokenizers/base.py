from __future__ import annotations

from abc import ABC, abstractmethod

from audiotokenlab.models import AudioClip, TokenBundle


class AudioTokenizer(ABC):
    name: str

    @abstractmethod
    def encode(self, clip: AudioClip) -> TokenBundle:
        raise NotImplementedError

    @abstractmethod
    def decode(self, bundle: TokenBundle) -> tuple[float, ...]:
        raise NotImplementedError

