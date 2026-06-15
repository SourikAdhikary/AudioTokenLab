from __future__ import annotations

from typing import Any

from audiotokenlab.models import AudioClip, TokenBundle
from audiotokenlab.tokenizers.base import AudioTokenizer


class EncodecTokenizer(AudioTokenizer):
    """Optional EnCodec backend.

    The dependency is intentionally optional so the default package stays light
    and local tests can run without PyTorch. Install `audiotokenlab[encodec]`
    before using this tokenizer.
    """

    name = "encodec"

    def __init__(
        self,
        model_name: str = "encodec_24khz",
        bandwidth: float = 6.0,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.bandwidth = bandwidth
        self.device = device
        self._torch = _import_required("torch")
        encodec_module = _import_required("encodec")
        model_cls = getattr(encodec_module, "EncodecModel")
        if model_name != "encodec_24khz":
            raise ValueError("Only encodec_24khz is supported in the first backend")
        self.model = model_cls.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        self.model.to(device)
        self.model.eval()
        self.sample_rate = 24000

    def encode(self, clip: AudioClip) -> TokenBundle:
        if clip.sample_rate != self.sample_rate:
            raise ValueError(
                f"{self.name} requires {self.sample_rate} Hz audio in v1; "
                f"got {clip.sample_rate} Hz"
            )

        wav = self._torch.tensor(clip.samples, dtype=self._torch.float32).view(1, 1, -1)
        wav = wav.to(self.device)
        with self._torch.no_grad():
            encoded_frames = self.model.encode(wav)

        if len(encoded_frames) != 1:
            raise ValueError("Chunked EnCodec frames are not supported yet")

        codes, scale = encoded_frames[0]
        codes = codes.detach().cpu()
        if codes.ndim != 3 or codes.shape[0] != 1:
            raise ValueError(f"Unexpected EnCodec code shape: {tuple(codes.shape)}")

        codebook_count = int(codes.shape[1])
        frame_count = int(codes.shape[2])
        tokens: list[int] = []
        for frame_index in range(frame_count):
            for codebook_index in range(codebook_count):
                tokens.append(int(codes[0, codebook_index, frame_index].item()))

        scale_value: float | None
        if scale is None:
            scale_value = None
        else:
            scale_value = float(scale.detach().cpu().reshape(-1)[0].item())

        return TokenBundle(
            clip_id=clip.clip_id,
            tokenizer=self.name,
            tokens=tuple(tokens),
            frame_rate=frame_count / max(clip.duration_seconds, 1e-9),
            codebook_count=codebook_count,
            sample_rate=clip.sample_rate,
            duration_seconds=clip.duration_seconds,
            metadata={
                "token_layout": "frame_major",
                "frame_count": frame_count,
                "model_name": self.model_name,
                "bandwidth": self.bandwidth,
                "scale": scale_value,
            },
        )

    def decode(self, bundle: TokenBundle) -> tuple[float, ...]:
        if bundle.metadata.get("token_layout") != "frame_major":
            raise ValueError("EnCodec backend requires frame_major token layout")
        if len(bundle.tokens) % bundle.codebook_count != 0:
            raise ValueError("Token count must be divisible by codebook_count")

        frame_count = len(bundle.tokens) // bundle.codebook_count
        codes = self._torch.tensor(
            bundle.tokens,
            dtype=self._torch.long,
            device=self.device,
        ).view(1, frame_count, bundle.codebook_count)
        codes = codes.permute(0, 2, 1).contiguous()

        scale = bundle.metadata.get("scale")
        scale_tensor: Any
        if scale is None:
            scale_tensor = None
        else:
            scale_tensor = self._torch.tensor([scale], device=self.device).view(1, 1)

        with self._torch.no_grad():
            decoded = self.model.decode([(codes, scale_tensor)])
        samples = decoded.detach().cpu().reshape(-1).tolist()

        target_len = int(round(bundle.duration_seconds * bundle.sample_rate))
        if len(samples) < target_len:
            samples.extend([0.0] * (target_len - len(samples)))
        return tuple(float(value) for value in samples[:target_len])


def _import_required(module_name: str) -> Any:
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise ImportError(
            "EnCodec support requires optional dependencies. "
            "Install with `python3 -m pip install -e '.[encodec]'`."
        ) from exc
