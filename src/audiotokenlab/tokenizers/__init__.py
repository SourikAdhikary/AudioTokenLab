from audiotokenlab.tokenizers.dummy import DummyTokenizer
from audiotokenlab.tokenizers.encodec_backend import EncodecTokenizer
from audiotokenlab.tokenizers.mulaw import MuLawTokenizer

__all__ = ["DummyTokenizer", "EncodecTokenizer", "MuLawTokenizer", "build_tokenizer"]


def build_tokenizer(spec: dict):
    name = spec.get("name", "dummy")
    if name == "dummy":
        return DummyTokenizer(
            frame_size=int(spec.get("frame_size", 320)),
            codebook_size=int(spec.get("codebook_size", 256)),
        )
    if name == "mulaw":
        return MuLawTokenizer(
            quantization_channels=int(spec.get("quantization_channels", 256)),
            hop_size=int(spec.get("hop_size", 1)),
        )
    if name == "encodec":
        return EncodecTokenizer(
            model_name=str(spec.get("model_name", "encodec_24khz")),
            bandwidth=float(spec.get("bandwidth", 6.0)),
            device=str(spec.get("device", "cpu")),
        )
    raise ValueError(f"Unsupported tokenizer: {name}")
