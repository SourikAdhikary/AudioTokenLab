from audiotokenlab.tokenizers.dummy import DummyTokenizer

__all__ = ["DummyTokenizer", "build_tokenizer"]


def build_tokenizer(spec: dict):
    name = spec.get("name", "dummy")
    if name != "dummy":
        raise ValueError(f"Unsupported tokenizer: {name}")
    return DummyTokenizer(
        frame_size=int(spec.get("frame_size", 320)),
        codebook_size=int(spec.get("codebook_size", 256)),
    )

