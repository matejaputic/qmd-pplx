#!/usr/bin/env python3
"""PPLX token operations: legacy local CLI or server-side QMD protocol.

The remote client needs no Python, tokenizer files, or caller-supplied revision.
"""
from contextlib import nullcontext
import hashlib
import inspect
import json
from pathlib import Path
import sys

from tokenizers import Tokenizer


def tokenize(tokenizer, config, request):
    if not isinstance(request, dict):
        raise ValueError("tokenizer must be an object")
    if len(json.dumps(request, ensure_ascii=False).encode()) > 16 * 1024 * 1024:
        raise ValueError("Tokenizer request exceeds 16 MiB")
    op = request.get("op")
    if op == "encode":
        if not isinstance(request.get("text"), str):
            raise ValueError("encode requires text")
    elif op == "decode":
        ids = request.get("tokens")
        if not isinstance(ids, list) or len(ids) > 1_000_000 or any(type(i) is not int or i < 0 or i >= tokenizer.get_vocab_size() for i in ids):
            raise ValueError("decode requires valid token IDs")
    elif op in ("counts", "windows"):
        texts = request.get("texts")
        if not isinstance(texts, list) or len(texts) > 65536 or any(not isinstance(t, str) for t in texts):
            raise ValueError("Expected a list of text strings")
        if op == "windows" and (type(request.get("max_tokens")) is not int or not 1024 <= request["max_tokens"] <= 8192):
            raise ValueError("max_tokens must be between 1024 and 8192")
        if op == "counts" and (request.get("task") not in ("query", "document") or type(request.get("contextual")) is not bool):
            raise ValueError("counts requires task and contextual")
    else:
        raise ValueError(f"Unknown tokenizer operation: {op}")

    tokenizer.encode_special_tokens = True  # HF split_special_tokens=True
    tokenizer.no_truncation()
    tokenizer.no_padding()

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids

    def document(texts):
        return len(encode(config["document_prefix"] + config["boundary_marker"].join(texts)))

    if op == "encode":
        return encode(request["text"])
    if op == "decode":
        return tokenizer.decode(request["tokens"], skip_special_tokens=False)
    if op == "counts":
        if request["task"] == "query":
            return [1 + len(encode(text)) for text in texts]
        return [document(texts)] if request["contextual"] else [document([text]) for text in texts]

    limit, start, result = request["max_tokens"], 0, []
    # ponytail: re-tokenize candidate windows for exact boundary accounting;
    # bounded by 8192 tokens; use offset-aware incremental budgeting if needed.
    for end in range(1, len(texts) + 1):
        if document(texts[start:end]) > limit:
            if end - 1 == start:
                raise ValueError(f"Chunk {start} exceeds the {limit}-token contextual limit")
            result.append([start, end - 1])
            start = end - 1
            if document(texts[start:end]) > limit:
                raise ValueError(f"Chunk {start} exceeds the {limit}-token contextual limit")
    if start < len(texts):
        result.append([start, len(texts)])
    return result


def model_fingerprint(model, model_id):
    """Hash actual artifacts once per loaded model, not a user-entered revision."""
    if getattr(model, "model_type", None) != "pplx_contextual_qwen3_5":
        raise ValueError("QMD remote protocol requires the PPLX contextual model adapter")
    if not hasattr(model, "_qmd_fingerprint"):
        from mlx_vlm.utils import get_model_path
        directory = Path(get_model_path(model_id))
        weights = sorted(directory.glob("*.safetensors"))
        if not weights:
            raise ValueError("No embedding weights found for fingerprinting")
        files = [directory / "config.json", directory / "tokenizer.json", *weights,
                 Path(inspect.getfile(type(model))), Path(__file__)]
        digest = hashlib.sha256(b"qmd-pplx-remote-v1\0")
        for path in files:
            digest.update(path.name.encode() + b"\0")
            with path.open("rb") as stream:
                digest.update(hashlib.file_digest(stream, "sha256").digest())
        model._qmd_fingerprint = digest.hexdigest()
    return model._qmd_fingerprint


def tokenizer_response(model, processor, model_id, request, expected_fingerprint=None):
    if not isinstance(request, dict):
        raise ValueError("tokenizer must be an object")
    # Reuse the encoder's lock: fast-tokenizer settings are shared with inference.
    with getattr(model, "_inference_lock", nullcontext()):
        fingerprint = model_fingerprint(model, model_id)
        if expected_fingerprint is not None and expected_fingerprint != fingerprint:
            raise ValueError("PPLX model changed; restart QMD and rebuild embeddings")
        if request.get("op") == "info":
            limit = min(8192, model._max_tokens, model.config.query_length, model.config.document_length)
            if limit < 1024:
                raise ValueError("QMD requires a server context limit of at least 1024 tokens")
            result = {"fingerprint": fingerprint, "maxTokens": limit}
        else:
            tokenizer = processor.tokenizer.backend_tokenizer
            config = {"document_prefix": model.config.document_prefix,
                      "boundary_marker": model.config.boundary_marker}
            result = tokenize(tokenizer, config, request)
        return {"model": model_id, "fingerprint": fingerprint, "result": result}


def main():
    path = Path(sys.argv[1])
    config = json.loads((path.parent / "config.json").read_text())
    if config.get("model_type") != "pplx_contextual_qwen3_5":
        raise ValueError("Expected the PPLX contextual Qwen3.5 tokenizer/model directory")
    print(json.dumps(tokenize(Tokenizer.from_file(str(path)), config, json.load(sys.stdin)), ensure_ascii=False))


if __name__ == "__main__":
    main()
