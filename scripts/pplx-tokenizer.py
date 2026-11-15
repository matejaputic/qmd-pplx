#!/usr/bin/env python3
"""Exact PPLX text budgeting using tokenizers already installed with mlx-vlm.

No model weights, remote code, or downloads. stdout is one JSON response.
"""
import json
from pathlib import Path
import sys

from tokenizers import Tokenizer


def main():
    path = Path(sys.argv[1])
    config = json.loads((path.parent / "config.json").read_text())
    if config.get("model_type") != "pplx_contextual_qwen3_5":
        raise ValueError("Expected the PPLX contextual Qwen3.5 tokenizer/model directory")
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.encode_special_tokens = True  # HF split_special_tokens=True
    tokenizer.no_truncation()
    tokenizer.no_padding()
    request = json.load(sys.stdin)

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids

    def document(texts):
        return len(encode(config["document_prefix"] + config["boundary_marker"].join(texts)))

    op = request["op"]
    if op == "encode":
        result = encode(request["text"])
    elif op == "decode":
        result = tokenizer.decode(request["tokens"], skip_special_tokens=False)
    elif op == "counts":
        texts = request["texts"]
        if request["task"] == "query":
            result = [1 + len(encode(text)) for text in texts]
        elif request["contextual"]:
            result = [document(texts)]
        else:
            result = [document([text]) for text in texts]
    elif op == "windows":
        texts, limit, start, result = request["texts"], request["max_tokens"], 0, []
        # ponytail: re-tokenize each candidate window for exact boundary accounting;
        # bounded by max_tokens (8192); use offset-aware incremental budgeting if needed.
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
    else:
        raise ValueError(f"Unknown operation: {op}")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
