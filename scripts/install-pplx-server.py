#!/usr/bin/env python3
"""Run with the MLX server's Python, then restart mlx_vlm.server.

Extends the existing PPLX embedding patch; no client-side Python is needed.
"""
from pathlib import Path
import shutil
import sysconfig

root = Path(__file__).resolve().parent
server = Path(sysconfig.get_paths()["purelib"]) / "mlx_vlm/server"
path = server / "embeddings.py"
text = path.read_text()
if "model.embed_request(processor, body)" not in text:
    raise SystemExit("Install the PPLX model adapter/basic embedding server patch first")
edits = [
    ("from .runtime import runtime", "from .runtime import runtime\nfrom contextlib import nullcontext\nfrom .pplx_tokenizer import model_fingerprint, tokenizer_response"),
    ("    chunks: Optional[int] = Field(None, gt=0, strict=True)",
     "    chunks: Optional[int] = Field(None, gt=0, strict=True)\n    tokenizer: Optional[Dict[str, Any]] = None\n    expected_fingerprint: Optional[str] = None"),
    ("        request_start = time.perf_counter()\n        runtime.metrics.begin_request(",
     "        if body.tokenizer is not None:\n"
     "            def _tokenize():\n"
     "                model, processor, *_ = get_cached_model(model_id, model_kind=\"embedding\")\n"
     "                return tokenizer_response(model, processor, model_id, body.tokenizer, body.expected_fingerprint)\n"
     "            try:\n"
     "                return await asyncio.to_thread(_tokenize)\n"
     "            except ValueError as exc:\n"
     "                raise HTTPException(status_code=400, detail=str(exc)) from exc\n\n"
     "        request_start = time.perf_counter()\n        runtime.metrics.begin_request("),
    ("                    if hasattr(model, \"embed_request\"):\n                        return model.embed_request(processor, body)",
     "                    if body.expected_fingerprint is not None:\n"
     "                        with getattr(model, \"_inference_lock\", nullcontext()):\n"
     "                            if model_fingerprint(model, model_id) != body.expected_fingerprint:\n"
     "                                raise ValueError(\"PPLX model changed; restart QMD and rebuild embeddings\")\n"
     "                    if hasattr(model, \"embed_request\"):\n                        return model.embed_request(processor, body)"),
]
for old, new in edits:
    if new in text:
        continue
    if text.count(old) != 1:
        raise SystemExit(f"Incompatible mlx-vlm version; no files changed (missing {old!r})")
    text = text.replace(old, new, 1)
compile(text, str(path), "exec")
backup = path.with_suffix(".py.qmd-backup")
if not backup.exists():
    shutil.copyfile(path, backup)
shutil.copyfile(root / "pplx-tokenizer.py", server / "pplx_tokenizer.py")
path.write_text(text)
print(f"Installed QMD remote tokenizer support in {server}; restart mlx_vlm.server")
