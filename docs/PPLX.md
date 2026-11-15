# PPLX embeddings through MLX

This fork adds **text embeddings** from `perplexity-ai/pplx-embed-v2-context-9b`
served by an MLX embedding server. QMD keeps its existing SQLite/BM25/vector
retrieval, query expansion, RRF fusion, metadata filters, and GGUF reranker.
The upstream GGUF embedding backend is unchanged when a `pplx:` model is not selected.

## Prerequisites

- This fork, not the upstream `@tobilu/qmd` package.
- An MLX conversion of the **non-preview** PPLX model with its matching
  `config.json` and `tokenizer.json` beside the weights.
- `mlx_vlm.server` with the PPLX model adapter and extended embedding endpoint.
  **Vanilla mlx-vlm without that adapter is not sufficient.** On the machine this
  integration was developed for, that installation already exists; its conversion,
  server patch, parity checks, and examples are in `~/Projects/pplx-embed-mlx/README.md`.
- Python with `tokenizers`, already included in that mlx-vlm environment.
  No new Python or JavaScript dependencies are introduced by this backend.

QMD sends text to the configured endpoint. Prefer loopback plus an SSH tunnel
for remote access. If binding to a LAN interface, restrict access to trusted
clients with a firewall; the service is unauthenticated. Do not expose it to the
public Internet. Changing the endpoint is an explicit data-routing decision,
not an automatic fallback.

## Setup on the existing machine

```bash
cd ~/Projects/qmd-pplx
bun install
bun run build
```

Start the existing PPLX server in another terminal:

```bash
~/Projects/pplx-embed-mlx/serve.sh
```

Or use the server executable directly:

```bash
~/.local/bin/mlx_vlm.server \
  --embedding-model "$HOME/models/pplx-embed-v2-context-9b-MLX-8bit" \
  --host 127.0.0.1 --port 8080
```

Configure the QMD process:

```bash
export QMD_PPLX_PYTHON="$HOME/.local/share/uv/tools/mlx-vlm/bin/python"
export QMD_PPLX_URL="http://127.0.0.1:8080/v1/embeddings"
export QMD_EMBED_MODEL="pplx:$HOME/models/pplx-embed-v2-context-9b-MLX-8bit?revision=dab89ff203fc943c4ca16e51c5882ff92790d758-q8&dimensions=1024&mode=contextual&max_tokens=8192"
```

Use a **new named index**, leaving your previous QMD index untouched:

```bash
# Run these yourself when ready to index your collection.
~/Projects/qmd-pplx/bin/qmd --index pplx collection add ~/notes --name notes
~/Projects/qmd-pplx/bin/qmd --index pplx update
~/Projects/qmd-pplx/bin/qmd --index pplx embed
~/Projects/qmd-pplx/bin/qmd --index pplx vsearch "How do I configure authentication?"
~/Projects/qmd-pplx/bin/qmd --index pplx query "How do I configure authentication?"
~/Projects/qmd-pplx/bin/qmd --index pplx doctor
```

Use the fork's executable explicitly; installing upstream with
`npm install -g @tobilu/qmd` will not install these changes. No global executable
or existing personal collection was replaced during development.

`vsearch` needs only the running embedding server. `query` additionally uses
QMD's original GGUF expansion and reranking models. `pull` downloads those
GGUF models but skips `pplx:` entries: MLX weights are managed outside QMD.

### Existing named indexes and configuration precedence

QMD's `models.embed` in an index's YAML **takes precedence over**
`QMD_EMBED_MODEL`. The environment variable selects the initial model for a
new index; it does not override a model already saved in that index.

For an existing named index, set the actual absolute path in its YAML, for example
`~/.config/qmd/pplx.yml`:

```yaml
models:
  embed: "pplx:/Users/YOU/models/pplx-embed-v2-context-9b-MLX-8bit?revision=dab89ff203fc943c4ca16e51c5882ff92790d758-q8&dimensions=1024&mode=contextual&max_tokens=8192"
```

YAML values do not expand `$HOME` or `~`. Leave `generate` and `rerank` unchanged.
Normal QMD project-local configuration trust rules still apply.

## Remote server, local tokenizer

The QMD client does **not** need MLX or the model weights. It needs Python with
`tokenizers` and matching `config.json` / `tokenizer.json` files. The URI path
is the **client-local tokenizer directory**; optional `remote_model` is the
**server's model ID**, sent in the HTTP request and checked in the response.
They can be completely different paths (or the remote ID can be a served alias).

First update the fork on the client (`git pull --ff-only && bun run build` for a
source checkout). On the embedding host, keep the patched MLX server running.
For direct LAN access, start it with `--host <HOST_LAN_IP>` instead of localhost
and allow port 8080 only from trusted clients.

On the client, replace the example host/address and use the exact remote ID
reported by the host's `/v1/models` endpoint:

```bash
SERVER="macos@192.168.1.100"
REMOTE_MODEL="/Users/macos/models/pplx-embed-v2-context-9b-MLX-8bit"
TOKENIZER_DIR="$HOME/.local/share/qmd/pplx-tokenizer"

mkdir -p "$TOKENIZER_DIR"
scp "${SERVER}:${REMOTE_MODEL}/config.json" "$TOKENIZER_DIR/"
scp "${SERVER}:${REMOTE_MODEL}/tokenizer.json" "$TOKENIZER_DIR/"

# CLIENT-side Python with tokenizers installed; no MLX installation needed.
export QMD_PPLX_PYTHON="/absolute/path/to/client/python"
export QMD_PPLX_URL="http://192.168.1.100:8080/v1/embeddings"
export QMD_EMBED_MODEL="pplx:$TOKENIZER_DIR?revision=dab89ff203fc943c4ca16e51c5882ff92790d758-q8&remote_model=$REMOTE_MODEL&dimensions=1024&mode=contextual&max_tokens=8192"
```

If the client has no Python environment with `tokenizers`, create one:

```bash
python3 -m venv "$HOME/.venvs/qmd-pplx"
"$HOME/.venvs/qmd-pplx/bin/python" -m pip install tokenizers
export QMD_PPLX_PYTHON="$HOME/.venvs/qmd-pplx/bin/python"
```

For an SSH tunnel instead of exposing the service on the LAN, leave the server
bound to `127.0.0.1:8080`, run this in a separate client terminal:

```bash
ssh -N -L 18080:127.0.0.1:8080 macos@192.168.1.100
```

Then use `QMD_PPLX_URL=http://127.0.0.1:18080/v1/embeddings`. The `remote_model`
value is unchanged: it identifies the model on the server, not the tunnel.

For an existing named index, update its YAML `models.embed`, not just the
environment variable. For example, on a Linux client with user `alice`:

```yaml
models:
  embed: "pplx:/home/alice/.local/share/qmd/pplx-tokenizer?revision=dab89ff203fc943c4ca16e51c5882ff92790d758-q8&remote_model=/Users/macos/models/pplx-embed-v2-context-9b-MLX-8bit&dimensions=1024&mode=contextual&max_tokens=8192"
```

Use the normal named-index workflow above. Only embedding inference runs on the
server; the database, tokenization, GGUF query expansion and reranking remain on
the QMD client. Native expansion/reranking support still depends on the client's
platform; `vsearch` needs no client-side GGUF models.

Without `remote_model`, the existing behavior and fingerprints are unchanged:
the URI path is used both locally and as the HTTP model ID. Adding/changing
`remote_model` (or moving the local tokenizer directory) changes the embedding
profile; use a fresh named index or rebuild all collections once with `embed --force`.
Keep the tokenizer metadata synchronized with the hosted weights/revision.

## Model URI and transport settings

```text
pplx:/absolute/local/tokenizer/directory?revision=WEIGHTS-AND-TOKENIZER-ID&remote_model=SERVER-MODEL-ID&dimensions=1024&mode=contextual&max_tokens=8192
```

| URI parameter | Values | Default |
| --- | --- | --- |
| `revision` | Required nonempty identifier; change whenever weights/tokenizer change | None |
| `remote_model` | Optional nonempty server model ID; no control characters | URI's local directory path |
| `dimensions` | `1024`, `2048` | `1024` |
| `mode` | `independent`, `contextual` | `contextual` |
| `max_tokens` | Integer from `1024` through `8192` | `8192` |

Unknown/duplicate parameters are rejected. Percent-encode reserved characters in
paths (notably `?` and `#`) and query values (notably `&`, `#`, `%` and `+`).
The server ID in the example contains only path-safe characters; use
`URLSearchParams.set("remote_model", id)` when building URIs for arbitrary IDs.
`revision` is an index identity supplied by you,
**not a download instruction or automatic cryptographic verification of weights**.
Use a new revision when replacing files in place.

Transport-only environment variables:

- `QMD_PPLX_URL`: complete embedding endpoint; defaults to
  `http://127.0.0.1:8080/v1/embeddings`.
- `QMD_PPLX_PYTHON`: executable with `tokenizers`; defaults to `python3`.

Vectors are always normalized. Indexing and queries use the same dimensions.
HTTP requests time out after 180 seconds; tokenizer calls after 60 seconds.
HTTP errors, incorrect model/vector counts, invalid dimensions, nonfinite or
non-unit vectors, and inconsistent response mappings fail explicitly. There is
no fallback to another embedding model and no silent truncation.

## What contextual indexing does

1. Use QMD's Markdown-aware regex boundaries, with a **900-token ceiling and
   zero overlap**, verified using this model's tokenizer.
2. Keep the title as ordinary text (`title + newline + passage`), without
   EmbeddingGemma or Qwen instruction prefixes.
3. Greedily pack consecutive chunks from **one document** into windows.
4. Count the exact encoded document prefix, separators, titles and text when
   enforcing `max_tokens`. Match this ceiling to the server's configured limit.
5. Submit `{"input":{"chunks":[...]},"task":"document",...}` and store one
   vector per chunk, preserving QMD's original hash, sequence and character offset.
6. Retry an entire failed contextual window, never a chunk in isolation.
   Window writes are transactional; incomplete documents are not treated as
   successfully embedded.

A top-level list of strings means **independent** embeddings, not shared context.
That is what `mode=independent` uses for documents. Both profiles use the same
non-overlapping chunker, making them useful for an initial controlled comparison.

Queries always use `task: "query"`, including expanded/HyDE queries in hybrid
and structured search. Query strings are not joined into a document context.
The MLX adapter adds the learned PPLX prefixes exactly once.

`doctor` and legacy-fingerprint adoption reconstruct the same document windows
when reproducing stored vectors. A force rebuild checks the tokenizer and
embedding service before clearing existing embeddings.

### Index identity and rebuilding

The embedding fingerprint includes the model URI, resolved dimensions/context
mode/budget/revision, formatting, zero-overlap policy and a backend profile version.
Vector search rejects incompatible active profiles, including native-vs-PPLX
vectors, rather than comparing unrelated embedding spaces.

Use **one embedding profile per named index**. To compare profiles, use separate
indexes. To intentionally replace a profile in place, edit `models.embed`, then
rebuild **all collections** with:

```bash
~/Projects/qmd-pplx/bin/qmd --index pplx embed --force
```

Changing only the endpoint is safe only if it serves identical weights/tokenizer.
Changing dimensions, weights, tokenizer, formatting, chunking or grouping requires
a rebuild. Partial collection rebuilds must not be used to mix profiles.

## SDK

The same backend is used by SDK vector and hybrid/structured search:

```typescript
import { createStore } from "./dist/index.js";

const store = await createStore({
  dbPath: "/absolute/path/to/pplx.sqlite",
  config: {
    models: { embed: process.env.QMD_EMBED_MODEL! },
    collections: { notes: { path: "/absolute/path/to/notes", pattern: "**/*.md" } },
  },
});
try {
  await store.update();
  await store.embed();
  console.log(await store.searchVector("How do I reset my password?"));
  console.log(await store.search({
    queries: [{ type: "vec", query: "OAuth credential rotation" }],
    rerank: false,
  }));
} finally {
  await store.close();
}
```

Set the embedding model on the store/LLM. Supplying a different `model` only to
`store.embed()` is rejected for PPLX: that would otherwise mislabel the vectors.

## Verification

Fast assert-based checks using generated in-memory documents and an intercepted
transport (no server, tokenizer/model download, personal config or personal index):

```bash
bun scripts/check-pplx.ts
# Also verified on Node:
node --import tsx scripts/check-pplx.ts
```

With the real MLX server running and the environment above:

```bash
export QMD_PPLX_TEST_MODEL="$QMD_EMBED_MODEL"
bun scripts/check-pplx.ts --live
```

To also exercise **real GGUF query expansion and reranking** rather than only the
PPLX vector and pre-expanded structured paths, first ensure QMD's default
GGUF generation/reranking weights are cached, then:

```bash
GGML_METAL_NO_RESIDENCY=1 QMD_PPLX_CHECK_NATIVE=1 \
  bun scripts/check-pplx.ts --live
```

The live check builds and deletes temporary indexes, tests both document modes,
real multi-window indexing, sampled vector reproduction, oversized-input rejection,
1024/2048-dimensional queries, Unicode tokenization, and source/built CLI JSON
search plus `doctor`. Native-model checks may download their default GGUF weights
if not already cached. They do not select a GGUF embedding model.

The offline check covers validation, 64-item request batching, response ordering,
query-role routing, contextual retry preservation, failure recovery, profile
isolation, SDK calls, all three search paths, remote-ID routing/response checks,
and unchanged fingerprints for legacy URIs. The live check copies only the two
metadata files into a separate temporary client directory (including spaces,
Unicode and `#` in its path), then uses `remote_model` for all server requests,
including source/built CLI search and `doctor`.

Upstream regression commands:

```bash
bun run build
bun run lint
bun run test:types
CI=true GGML_METAL_NO_RESIDENCY=1 bun test --timeout 60000 \
  --preload ./src/test-preload.ts test/
CI=true GGML_METAL_NO_RESIDENCY=1 bun run test:node
```

On a machine whose global `init.templateDir` is empty, upstream release-script
tests need Git's stock templates. For that command only, prefix:

```bash
GIT_CONFIG_COUNT=1 \
GIT_CONFIG_KEY_0=init.templateDir \
GIT_CONFIG_VALUE_0=/opt/homebrew/share/git-core/templates \
CI=true GGML_METAL_NO_RESIDENCY=1 bun test --timeout 60000 \
  --preload ./src/test-preload.ts test/
```

This avoids changing global Git configuration or unrelated upstream tests.

### Results on the development machine

- Full upstream suites: **1529 passed / 89 skipped on Bun**;
  **1533 passed / 81 skipped on Node/Vitest**, with the command-scoped Git override.
- Tokenizer IDs match Transformers `AutoTokenizer` on Unicode, literal special
  tokens, NUL, whitespace, and contextual document-prefix/separator inputs.
- Initial three-topic retrieval checks: **3/3 expected top documents** for both
  independent and contextual 1024-dimensional indexing.
- A long document produced **36 chunks across 2 context windows**; its last-window
  vector reproduced the stored vector within `1e-6` per component.
- Exact query/independent/contextual token counts match the server's reported usage.
- Real GGUF expansion produced **4 subqueries**; hybrid reranking ranked the backup
  document first, and structured reranking ranked the gardening document first.
- Source and built CLI JSON vector search and contextual `doctor` passed. Live
  checks passed under both Bun and Node.
- Remote-ID separation passed live with a metadata-only client directory different
  from the server model path, including multi-window indexing, diagnostics and CLI.
  Legacy URI fingerprints remain unchanged.

These are correctness/smoke checks, **not evidence that PPLX beats QMD's default
model on your collection**. Evaluate your own queries before replacing a working
index. Initial tiny-corpus indexing took roughly 2.5s independent / 3.3s contextual;
that is not a throughput benchmark.

## Deliberate limits

- **Text only in QMD.** The MLX server can embed images/pages, but this fork does
  not add PDF rendering, image ingestion, visual-band storage or page-result mapping.
  Convert documents to Markdown for QMD, or use a separate visual retrieval index.
  Markdown image links do not cause their pixels to be embedded.
- **Regex chunking profile.** PPLX indexing rejects `--chunk-strategy auto` rather
  than allowing AST boundaries that its fingerprint/diagnostics cannot reproduce.
  Existing GGUF AST support is unchanged.
- **8192-token validated ceiling**, not the advertised 262144-token model maximum.
  Long documents are split into bounded context windows; no cross-window context.
- A lightweight tokenizer subprocess per call (~0.2s locally). A persistent
  tokenizer worker is the upgrade path if this dominates large-corpus indexing.
- An O(n) compatibility scan per vector lookup. Persisting an index-level profile
  is the upgrade path if profiling shows this matters on large indexes.
- The server-side inference lock bounds memory; concurrent client requests queue
  rather than performing parallel model inference.
- No boot/login service is installed. Start the server explicitly when needed.
