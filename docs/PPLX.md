# PPLX embeddings through MLX

## Client setup: two variables

Update this fork (`git pull --ff-only && bun run build` for a source checkout),
then set these on the **QMD client**:

```bash
export QMD_PPLX_URL="http://192.168.30.205/v1/embeddings"
export QMD_PPLX_MODEL="/Users/macos/models/pplx-embed-v2-context-9b-MLX-8bit"
```

That is all the embedding configuration required. `QMD_PPLX_MODEL` is the exact
model ID listed by the host's `GET /v1/models`, not a client-local filesystem path.
The client needs **no revision string, model weights, tokenizer files, Python,
or MLX installation**. Do not copy any model files to the client.

Use the fork's executable (shown as `qmd` below), not upstream `@tobilu/qmd`:

```bash
# Run these yourself when ready to index your own collection.
qmd --index pplx collection add ~/notes --name notes
qmd --index pplx update
qmd --index pplx embed
qmd --index pplx vsearch "How do I configure authentication?"
qmd --index pplx query "How do I configure authentication?"
qmd --index pplx doctor
```

For the local checkout, the executable is `~/Projects/qmd-pplx/bin/qmd`.
A fresh named index leaves an existing working index untouched. If replacing an
existing embedding profile, rebuild **all collections** once with
`qmd --index pplx embed --force`; incompatible vectors are never silently mixed.

**Environment precedence:** `QMD_PPLX_MODEL` plus `QMD_PPLX_URL` explicitly select
remote PPLX embeddings, overriding saved `models.embed` and `QMD_EMBED_MODEL`.
There is no need to edit an index YAML or construct a `pplx:` URI. Unset
`QMD_PPLX_MODEL` to return to normal QMD model selection.

Only embedding inference and tokenization run on the server. QMD's database,
BM25, query expansion and reranking remain on the client. `vsearch` needs no
client-side GGUF models; `query` uses QMD's existing GGUF expansion/reranking
models and their normal platform requirements. `pull` skips the remote embedding
model and manages only GGUF models.

## Server prerequisite (already installed on this machine)

The host must run the patched **`mlx_vlm.server`**, with the PPLX contextual
adapter and QMD remote-tokenizer support. An unmodified OpenAI-compatible
embedding endpoint is not sufficient for contextual encoding and exact budgeting.

The existing conversion, base server patch and model adapter are documented at
`~/Projects/pplx-embed-mlx/README.md`. To add/update the QMD server extension,
stop the server, then run this **on the host using the server's Python**:

```bash
cd ~/Projects/qmd-pplx
~/.local/share/uv/tools/mlx-vlm/bin/python scripts/install-pplx-server.py
```

The installer is repeatable, checks compatibility before editing, and preserves
`embeddings.py.qmd-backup`. It installs the shared tokenizer helper into the
server environment. Re-run it after upgrading mlx-vlm or this extension.
The base PPLX model adapter/basic embedding patch must already be installed.

Start the host's service, using an appropriate trusted interface and port:

```bash
~/.local/bin/mlx_vlm.server \
  --embedding-model "$HOME/models/pplx-embed-v2-context-9b-MLX-8bit" \
  --host 192.168.30.205 --port 8080
```

The client's URL must match the reachable service URL. The example client URL
uses HTTP's default port 80: a reverse proxy must forward that address to the
server, or use `http://192.168.30.205:8080/v1/embeddings` for direct access.
No proxy or firewall configuration is installed automatically.

The service is **unauthenticated**. Restrict LAN access to trusted clients; do
not expose it to the public Internet. Prefer an SSH tunnel on untrusted networks:

```bash
# Host server stays bound to 127.0.0.1:8080.
# Run on the client in a separate terminal:
ssh -N -L 18080:127.0.0.1:8080 macos@192.168.30.205

# In the QMD client's terminal:
export QMD_PPLX_URL="http://127.0.0.1:18080/v1/embeddings"
```

`QMD_PPLX_MODEL` remains the host's model ID, unchanged by the tunnel.

## Automatic identity and exact tokenization

QMD handles the details internally:

- Query encoding uses `task: "query"`; document encoding uses `task: "document"`.
  Learned model prefixes are added exactly once by the host.
- Default output is **1024-dimensional, normalized** embeddings.
- Markdown-aware chunks have a **900-token ceiling and zero overlap**.
- Consecutive chunks from one document share bounded contextual windows; unrelated
  documents never share context.
- The host's actual tokenizer counts prefixes, separators, titles and text.
  It supplies the validated context ceiling, at most **8192 tokens**.
- Tokenization/metadata use the **same configured embedding URL**, so reverse
  proxies do not need additional routes.
- On the first metadata request after loading a model, the host hashes its actual
  weights, tokenizer, config, PPLX adapter and tokenizer-service code. This may take several seconds;
  the result is reused for that loaded model.
- QMD automatically caches only the small identity/limit metadata under
  `$XDG_CACHE_HOME/qmd/pplx/` (normally `~/.cache/qmd/pplx/`). No tokenizer is
  downloaded and no client-side subprocess is run.
- Each client process refreshes metadata before embedding/search. Token operations
  and embedding requests include the expected identity; a changed host model is
  rejected instead of silently contaminating an index.

Restart QMD after changing/reloading the host model, then rebuild affected
embeddings. The client never needs to know or supply a revision. Offline status
uses the last cached identity; it is not a live server-health check.

Profiles are conservatively scoped to the selected endpoint/model/settings.
Changing the endpoint/model selection is treated as a different index profile;
use a separate named index or rebuild. Matching weights at two endpoints are not
an automatic index-migration mechanism.

Window writes are transactional. Failed contextual windows are retried together,
never as isolated chunks. Incomplete documents remain pending, and a force
rebuild checks the service before deleting existing vectors. HTTP failures,
invalid metadata/mappings/dimensions/nonfinite vectors, stale identities and
oversized inputs fail explicitly. There is no fallback model or silent truncation.

`doctor` reproduces the same context windows when checking saved vectors and
reports a **server-side tokenizer**, not missing local files.

## SDK

Set the same two environment variables before creating a store. No embedding
configuration object is required, even if an old config specifies a GGUF model:

```typescript
import { createStore } from "./dist/index.js";

const store = await createStore({ dbPath: "/absolute/path/to/pplx.sqlite" });
try {
  console.log(await store.searchVector("How do I reset my password?"));
} finally {
  await store.close();
}
```

The store must already contain indexed/embedded documents; normal SDK collection,
update and embed methods are unchanged.

## Verification

Fast assert-based checks; no service, weights, Python or test framework required:

```bash
bun run test:pplx
node --import tsx scripts/check-pplx.ts
```

With the server running, test the actual two-variable setup against temporary
indexes (nothing touches personal collections):

```bash
export QMD_PPLX_URL="http://127.0.0.1:8080/v1/embeddings"
export QMD_PPLX_MODEL="/Users/macos/models/pplx-embed-v2-context-9b-MLX-8bit"
GGML_METAL_NO_RESIDENCY=1 bun scripts/check-pplx.ts --remote-live
```

This deliberately points `QMD_PPLX_PYTHON` at a nonexistent executable and uses
no local model/tokenizer files. It checks SDK indexing, expected top documents,
Unicode tokenization, source/built CLI JSON vector search, `doctor`, invalid
requests, and rejection of an incorrect model fingerprint. Saved GGUF model
settings are deliberately present to verify environment precedence.

Upstream regression checks:

```bash
bun run build
bun run lint
bun run test:types
CI=true GGML_METAL_NO_RESIDENCY=1 bun test --timeout 60000 \
  --preload ./src/test-preload.ts test/
```

On this machine, the unrelated upstream release tests need stock Git templates
because global `init.templateDir` is empty. Prefix only that command with
`GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=init.templateDir
GIT_CONFIG_VALUE_0=/opt/homebrew/share/git-core/templates`; no global Git settings
need changing. Full Bun regression: **1529 passed / 89 skipped**. Targeted Node
embedding/store/SDK/doctor/structured-search regression: **540 passed / 50 skipped**.

These are correctness checks, not evidence that PPLX beats QMD's default
embedding model on your collection. Evaluate representative queries yourself.

## Legacy / advanced local-tokenizer mode

Existing `QMD_EMBED_MODEL=pplx:/local/tokenizer/path?...` URIs remain supported
when `QMD_PPLX_MODEL` is **unset**. Their fingerprints are unchanged. Only this
legacy mode needs local `config.json`, `tokenizer.json` and a Python executable
with `tokenizers` (`QMD_PPLX_PYTHON`, default `python3`).

Its optional URI parameters remain `revision` (required in legacy mode),
`remote_model` (defaults to the local path), `dimensions` (`1024`/`2048`),
`mode` (`contextual`/`independent`) and `max_tokens` (`1024`–`8192`). Unknown or
duplicate keys are rejected. Percent-encode reserved characters in paths and
query values; use `URLSearchParams` when constructing arbitrary IDs.

Do not configure this URI for the normal remote setup: **URL + MODEL replaces it**.

## Deliberate limits

- Text only in QMD: no PDF rendering, image ingestion, visual-band storage or
  page-result mapping. Markdown image links do not embed pixels.
- PPLX indexing uses the reproducible regex profile; `--chunk-strategy auto` is
  rejected. GGUF AST support remains unchanged.
- At most 8192 tokens per shared window, not the advertised 262144-token model
  maximum. No cross-window context.
- The encoder lock serializes tokenization and inference to bound memory and
  protect tokenizer state. Remote tokenization adds LAN round trips.
- Compatibility checks currently scan active vectors, O(n) per vector lookup.
  Persisting an index-level profile is the upgrade path if profiling warrants it.
- No boot/login service or public network exposure is installed automatically.
