// Run: bun scripts/check-pplx.ts [--live]
// No framework, no production index. --live requires QMD_PPLX_TEST_MODEL and a running MLX server.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtempSync, rmSync, mkdirSync, writeFileSync } from "node:fs";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PplxEmbeddings, parsePplxModel } from "../src/pplx.js";
import { LlamaCpp, formatDocForEmbedding, formatQueryForEmbedding, setDefaultLlamaCpp,
  withLLMSessionForLlm } from "../src/llm.js";
import { createStore, generateEmbeddings, chunkDocumentByTokens, getEmbeddingFingerprint,
  hashContent, hybridQuery, structuredSearch, maybeAdoptLegacyEmbeddingFingerprint, embedDocumentPassage } from "../src/store.js";
import { checkEmbeddingVectorSamples } from "../src/cli/qmd.js";
import { createStore as createSdkStore } from "../src/index.js";
import { storedEmbedding } from "../src/vec-layout.js";

const fakeModel = "pplx:/not-loaded/model?revision=check-q8&dimensions=1024&mode=contextual";
const originalFetch = globalThis.fetch;
const originalTokenizer = PplxEmbeddings.prototype.tokenizer;
const originalUrl = process.env.QMD_PPLX_URL;
const requests: any[] = [];
let fault: string | undefined;
let transientFailures = 0;

function vector(text: string, context = "", dimensions = 1024) {
  const category = /auth|password|oauth/i.test(text) ? 0 : /backup|restore/i.test(text) ? 1 : 2;
  const values = Array(dimensions).fill(0);
  values[category] = 1;
  values[3] = createHash("sha256").update(text + context).digest()[0]! / 1000;
  const norm = Math.hypot(...values);
  return values.map(value => value / norm);
}

async function addDocs(store: ReturnType<typeof createStore>, long = true) {
  const docs = [
    ["auth.md", "Authentication", "Reset passwords through the OAuth identity provider. Rotate client credentials and invalidate old access tokens. "],
    ["backup.md", "Backups", "Restore encrypted database backups from snapshots. Verify the backup checksum before restoring the archive. "],
    ["garden.md", "Gardening", "Plant tomato seedlings in fertile soil. Water the vegetable garden and protect the roots from frost. "],
  ] as const;
  return Promise.all(docs.map(async ([path, title, text]) => {
    const body = `# ${title}\n\n` + text.repeat(long ? 45 : 2);
    const hash = await hashContent(body);
    store.insertContent(hash, body, "2026-01-01");
    store.insertDocument("check", path, title, hash, "2026-01-01", "2026-01-01");
    return { path, title, body, hash };
  }));
}

async function offline() {
  process.env.QMD_PPLX_URL = "http://127.0.0.1:8080/v1/embeddings";
  for (const invalid of ["pplx:relative?revision=x", "pplx://host/path?revision=x", "pplx:/path",
    "pplx:/path?revision=x&dimensions=768", "pplx:/path?revision=x&mode=oops",
    "pplx:/path?revision=x&max_tokens=9000", "pplx:/path?revision=x&dimensions=1024&dimensions=2048",
    "pplx:/path?revision=x&surprise=1"]) assert.throws(() => parsePplxModel(invalid));
  assert.equal(formatQueryForEmbedding("plain", fakeModel), "plain");
  assert.equal(formatDocForEmbedding("body", "Title", fakeModel), "Title\nbody");
  assert.match(formatQueryForEmbedding("plain", "gemma.gguf"), /^task:/);
  for (const suffix of ["&max_tokens=4096", "&mode=independent", "&dimensions=2048", "&revision=other"]) {
    const key = suffix.split("=")[0]!.slice(1);
    const uri = new URL(fakeModel);
    uri.searchParams.set(key, suffix.split("=")[1]!);
    assert.notEqual(getEmbeddingFingerprint(fakeModel), getEmbeddingFingerprint(uri.href));
  }

  // Transport/indexing checks need no Python, model weights, native llama, or network.
  PplxEmbeddings.prototype.tokenizer = async function<T>(request): Promise<T> {
    if (request.op === "encode") return Array.from(request.text, char => char.codePointAt(0)!) as T;
    if (request.op === "decode") return String.fromCodePoint(...request.tokens) as T;
    if (request.op === "windows") return Array.from({ length: Math.ceil(request.texts.length / 2) },
      (_, i) => [i * 2, Math.min(i * 2 + 2, request.texts.length)]) as T;
    return (request.contextual ? [request.texts.join("").length + 10]
      : request.texts.map(text => text.length + 10)) as T;
  };
  globalThis.fetch = (async (_url, options) => {
    const body = JSON.parse(options!.body as string);
    requests.push(body);
    options?.signal?.throwIfAborted();
    if (fault === "http" || transientFailures-- > 0) return new Response("temporarily unavailable", { status: 503 });
    const contextual = !Array.isArray(body.input);
    const texts: string[] = contextual ? body.input.chunks : body.input;
    const data = texts.map((text, index) => ({ index, input_index: contextual ? 0 : index,
      chunk_index: contextual ? index : 0,
      embedding: vector(text, contextual ? texts.join("\n") : "", body.dimensions) }));
    if (fault === "count") data.pop();
    if (fault === "mapping") data[0]!.chunk_index = 123;
    if (fault === "dimensions") data[0]!.embedding.pop();
    if (fault === "norm") data[0]!.embedding.fill(0);
    if (fault === "duplicate") data.push(data[0]!);
    if (fault === "nan") data[0]!.embedding[0] = NaN;
    return Response.json({ model: fault === "model" ? "wrong-model" : body.model, data: data.reverse() });
  }) as typeof fetch;

  const llm = new LlamaCpp({ embedModel: fakeModel });
  setDefaultLlamaCpp(llm);
  try {
    assert.deepEqual(await llm.tokenize("雪😀"), [38634, 128512]);
    assert.equal(await llm.detokenize(await llm.tokenize("雪😀")), "雪😀");
    assert.deepEqual(await llm.embedBatch([]), []);
    const result = await llm.embedBatch(["password", "restore", "tomato"], { isQuery: true });
    assert.equal(result.length, 3);
    assert.equal(result[0]!.embedding.indexOf(Math.max(...result[0]!.embedding)), 0);
    assert.equal(result[1]!.embedding.indexOf(Math.max(...result[1]!.embedding)), 1);
    assert.equal(requests.at(-1).task, "query");
    const before = requests.length;
    await llm.embedBatch(Array(65).fill("password"), { isQuery: true });
    assert.equal(requests.length - before, 2);
    assert.equal(requests.at(-2).input.length, 64);
    assert.equal(requests.at(-1).input.length, 1);
    await assert.rejects(llm.embedBatch(["q"], { isQuery: true, contextual: true }), /cannot share/);
    await assert.rejects(llm.embed("x".repeat(8193)), /exceeds/);
    await assert.rejects(llm.embed(""), /nonempty/);
    await assert.rejects(llm.embed("q", { model: "wrong" }), /differs/);
    await assert.rejects(llm.embed("q", { signal: AbortSignal.abort() }));
    const abort = new AbortController();
    await assert.rejects(withLLMSessionForLlm(llm, async session => {
      abort.abort();
      return session.embed("q");
    }, { signal: abort.signal }));
    for (fault of ["count", "mapping", "dimensions", "norm", "duplicate", "nan", "model", "http"]) {
      await assert.rejects(llm.embedBatch(["password", "restore"], { contextual: true }));
    }
    fault = undefined;

    const store = createStore(":memory:");
    store.llm = llm;
    try {
      const docs = await addDocs(store);
      const start = requests.length;
      transientFailures = 1;
      const embedded = await generateEmbeddings(store);
      assert.equal(embedded.errors, 0);
      assert.ok(embedded.chunksEmbedded > 6);
      assert.deepEqual(requests[start].input, requests[start + 1].input, "retry must keep the whole original context");
      assert.ok(requests.slice(start).every(body => body.input.chunks && body.task === "document"));
      for (const body of requests.slice(start)) {
        const titles = new Set(body.input.chunks.map((text: string) => text.split("\n")[0]));
        assert.equal(titles.size, 1, "never cross documents inside a context window");
      }
      for (const doc of docs) {
        const chunks = await chunkDocumentByTokens(doc.body);
        for (let i = 0; i < chunks.length; i++) {
          assert.equal(chunks[i]!.text, doc.body.slice(chunks[i]!.pos, chunks[i]!.pos + chunks[i]!.text.length));
          if (i) assert.ok(chunks[i]!.pos >= chunks[i - 1]!.pos + chunks[i - 1]!.text.length, "no overlap");
          assert.ok(storedEmbedding(store.db, doc.hash, i));
        }
      }
      assert.equal(store.getHashesNeedingEmbedding(), 0);
      assert.equal((await generateEmbeddings(store)).chunksEmbedded, 0);
      assert.equal((await checkEmbeddingVectorSamples(store.db, fakeModel, getEmbeddingFingerprint(fakeModel), 3)).ok, true);
      // Legacy adoption must also reproduce the contextual window.
      store.db.prepare("UPDATE content_vectors SET embed_fingerprint = ''").run();
      assert.ok((await maybeAdoptLegacyEmbeddingFingerprint(store, fakeModel)).adopted > 0);
      assert.equal((await store.searchVec("password reset", fakeModel, 1))[0]!.displayPath, "check/auth.md");
      llm.expandQuery = async () => [{ type: "vec", text: "OAuth credentials" }];
      const queryStart = requests.length;
      assert.ok((await hybridQuery(store, "password reset", { skipRerank: true })).length > 0);
      assert.ok((await structuredSearch(store, [{ type: "vec", query: "password reset" },
        { type: "hyde", query: "OAuth identity provider" }], { skipRerank: true })).length > 0);
      assert.ok(requests.slice(queryStart).every(body => body.task === "query" && Array.isArray(body.input)));
      const independent = fakeModel.replace("mode=contextual", "mode=independent");
      await assert.rejects(store.searchVec("password", independent), /Incompatible PPLX/);
      await assert.rejects(store.searchVec("password", "gemma.gguf"), /Incompatible PPLX/);
      await assert.rejects(generateEmbeddings(store, { model: independent, force: true }), /Select the PPLX model/);
      await assert.rejects(generateEmbeddings(store, { chunkStrategy: "auto", force: true }), /regex chunking profile/);
      const independentLlm = new LlamaCpp({ embedModel: independent });
      try { await assert.rejects(independentLlm.embedBatch(["one", "two"], { contextual: true }), /Independent PPLX profile/); }
      finally { await independentLlm.dispose(); }
      assert.equal(store.getHashesNeedingEmbedding(), 0, "bad override must not clear good embeddings");
      fault = "http";
      await assert.rejects(generateEmbeddings(store, { force: true }), /PPLX HTTP 503/);
      fault = undefined;
      assert.equal(store.getHashesNeedingEmbedding(), 0, "failed force preflight must preserve existing vectors");
      const replacement = independent.replace("dimensions=1024", "dimensions=2048");
      const replacementLlm = new LlamaCpp({ embedModel: replacement });
      store.llm = replacementLlm;
      setDefaultLlamaCpp(replacementLlm);
      try {
        assert.equal((await generateEmbeddings(store, { force: true })).errors, 0);
        assert.equal(store.getHashesNeedingEmbedding(), 0);
        assert.equal((await store.searchVec("restore backup", replacement, 1))[0]!.displayPath, "check/backup.md");
        assert.equal((await checkEmbeddingVectorSamples(store.db, replacement, getEmbeddingFingerprint(replacement), 3)).ok, true);
        await assert.rejects(store.searchVec("password", fakeModel), /Incompatible PPLX/);
      } finally {
        await replacementLlm.dispose();
        store.llm = llm;
        setDefaultLlamaCpp(llm);
      }
    } finally { store.close(); }

    const broken = createStore(":memory:");
    broken.llm = llm;
    try {
      await addDocs(broken);
      fault = "http";
      const failed = await generateEmbeddings(broken);
      assert.ok(failed.errors > 0);
      assert.equal(broken.db.prepare("SELECT count(*) AS n FROM content_vectors").get<{ n: number }>()!.n, 0);
      assert.equal(broken.getHashesNeedingEmbedding(), 3);
      fault = undefined;
      assert.equal((await generateEmbeddings(broken)).errors, 0, "next run must recover cleanly");
    } finally { broken.close(); }

    const sdk = await createSdkStore({ dbPath: ":memory:", config: { collections: {}, models: { embed: fakeModel } } });
    try {
      await addDocs(sdk.internal, false);
      await sdk.embed();
      assert.ok((await sdk.searchVector("password", { limit: 1 }))[0]!.displayPath.endsWith("auth.md"));
      assert.ok((await sdk.search({ queries: [{ type: "vec", query: "password" }], rerank: false })).length > 0);
    } finally { await sdk.close(); }
  } finally {
    await llm.dispose();
    setDefaultLlamaCpp(null);
  }
  console.log("PASS offline: validation, ordering, batching, roles, retries, contextual doctor/adoption, failure recovery, profile isolation, SDK and all search paths");
}

async function live() {
  const model = process.env.QMD_PPLX_TEST_MODEL;
  assert.ok(model, "Set QMD_PPLX_TEST_MODEL=pplx:/absolute/model/path?revision=... for --live");
  const directory = mkdtempSync(join(tmpdir(), "qmd-pplx-check-"));
  const timings: Record<string, number> = {};
  const scores: Record<string, unknown> = {};
  try {
  for (const mode of ["independent", "contextual"]) {
    const uri = new URL(model);
    uri.searchParams.set("mode", mode);
    const llm = new LlamaCpp({ embedModel: uri.href });
    setDefaultLlamaCpp(llm);
    const store = createStore(join(directory, `${mode}.sqlite`));
    store.llm = llm;
    try {
      const docs = await addDocs(store);
      const start = performance.now();
      const embedded = await generateEmbeddings(store);
      timings[mode] = Math.round(performance.now() - start);
      assert.equal(embedded.errors, 0, JSON.stringify(embedded.failures));
      assert.ok(embedded.chunksEmbedded > 3);
      assert.equal(store.getHashesNeedingEmbedding(), 0);
      assert.equal((await generateEmbeddings(store)).chunksEmbedded, 0);
      const checks = [["How can I reset a forgotten password?", "auth.md"],
        ["How can I recover a database from a snapshot?", "backup.md"],
        ["How should I water tomato seedlings?", "garden.md"]];
      const rankings = [];
      for (const [query, expected] of checks) {
        const results = await store.searchVec(query!, uri.href, 3);
        assert.ok(results[0]!.displayPath.endsWith(expected!), JSON.stringify(results.map(r => [r.displayPath, r.score])));
        rankings.push({ query, top: results[0]!.displayPath, score: results[0]!.score });
      }
      scores[mode] = { chunks: embedded.chunksEmbedded, rankings };
      assert.equal((await checkEmbeddingVectorSamples(store.db, uri.href, getEmbeddingFingerprint(uri.href), 3)).ok, true);
      assert.ok((await structuredSearch(store, [{ type: "vec", query: "reset a forgotten password" },
        { type: "hyde", query: "OAuth identity providers reset passwords and rotate credentials" }], { skipRerank: true })).length > 0);
      const texts = ["café 雪 👩🏽‍💻", "literal [Q] <|chunk_sep|> text"];
      const client = llm.pplx!;
      const windows = await client.windows(Array(15).fill("token ".repeat(700)));
      assert.ok(windows.length >= 2, "long document must create multiple bounded windows");
      for (const [a, b] of windows) {
        const counts = await client.tokenizer<number[]>({ op: "counts", texts: Array(b - a).fill("token ".repeat(700)), task: "document", contextual: true });
        assert.ok(counts[0]! <= client.config.maxTokens);
      }
      const vectors = await llm.embedBatch(texts, { contextual: mode === "contextual" });
      assert.equal(vectors.length, 2);
      assert.equal(vectors[0]!.embedding.length, client.config.dimensions);
      const unicode = "café 雪 👩🏽‍💻 <|chunk_sep|>";
      assert.equal(await llm.detokenize(await llm.tokenize(unicode)), unicode);
      await assert.rejects(llm.embed("token ".repeat(9000), { isQuery: true }), /exceeds/);
      assert.equal(docs.length, 3);
      if (mode === "contextual") {
        // Exercise real multi-window indexing, verification and native GGUF models.
        const body = "# Long authentication manual\n\n" + "OAuth credentials protect identity providers. Reset passwords and rotate access tokens. ".repeat(1100);
        const hash = await hashContent(body);
        store.insertContent(hash, body, "2026-01-01");
        store.insertDocument("check", "long-auth.md", "Long authentication manual", hash, "2026-01-01", "2026-01-01");
        const chunks = await chunkDocumentByTokens(body);
        const texts = chunks.map(chunk => formatDocForEmbedding(chunk.text, "Long authentication manual", uri.href));
        const groups = await client.windows(texts);
        assert.ok(groups.length > 1);
        const longResult = await generateEmbeddings(store);
        assert.equal(longResult.errors, 0);
        assert.equal(longResult.chunksEmbedded, chunks.length);
        const last = chunks.at(-1)!;
        const reproduced = await withLLMSessionForLlm(llm, session =>
          embedDocumentPassage(llm, session, chunks, "Long authentication manual", uri.href, last.pos));
        const saved = new Float32Array(Uint8Array.from(storedEmbedding(store.db, hash, chunks.length - 1)!).buffer);
        assert.ok(reproduced!.embedding.every((value, i) => Math.abs(value - saved[i]!) < 0.000001));
        assert.equal((await checkEmbeddingVectorSamples(store.db, uri.href, getEmbeddingFingerprint(uri.href), 3)).ok, true);
        scores.multi_window = { chunks: chunks.length, windows: groups.length, last_window_reproduces: true };
        for (const [task, contextual] of [["query", false], ["document", false], ["document", true]] as const) {
          const inputs = ["café 雪 👩🏽‍💻", "literal [Q] <|chunk_sep|> <|endoftext|>"];
          const counts = await client.tokenizer<number[]>({ op: "counts", texts: inputs, task, contextual });
          const response = await originalFetch(process.env.QMD_PPLX_URL ?? "http://127.0.0.1:8080/v1/embeddings", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model: client.config.path, input: contextual ? { chunks: inputs } : inputs,
              task, dimensions: client.config.dimensions, normalize_embeddings: true }),
          });
          assert.ok(response.ok);
          const payload = await response.json() as { usage: { prompt_tokens: number } };
          assert.equal(payload.usage.prompt_tokens, counts.reduce((a, b) => a + b, 0), "exact server/tokenizer budget parity");
        }
        scores.token_budget_parity = "query, independent document, contextual document PASS";
        if (process.env.QMD_PPLX_CHECK_NATIVE === "1") {
          const expanded = await llm.expandQuery("recover a database from an encrypted snapshot");
          assert.ok(expanded.length > 0, "native GGUF expansion should produce subqueries");
          const ranked = await hybridQuery(store, "recover a database from an encrypted snapshot");
          assert.ok(ranked[0]!.displayPath.endsWith("backup.md"));
          const structured = await structuredSearch(store, [{ type: "vec", query: "water tomato seedlings" }]);
          assert.ok(structured[0]!.displayPath.endsWith("garden.md"));
          scores.native = { expansion_count: expanded.length, hybrid_top: ranked[0]!.displayPath,
            structured_reranked_top: structured[0]!.displayPath };
        }
        const configDir = join(directory, "config");
        mkdirSync(configDir);
        writeFileSync(join(configDir, "check.yml"), JSON.stringify({ collections: {}, models: { embed: uri.href } }));
        const run = promisify(execFile);
        const env = { ...process.env, INDEX_PATH: join(directory, `${mode}.sqlite`), QMD_CONFIG_DIR: configDir,
          GGML_METAL_NO_RESIDENCY: "1", QMD_EMBED_MODEL: uri.href };
        // Read-only CLI checks use the temporary SDK-built index, not any personal collection.
        for (const sourceMode of ["1", "0"]) {
          const cli = await run("./bin/qmd", ["--index", "check", "vsearch", "recover a database from a snapshot", "--json"],
            { env: { ...env, QMD_SOURCE_MODE: sourceMode }, timeout: 120_000 });
          const rows = JSON.parse(cli.stdout);
          assert.ok(new URL(rows[0].file).pathname.endsWith("backup.md"), cli.stdout);
        }
        const doctor = await run("./bin/qmd", ["--index", "check", "doctor"], { env, timeout: 120_000 });
        assert.match(doctor.stdout, /sampled chunks reproduce stored vectors/);
        assert.doesNotMatch(doctor.stdout, /PPLX.*not valid GGUF|stored vector distance/);
        scores.cli = "source + built vsearch JSON, contextual doctor PASS";
      }
    } finally {
      store.close();
      await llm.dispose();
      setDefaultLlamaCpp(null);
    }
  }
  // 2048d is a separately indexed space; verify its query transport too.
  const wide = new URL(model);
  wide.searchParams.set("dimensions", "2048");
  const llm = new LlamaCpp({ embedModel: wide.href });
  try { assert.equal((await llm.embed("password reset", { isQuery: true }))!.embedding.length, 2048); }
  finally { await llm.dispose(); }
  console.log(JSON.stringify({ live: "PASS", timings_ms: timings, retrieval: scores }, null, 2));
  } finally { rmSync(directory, { recursive: true, force: true }); }
}

try { await offline(); }
finally {
  globalThis.fetch = originalFetch;
  PplxEmbeddings.prototype.tokenizer = originalTokenizer;
  if (originalUrl === undefined) delete process.env.QMD_PPLX_URL;
  else process.env.QMD_PPLX_URL = originalUrl;
}
if (process.argv.includes("--live")) await live();
