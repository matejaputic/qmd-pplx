import { execFile } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

type RemoteInfo = { fingerprint: string; maxTokens: number };
const remoteInfo = new Map<string, RemoteInfo>();
const preparing = new Map<string, Promise<void>>();

function endpoint(value: string): string {
  const url = new URL(value);
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.hash) {
    throw new Error("QMD_PPLX_URL must be an HTTP(S) embeddings endpoint without credentials or fragment");
  }
  return url.href;
}

/** Explicit environment selection overrides a saved GGUF/legacy model setting. */
export function pplxModelFromEnv(): string | undefined {
  const model = process.env.QMD_PPLX_MODEL;
  if (model === undefined) return undefined;
  if (!model.trim() || /\p{Cc}/u.test(model)) throw new Error("QMD_PPLX_MODEL must be a nonempty model ID without control characters");
  if (!process.env.QMD_PPLX_URL) throw new Error("Set QMD_PPLX_URL together with QMD_PPLX_MODEL");
  const uri = new URL("pplx:remote");
  uri.searchParams.set("url", endpoint(process.env.QMD_PPLX_URL));
  uri.searchParams.set("model", model);
  return uri.href;
}

function infoPath(model: string): string {
  return join(process.env.XDG_CACHE_HOME || join(homedir(), ".cache"), "qmd", "pplx",
    createHash("sha256").update(model).digest("hex") + ".json");
}

function checkedInfo(value: unknown): RemoteInfo {
  const info = value as Partial<RemoteInfo> | null;
  if (!info || typeof info.fingerprint !== "string" || !/^[a-f0-9]{64}$/.test(info.fingerprint)
    || !Number.isInteger(info.maxTokens) || info.maxTokens! < 1024 || info.maxTokens! > 8192) {
    throw new Error("Invalid PPLX server metadata; update the server's QMD tokenizer support");
  }
  return info as RemoteInfo;
}

export async function preparePplxModel(model: string): Promise<void> {
  if (!model.startsWith("pplx:remote?")) return;
  let promise = preparing.get(model);
  if (!promise) {
    promise = (async () => {
      const config = parsePplxModel(model);
      const response = await fetch(config.endpoint!, { method: "POST", redirect: "error",
        headers: { "Content-Type": "application/json" }, signal: AbortSignal.timeout(180_000),
        body: JSON.stringify({ model: config.remoteModel, input: [], tokenizer: { op: "info" } }) });
      if (!response.ok) throw new Error(`PPLX metadata HTTP ${response.status}: ${(await response.text()).slice(0, 500)}. Update the server's QMD tokenizer support.`);
      const body = await response.json() as { model?: unknown; result?: unknown };
      if (body.model !== config.remoteModel) throw new Error("PPLX metadata model mismatch");
      const info = checkedInfo(body.result);
      const path = infoPath(model);
      mkdirSync(dirname(path), { recursive: true });
      const temporary = `${path}.${randomUUID()}.tmp`;
      writeFileSync(temporary, JSON.stringify(info), { mode: 0o600 });
      renameSync(temporary, path);
      remoteInfo.set(model, info);
    })();
    preparing.set(model, promise);
    promise.catch(() => preparing.delete(model));
  }
  await promise;
}

type TokenizerRequest =
  | { op: "encode"; text: string }
  | { op: "decode"; tokens: readonly number[] }
  | { op: "windows"; texts: string[]; max_tokens: number }
  | { op: "counts"; texts: string[]; task: "query" | "document"; contextual: boolean };

export const isPplxModel = (model: string): boolean => model.startsWith("pplx:");

/** All vector-affecting settings belong in the model URI, not environment variables. */
export function parsePplxModel(model: string) {
  const uri = new URL(model);
  if (uri.pathname === "remote") {
    if (uri.protocol !== "pplx:" || uri.host || uri.hash || [...uri.searchParams.keys()].sort().join(",") !== "model,url") {
      throw new Error("Invalid remote PPLX model selection");
    }
    const remoteModel = uri.searchParams.get("model")!;
    if (!remoteModel.trim() || /\p{Cc}/u.test(remoteModel)) throw new Error("Invalid remote PPLX model ID");
    const url = endpoint(uri.searchParams.get("url")!);
    let info = remoteInfo.get(model);
    if (!info && existsSync(infoPath(model))) info = checkedInfo(JSON.parse(readFileSync(infoPath(model), "utf8")));
    return { path: "", dimensions: 1024, maxTokens: info?.maxTokens ?? 8192, mode: "contextual",
      revision: info?.fingerprint ?? "unresolved", remoteModel, endpoint: url, remote: true };
  }
  const allowed = ["dimensions", "mode", "max_tokens", "revision", "remote_model"];
  if (uri.protocol !== "pplx:" || uri.host || uri.hash || !uri.pathname.startsWith("/")) {
    throw new Error("PPLX model must be pplx:/absolute/model/path?revision=<weights-revision>");
  }
  for (const key of uri.searchParams.keys()) {
    if (!allowed.includes(key) || uri.searchParams.getAll(key).length !== 1) {
      throw new Error(`Unknown or duplicate PPLX setting: ${key}`);
    }
  }
  const dimensions = Number(uri.searchParams.get("dimensions") ?? 1024);
  const maxTokens = Number(uri.searchParams.get("max_tokens") ?? 8192);
  const mode = uri.searchParams.get("mode") ?? "contextual";
  const revision = uri.searchParams.get("revision");
  if (![1024, 2048].includes(dimensions)) throw new Error("PPLX dimensions must be 1024 or 2048");
  if (!Number.isInteger(maxTokens) || maxTokens < 1024 || maxTokens > 8192) {
    throw new Error("PPLX max_tokens must be an integer between 1024 and the validated 8192-token ceiling");
  }
  if (!["contextual", "independent"].includes(mode)) throw new Error("PPLX mode must be contextual or independent");
  if (!revision?.trim()) throw new Error("PPLX revision is required; change it whenever weights/tokenizer change");
  const remoteModel = uri.searchParams.get("remote_model");
  if (remoteModel !== null && (!remoteModel.trim() || /\p{Cc}/u.test(remoteModel))) {
    throw new Error("PPLX remote_model must be nonempty and contain no control characters");
  }
  // Omit absent remote_model so existing index fingerprints remain unchanged.
  return { path: decodeURIComponent(uri.pathname), dimensions, maxTokens, mode, revision,
    ...(remoteModel === null ? {} : { remoteModel }) };
}

/** Model-specific transport; generation and reranking stay in node-llama-cpp. */
export class PplxEmbeddings {
  readonly config;
  private readonly endpoint: string;

  constructor(readonly model: string) {
    this.config = parsePplxModel(model);
    this.endpoint = endpoint(this.config.endpoint ?? process.env.QMD_PPLX_URL ?? "http://127.0.0.1:8080/v1/embeddings");
  }

  async prepare(): Promise<void> {
    await preparePplxModel(this.model);
    if (this.config.remote) Object.assign(this.config, parsePplxModel(this.model));
  }

  async tokenizer<T>(request: TokenizerRequest, signal?: AbortSignal): Promise<T> {
    if (this.config.remote) {
      await this.prepare();
      const timeout = AbortSignal.timeout(60_000);
      const response = await fetch(this.endpoint, { method: "POST", redirect: "error",
        headers: { "Content-Type": "application/json" },
        signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
        body: JSON.stringify({ model: this.config.remoteModel, input: [], tokenizer: request,
          expected_fingerprint: this.config.revision }) });
      if (!response.ok) throw new Error(`PPLX tokenizer HTTP ${response.status}: ${(await response.text()).slice(0, 500)}`);
      const body = await response.json() as { model?: unknown; fingerprint?: unknown; result: T };
      if (body.model !== this.config.remoteModel || body.fingerprint !== this.config.revision) {
        throw new Error("PPLX server identity changed; restart QMD and rebuild embeddings");
      }
      const result: unknown = body.result;
      if (request.op === "decode") {
        if (typeof result !== "string") throw new Error("Invalid PPLX decoded text");
      } else {
        if (!Array.isArray(result)) throw new Error("Invalid PPLX tokenizer result");
        if (request.op === "windows") {
          let end = 0;
          for (const pair of result) {
            if (!Array.isArray(pair) || pair.length !== 2 || pair[0] !== end
              || !Number.isInteger(pair[1]) || pair[1] <= end || pair[1] > request.texts.length) {
              throw new Error("Invalid PPLX context window mapping");
            }
            end = pair[1];
          }
          if (end !== request.texts.length) throw new Error("Incomplete PPLX context windows");
        } else {
          if (!result.every(n => Number.isSafeInteger(n) && n >= 0)) throw new Error("Invalid PPLX token IDs/counts");
          if (request.op === "counts" && result.length !== (request.contextual ? 1 : request.texts.length)) {
            throw new Error("PPLX token count length mismatch");
          }
        }
      }
      return result as T;
    }
    // ponytail: one lightweight tokenizer subprocess per call, ~0.2s locally;
    // use a persistent tokenizer worker if this dominates large-corpus indexing.
    const python = process.env.QMD_PPLX_PYTHON ?? "python3";
    const helper = fileURLToPath(new URL("../scripts/pplx-tokenizer.py", import.meta.url));
    return new Promise((resolve, reject) => {
      const child = execFile(python, [helper, join(this.config.path, "tokenizer.json")],
        { timeout: 60_000, maxBuffer: 64 * 1024 * 1024, signal }, (error, stdout, stderr) => {
          if (error) return reject(new Error(`PPLX tokenizer failed (${python}): ${stderr || error.message}`));
          try { resolve(JSON.parse(stdout) as T); } catch (error) { reject(error); }
        });
      child.stdin?.on("error", () => {}); // execFile's callback reports early process exit.
      child.stdin?.end(JSON.stringify(request));
    });
  }

  async windows(texts: string[], signal?: AbortSignal): Promise<[number, number][]> {
    await this.prepare();
    if (this.config.mode === "independent") return texts.map((_, i) => [i, i + 1]);
    return this.tokenizer({ op: "windows", texts, max_tokens: this.config.maxTokens }, signal);
  }

  async embedBatch(texts: string[], options: { model?: string; isQuery?: boolean; contextual?: boolean; signal?: AbortSignal } = {}) {
    if (options.model && options.model !== this.model) throw new Error("PPLX request model differs from the configured model");
    if (texts.length === 0) return [];
    await this.prepare();
    if (options.contextual && options.isQuery) throw new Error("Queries cannot share document context");
    if (options.contextual && this.config.mode !== "contextual") throw new Error("Independent PPLX profile cannot request shared context");
    if (texts.some(text => typeof text !== "string" || !text.trim())) throw new Error("PPLX inputs must be nonempty strings");
    const results: { embedding: number[]; model: string }[] = [];
    const batchSize = options.contextual ? texts.length : 64;
    for (let start = 0; start < texts.length; start += batchSize) {
      const batch = texts.slice(start, start + batchSize);
      // Budget the exact prefixed/separated input before contacting the encoder.
      const counts = await this.tokenizer<number[]>({ op: "counts", texts: batch,
        task: options.isQuery ? "query" : "document", contextual: !!options.contextual }, options.signal);
      if (counts.some(n => n > this.config.maxTokens)) throw new Error(`PPLX input exceeds ${this.config.maxTokens} tokens; split it without truncation`);
      const timeout = AbortSignal.timeout(180_000);
      const response = await fetch(this.endpoint, {
        method: "POST", redirect: "error",
        signal: options.signal ? AbortSignal.any([timeout, options.signal]) : timeout,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: this.config.remoteModel ?? this.config.path,
          input: options.contextual ? { chunks: batch } : batch,
          task: options.isQuery ? "query" : "document", dimensions: this.config.dimensions,
          normalize_embeddings: true, encoding_format: "float",
          ...(this.config.remote ? { expected_fingerprint: this.config.revision } : {}) }),
      });
      if (!response.ok) throw new Error(`PPLX HTTP ${response.status}: ${(await response.text()).slice(0, 500)}`);
      const body = await response.json() as { model?: unknown; data?: unknown };
      if (body.model !== (this.config.remoteModel ?? this.config.path) || !Array.isArray(body.data) || body.data.length !== batch.length) {
        throw new Error("PPLX response model or vector count mismatch; no vectors stored");
      }
      const ordered = new Array<{ embedding: number[]; model: string }>(batch.length);
      for (const row of body.data) {
        const { index, input_index, chunk_index, embedding } = row ?? {};
        if (!Number.isInteger(index) || index < 0 || index >= batch.length || ordered[index]
          || input_index !== (options.contextual ? 0 : index) || chunk_index !== (options.contextual ? index : 0)
          || !Array.isArray(embedding) || embedding.length !== this.config.dimensions
          || !embedding.every((n: unknown) => typeof n === "number" && Number.isFinite(n))) {
          throw new Error("Invalid PPLX vector dimensions, values, or chunk mapping; no vectors stored");
        }
        const norm = Math.hypot(...embedding);
        if (Math.abs(norm - 1) > 0.001) throw new Error("PPLX returned a non-unit vector; no vectors stored");
        ordered[index] = { embedding, model: this.model };
      }
      results.push(...ordered);
    }
    return results;
  }
}
