import { execFile } from "node:child_process";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

type TokenizerRequest =
  | { op: "encode"; text: string }
  | { op: "decode"; tokens: readonly number[] }
  | { op: "windows"; texts: string[]; max_tokens: number }
  | { op: "counts"; texts: string[]; task: "query" | "document"; contextual: boolean };

export const isPplxModel = (model: string): boolean => model.startsWith("pplx:");

/** All vector-affecting settings belong in the model URI, not environment variables. */
export function parsePplxModel(model: string) {
  const uri = new URL(model);
  const allowed = ["dimensions", "mode", "max_tokens", "revision"];
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
  return { path: decodeURIComponent(uri.pathname), dimensions, maxTokens, mode, revision };
}

/** Model-specific transport; generation and reranking stay in node-llama-cpp. */
export class PplxEmbeddings {
  readonly config;
  private readonly endpoint: string;

  constructor(readonly model: string) {
    this.config = parsePplxModel(model);
    const url = new URL(process.env.QMD_PPLX_URL ?? "http://127.0.0.1:8080/v1/embeddings");
    if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.hash) {
      throw new Error("QMD_PPLX_URL must be an HTTP(S) embeddings endpoint without credentials or fragment");
    }
    this.endpoint = url.href;
  }

  async tokenizer<T>(request: TokenizerRequest, signal?: AbortSignal): Promise<T> {
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
    if (this.config.mode === "independent") return texts.map((_, i) => [i, i + 1]);
    return this.tokenizer({ op: "windows", texts, max_tokens: this.config.maxTokens }, signal);
  }

  async embedBatch(texts: string[], options: { model?: string; isQuery?: boolean; contextual?: boolean; signal?: AbortSignal } = {}) {
    if (options.model && options.model !== this.model) throw new Error("PPLX request model differs from the configured model");
    if (texts.length === 0) return [];
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
        body: JSON.stringify({ model: this.config.path,
          input: options.contextual ? { chunks: batch } : batch,
          task: options.isQuery ? "query" : "document", dimensions: this.config.dimensions,
          normalize_embeddings: true, encoding_format: "float" }),
      });
      if (!response.ok) throw new Error(`PPLX HTTP ${response.status}: ${(await response.text()).slice(0, 500)}`);
      const body = await response.json() as { model?: unknown; data?: unknown };
      if (body.model !== this.config.path || !Array.isArray(body.data) || body.data.length !== batch.length) {
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
