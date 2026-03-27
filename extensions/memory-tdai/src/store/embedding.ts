/**
 * Embedding Service: converts text to vector embeddings.
 *
 * Supports two providers:
 * - "openai": OpenAI-compatible embedding APIs (OpenAI, Azure OpenAI, self-hosted)
 * - "local": node-llama-cpp with embeddinggemma-300m GGUF model (fully offline)
 *
 * When no remote embedding is configured, automatically falls back to local provider.
 *
 * Design:
 * - Single `embed()` for one text, `embedBatch()` for multiple.
 * - `getDimensions()` returns configured vector dimensions.
 * - Throws on failure; callers decide fallback strategy.
 */

// ============================
// Types
// ============================

export interface OpenAIEmbeddingConfig {
  /** Provider identifier — any value other than "local" (e.g. "openai", "deepseek", "azure") */
  provider: string;
  /** API base URL (required — must be specified by user, e.g. "https://api.openai.com/v1") */
  baseUrl: string;
  /** API Key (required) */
  apiKey: string;
  /** Model name (required — must be specified by user) */
  model: string;
  /** Output dimensions (required — must match the chosen model) */
  dimensions: number;
}

export interface LocalEmbeddingConfig {
  provider: "local";
  /** Custom GGUF model path (default: embeddinggemma-300m from HuggingFace) */
  modelPath?: string;
  /** Model cache directory (default: node-llama-cpp default cache) */
  modelCacheDir?: string;
}

export type EmbeddingConfig = OpenAIEmbeddingConfig | LocalEmbeddingConfig;

/** Identifies the embedding provider + model for change detection. */
export interface EmbeddingProviderInfo {
  /** Provider identifier (e.g. "local", "openai", "deepseek") */
  provider: string;
  /** Model identifier (e.g. "embeddinggemma-300m", "text-embedding-3-large") */
  model: string;
}

export interface EmbeddingService {
  /** Get embedding for a single text */
  embed(text: string): Promise<Float32Array>;
  /** Get embeddings for multiple texts (batched API call) */
  embedBatch(texts: string[]): Promise<Float32Array[]>;
  /** Return the configured vector dimensions */
  getDimensions(): number;
  /** Return provider + model identifiers for change detection */
  getProviderInfo(): EmbeddingProviderInfo;
  /**
   * Whether the service is ready to serve embed requests.
   * For remote providers (OpenAI), always true (stateless HTTP).
   * For local providers, true only after model download + load completes.
   */
  isReady(): boolean;
  /**
   * Start background warmup (model download + load).
   * For remote providers, this is a no-op.
   * For local providers, triggers async initialization without blocking.
   * Safe to call multiple times (idempotent).
   */
  startWarmup(): void;
  /** Optional: release resources (model memory, GPU, etc.) on shutdown */
  close?(): void | Promise<void>;
}

/**
 * Error thrown when embed() / embedBatch() is called before the local
 * embedding model has finished downloading and loading.
 * Callers should catch this and fall back to keyword-only mode.
 */
export class EmbeddingNotReadyError extends Error {
  constructor(message?: string) {
    super(message ?? "Local embedding model is not ready yet (still downloading or loading)");
    this.name = "EmbeddingNotReadyError";
  }
}

// ============================
// Logger interface
// ============================

interface Logger {
  debug?: (message: string) => void;
  info: (message: string) => void;
  warn: (message: string) => void;
  error: (message: string) => void;
}

const TAG = "[memory-tdai][embedding]";

// ============================
// Local (node-llama-cpp) implementation
// ============================

/** Default model: Google's embeddinggemma-300m, quantized Q8_0 (~300MB) */
const DEFAULT_LOCAL_MODEL =
  "hf:ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/embeddinggemma-300m-qat-Q8_0.gguf";

/** embeddinggemma-300m outputs 768-dimensional vectors */
const LOCAL_DIMENSIONS = 768;

/**
 * embeddinggemma-300m has a 256-token context window.
 * As a safe heuristic, we limit input to ~600 chars for CJK text
 * (CJK characters typically tokenize to 1-2 tokens each,
 *  so 600 chars ≈ 200-400 tokens, keeping well within 256-token limit
 *  after accounting for special tokens).
 * For Latin text, ~800 chars is a safe limit (~200 tokens).
 * We use 512 chars as a conservative universal limit.
 */
const LOCAL_MAX_INPUT_CHARS = 512;

/**
 * Sanitize NaN/Inf values and L2-normalize the vector.
 * Matches OpenClaw's own sanitizeAndNormalizeEmbedding().
 */
function sanitizeAndNormalize(vec: number[] | Float32Array): Float32Array {
  const arr = Array.from(vec).map((v) => (Number.isFinite(v) ? v : 0));
  const magnitude = Math.sqrt(arr.reduce((sum, v) => sum + v * v, 0));
  if (magnitude < 1e-10) {
    return new Float32Array(arr);
  }
  return new Float32Array(arr.map((v) => v / magnitude));
}

/**
 * Initialization state for LocalEmbeddingService.
 * - "idle":         not started yet
 * - "initializing": model download / load is in progress (background)
 * - "ready":        model is loaded and ready to serve
 * - "failed":       initialization failed (will retry on next startWarmup)
 */
type LocalInitState = "idle" | "initializing" | "ready" | "failed";

export class LocalEmbeddingService implements EmbeddingService {
  private readonly modelPath: string;
  private readonly modelCacheDir?: string;
  private readonly logger?: Logger;

  // Initialization state machine
  private initState: LocalInitState = "idle";
  private initPromise: Promise<void> | null = null;
  private initError: Error | null = null;
  private embeddingContext: {
    getEmbeddingFor: (text: string) => Promise<{ vector: Float32Array | number[] }>;
  } | null = null;

  constructor(config?: LocalEmbeddingConfig, logger?: Logger) {
    this.modelPath = config?.modelPath?.trim() || DEFAULT_LOCAL_MODEL;
    this.modelCacheDir = config?.modelCacheDir?.trim();
    this.logger = logger;
  }

  getDimensions(): number {
    return LOCAL_DIMENSIONS;
  }

  getProviderInfo(): EmbeddingProviderInfo {
    return { provider: "local", model: this.modelPath };
  }

  /**
   * Whether the local model is fully loaded and ready to serve requests.
   */
  isReady(): boolean {
    return this.initState === "ready" && this.embeddingContext !== null;
  }

  /**
   * Start background warmup: download model (if needed) and load into memory.
   * Does NOT block the caller — returns immediately.
   * Safe to call multiple times (idempotent); re-triggers on "failed" state.
   */
  startWarmup(): void {
    if (this.initState === "initializing" || this.initState === "ready") {
      return; // already in progress or done
    }
    this.logger?.info(`${TAG} Starting background warmup for local embedding model...`);
    this.initState = "initializing";
    this.initError = null;

    this.initPromise = this._doInitialize()
      .then(() => {
        this.initState = "ready";
        this.logger?.info(`${TAG} Background warmup complete — local embedding ready`);
      })
      .catch((err) => {
        this.initState = "failed";
        this.initError = err instanceof Error ? err : new Error(String(err));
        this.logger?.error(
          `${TAG} Background warmup failed: ${this.initError.message}. ` +
          `embed() calls will throw EmbeddingNotReadyError until retried.`,
        );
      });
  }

  /**
   * Get embedding for a single text.
   * @throws {EmbeddingNotReadyError} if model is not yet ready.
   */
  async embed(text: string): Promise<Float32Array> {
    this.assertReady();
    const truncated = this.truncateInput(text);
    const embedding = await this.embeddingContext!.getEmbeddingFor(truncated);
    return sanitizeAndNormalize(embedding.vector);
  }

  /**
   * Get embeddings for multiple texts.
   * @throws {EmbeddingNotReadyError} if model is not yet ready.
   */
  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    if (texts.length === 0) return [];
    this.assertReady();

    const results: Float32Array[] = [];
    for (const text of texts) {
      const truncated = this.truncateInput(text);
      const embedding = await this.embeddingContext!.getEmbeddingFor(truncated);
      results.push(sanitizeAndNormalize(embedding.vector));
    }
    return results;
  }

  /**
   * Release the node-llama-cpp embedding context and model resources.
   * Safe to call multiple times (idempotent).
   */
  close(): void {
    if (this.embeddingContext) {
      try {
        const ctx = this.embeddingContext as unknown as { dispose?: () => void };
        ctx.dispose?.();
      } catch {
        // best-effort cleanup
      }
      this.embeddingContext = null;
      this.initPromise = null;
      this.initState = "idle";
      this.initError = null;
      this.logger?.info(`${TAG} Local embedding resources released`);
    }
  }

  /**
   * Assert the model is ready. Throws EmbeddingNotReadyError if not.
   */
  private assertReady(): void {
    if (this.initState === "ready" && this.embeddingContext) {
      return;
    }
    if (this.initState === "failed") {
      throw new EmbeddingNotReadyError(
        `Local embedding model initialization failed: ${this.initError?.message ?? "unknown error"}. ` +
        `Call startWarmup() to retry.`,
      );
    }
    if (this.initState === "initializing") {
      throw new EmbeddingNotReadyError(
        "Local embedding model is still loading (download/initialization in progress). Please try again later.",
      );
    }
    // "idle" — startWarmup() was never called
    throw new EmbeddingNotReadyError(
      "Local embedding model warmup has not been started. Call startWarmup() first.",
    );
  }

  /**
   * Truncate input text to stay within the model's context window.
   * embeddinggemma-300m has a 256-token limit; we use a character-based
   * heuristic (LOCAL_MAX_INPUT_CHARS) as a safe proxy.
   */
  private truncateInput(text: string): string {
    if (text.length <= LOCAL_MAX_INPUT_CHARS) return text;
    this.logger?.debug?.(
      `${TAG} Input truncated from ${text.length} to ${LOCAL_MAX_INPUT_CHARS} chars (model context limit)`,
    );
    return text.slice(0, LOCAL_MAX_INPUT_CHARS);
  }

  /**
   * Internal: perform the actual model download + load.
   * Called by startWarmup(), runs in background.
   */
  private async _doInitialize(): Promise<void> {
    // Track partially-initialized resources for cleanup on failure
    let model: { createEmbeddingContext: () => Promise<unknown>; dispose?: () => void } | undefined;
    try {
      this.logger?.debug?.(`${TAG} Loading node-llama-cpp for local embedding...`);

      // Dynamic import — node-llama-cpp is a peer dependency of OpenClaw
      const { getLlama, resolveModelFile, LlamaLogLevel } = await import("node-llama-cpp");

      const llama = await getLlama({ logLevel: LlamaLogLevel.error });
      this.logger?.debug?.(`${TAG} Llama instance created`);

      const resolvedPath = await resolveModelFile(
        this.modelPath,
        this.modelCacheDir || undefined,
      );
      this.logger?.debug?.(`${TAG} Model resolved: ${resolvedPath}`);

      model = await (llama as unknown as { loadModel: (opts: { modelPath: string }) => Promise<typeof model> }).loadModel({ modelPath: resolvedPath });
      this.logger?.debug?.(`${TAG} Model loaded, creating embedding context...`);

      this.embeddingContext = await model!.createEmbeddingContext() as typeof this.embeddingContext;
      this.logger?.info(`${TAG} Local embedding ready (model=${this.modelPath}, dims=${LOCAL_DIMENSIONS})`);
    } catch (err) {
      // Clean up partially-initialized resources to prevent leaks
      if (model?.dispose) {
        try { model.dispose(); } catch { /* best-effort */ }
      }
      this.embeddingContext = null;
      throw err;
    }
  }

  /**
   * Wait for ongoing warmup to complete (used internally by tests).
   * Returns immediately if already ready or idle.
   */
  async waitForReady(): Promise<void> {
    if (this.initPromise) {
      await this.initPromise;
    }
  }
}

// ============================
// OpenAI-compatible implementation
// ============================

/** Max texts per batch (OpenAI limit is 2048, we use a safe value) */
const MAX_BATCH_SIZE = 256;

/** Max retries for API calls */
const MAX_RETRIES = 2;
/** Timeout per API call in milliseconds */
const API_TIMEOUT_MS = 90_000;

/**
 * Custom error class for embedding API errors that carries HTTP status code.
 * Used to distinguish non-retryable client errors (4xx except 429) from
 * retryable server errors (5xx) and rate limits (429).
 */
class EmbeddingApiError extends Error {
  readonly httpStatus: number;
  constructor(message: string, httpStatus: number) {
    super(message);
    this.name = "EmbeddingApiError";
    this.httpStatus = httpStatus;
  }
  /** Returns true for 4xx errors that should NOT be retried (excluding 429). */
  isClientError(): boolean {
    return this.httpStatus >= 400 && this.httpStatus < 500 && this.httpStatus !== 429;
  }
}

interface OpenAIEmbeddingResponse {
  data: Array<{
    index: number;
    embedding: number[];
  }>;
  usage?: {
    prompt_tokens: number;
    total_tokens: number;
  };
}

export class OpenAIEmbeddingService implements EmbeddingService {
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly model: string;
  private readonly dims: number;
  private readonly providerName: string;

  constructor(config: OpenAIEmbeddingConfig) {
    if (!config.apiKey) {
      throw new Error("EmbeddingService: apiKey is required for remote provider");
    }
    if (!config.baseUrl) {
      throw new Error("EmbeddingService: baseUrl is required for remote provider");
    }
    if (!config.model) {
      throw new Error("EmbeddingService: model is required for remote provider");
    }
    if (!config.dimensions || config.dimensions <= 0) {
      throw new Error("EmbeddingService: dimensions is required for remote provider (must be a positive integer)");
    }
    this.baseUrl = config.baseUrl.replace(/\/+$/, "");
    this.apiKey = config.apiKey;
    this.model = config.model;
    this.dims = config.dimensions;
    this.providerName = config.provider || "openai";
  }

  getDimensions(): number {
    return this.dims;
  }

  getProviderInfo(): EmbeddingProviderInfo {
    return { provider: this.providerName, model: this.model };
  }

  /** Remote embedding is always ready (stateless HTTP). */
  isReady(): boolean {
    return true;
  }

  /** No-op for remote embedding (no local model to warm up). */
  startWarmup(): void {
    // nothing to do — remote API is stateless
  }

  async embed(text: string): Promise<Float32Array> {
    const [result] = await this.embedBatch([text]);
    return result;
  }

  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    if (texts.length === 0) return [];

    // Split into sub-batches if needed
    if (texts.length > MAX_BATCH_SIZE) {
      const results: Float32Array[] = [];
      for (let i = 0; i < texts.length; i += MAX_BATCH_SIZE) {
        const chunk = texts.slice(i, i + MAX_BATCH_SIZE);
        const chunkResults = await this._callApi(chunk);
        results.push(...chunkResults);
      }
      return results;
    }

    return this._callApi(texts);
  }

  private async _callApi(texts: string[]): Promise<Float32Array[]> {
    const body: Record<string, unknown> = {
      input: texts,
      model: this.model,
      dimensions: this.dims,
    };

    // Retry loop with timeout
    let lastError: Error | undefined;
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT_MS);

        try {
          const resp = await fetch(`${this.baseUrl}/embeddings`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${this.apiKey}`,
            },
            body: JSON.stringify(body),
            signal: controller.signal,
          });

          if (!resp.ok) {
            const errBody = await resp.text().catch(() => "(unable to read body)");
            const err = new EmbeddingApiError(
              `Embedding API error: HTTP ${resp.status} ${resp.statusText} — ${errBody.slice(0, 500)}`,
              resp.status,
            );
            // Don't retry on 4xx client errors (except 429 rate limit)
            if (resp.status >= 400 && resp.status < 500 && resp.status !== 429) {
              throw err;
            }
            lastError = err;
            continue;
          }

          const json = (await resp.json()) as OpenAIEmbeddingResponse;

          if (!json.data || !Array.isArray(json.data)) {
            throw new Error("Embedding API returned unexpected format: missing 'data' array");
          }

          // Sort by index to ensure correct order, then sanitize+normalize for consistency with local provider
          const sorted = [...json.data].sort((a, b) => a.index - b.index);
          return sorted.map((d) => sanitizeAndNormalize(d.embedding));
        } finally {
          clearTimeout(timeoutId);
        }
      } catch (err) {
        // Non-retryable errors (4xx client errors) — rethrow immediately
        if (err instanceof EmbeddingApiError && err.isClientError()) {
          throw err;
        }
        lastError = err instanceof Error ? err : new Error(String(err));
        // AbortError = timeout, retry
        if (attempt < MAX_RETRIES) {
          // Exponential backoff: 500ms, 1000ms
          const delay = 500 * (attempt + 1);
          await new Promise((r) => setTimeout(r, delay));
        }
      }
    }

    throw lastError ?? new Error("Embedding API call failed after retries");
  }
}

// ============================
// Factory
// ============================

/**
 * Create an EmbeddingService from config.
 *
 * Strategy:
 * - If config has provider != "local" with valid apiKey, model, and dimensions → use remote OpenAI-compatible embedding
 * - If config has provider="local" → use node-llama-cpp local embedding
 * - If config is undefined or missing required fields → fall back to local embedding
 *
 * NOTE: For local providers, `startWarmup()` is NOT called here.
 * The caller is responsible for calling `startWarmup()` at the right time
 * (e.g. on first conversation) to avoid triggering model download during
 * short-lived CLI commands like `gateway stop` or `agents list`.
 */
export function createEmbeddingService(
  config: EmbeddingConfig | undefined,
  logger?: Logger,
): EmbeddingService {
  // Remote OpenAI-compatible provider: any provider value other than "local"
  if (config && config.provider !== "local" && "apiKey" in config && config.apiKey) {
    logger?.info(`${TAG} Using remote embedding (provider=${config.provider}, model=${config.model})`);
    return new OpenAIEmbeddingService(config as OpenAIEmbeddingConfig);
  }

  // Explicit local config
  if (config && config.provider === "local") {
    const localConfig = config as LocalEmbeddingConfig;
    logger?.info(`${TAG} Using local embedding (node-llama-cpp, model=${localConfig.modelPath ?? DEFAULT_LOCAL_MODEL})`);
    return new LocalEmbeddingService(localConfig, logger);
  }

  // Fallback: no config or empty apiKey → use local
  logger?.info(`${TAG} No remote embedding configured, falling back to local embedding (node-llama-cpp)`);
  return new LocalEmbeddingService(undefined, logger);
}
