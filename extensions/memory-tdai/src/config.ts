/**
 * Plugin configuration types and parser (v3).
 *
 * Config is organized into flat functional groups:
 *   capture, extraction, persona, pipeline, recall, embedding
 *
 * Minimal config (zero config): {} — all fields have sensible defaults.
 */

// ============================
// Type definitions
// ============================

/** Capture settings — controls L0 conversation recording. */
export interface CaptureConfig {
  /** Enable auto-capture (default: true) */
  enabled: boolean;
  /** Glob patterns to exclude agents (e.g. "bench-judge-*"); matched agents are fully ignored */
  excludeAgents: string[];
  /**
   * L0/L1 local file retention days used as TTL switch.
   * 0 means cleanup disabled.(default: 0)
   */
  l0l1RetentionDays: number;

  /**
   * Allow dangerous low retention (1 or 2 days).
   * Default false: when disabled, non-zero retention must be >= 3.
   */
  allowAggressiveCleanup: boolean;
}

/** Extraction settings (L1) — controls memory extraction from conversations. */
export interface ExtractionConfig {
  /** Enable background extraction (default: true) */
  enabled: boolean;
  /** Enable L1 smart dedup (default: true) */
  enableDedup: boolean;
  /** Max memories per session (default: 10) */
  maxMemoriesPerSession: number;
  /** LLM model for extraction, format: "provider/model" */
  model?: string;
}

/** Persona (L2/L3) settings — controls scene extraction (L2) and user profile generation (L3). */
export interface PersonaConfig {
  /** Trigger persona generation every N new memories (default: 50) */
  triggerEveryN: number;
  /** Max scene blocks (default: 20) */
  maxScenes: number;
  /** Persona backup count (default: 3) */
  backupCount: number;
  /** Scene blocks backup count (default: 10) */
  sceneBackupCount: number;
  /** LLM model for persona generation, format: "provider/model" */
  model?: string;
}

/** Pipeline trigger settings (L1→L2→L3 scheduling). */
export interface PipelineTriggerConfig {
  /** Trigger L1 after every N conversation rounds (default: 5) */
  everyNConversations: number;
  /** Enable warm-up: start threshold at 1, double after each L1 (1→2→4→...→everyN) (default: true) */
  enableWarmup: boolean;
  /** L1 idle timeout: trigger L1 after this many seconds of inactivity (default: 60) */
  l1IdleTimeoutSeconds: number;
  /** L2 delay after L1: wait this many seconds after L1 completes before triggering L2 (default: 90) */
  l2DelayAfterL1Seconds: number;
  /** L2 min interval: minimum seconds between L2 runs per session (default: 300 = 5 min) */
  l2MinIntervalSeconds: number;
  /** L2 max interval: even without new conversations, trigger L2 at most this often per session (default: 1800 = 30 min) */
  l2MaxIntervalSeconds: number;
  /** Sessions inactive longer than this (hours) stop L2 polling (default: 24) */
  sessionActiveWindowHours: number;
}

/** Recall settings — controls memory retrieval for context injection. */
export interface RecallConfig {
  /** Enable auto-recall (default: true) */
  enabled: boolean;
  /** Max results to return (default: 5) */
  maxResults: number;
  /** Minimum score threshold (default: 0.3) */
  scoreThreshold: number;
  /** Search strategy (default: "hybrid") */
  strategy: "embedding" | "keyword" | "hybrid";
}

/** Embedding service configuration for vector search. */
export interface EmbeddingConfig {
  /** Enable vector search (default: false for "none" provider; auto-set to false if remote config is incomplete) */
  enabled: boolean;
  /** Embedding provider: "none" disables embedding (default); any other value (e.g. "openai", "deepseek") uses OpenAI-compatible remote API */
  provider: string;
  /** API Base URL — remote mode: must be specified by user; local mode: unused */
  baseUrl: string;
  /** API Key — remote mode: must be specified by user; local mode: unused */
  apiKey: string;
  /** Model name — remote: must be specified by user; local: GGUF model path (local defaults to built-in model) */
  model: string;
  /** Vector dimensions — remote: must be specified by user (must match the chosen model); local: default 768 */
  dimensions: number;
  /** Top-K candidates to recall during conflict detection (default: 5) */
  conflictRecallTopK: number;
  /** Model cache directory for local provider (optional) */
  modelCacheDir?: string;
  /** If set, contains an error message about invalid remote config (embedding is disabled) */
  configError?: string;
}

/** Daily cleaner settings for local JSONL data (L0/L1). */
export interface MemoryCleanupConfig {
  /** TTL switch from capture.l0l1RetentionDays. Undefined means disabled. */
  retentionDays?: number;

  /** Whether cleanup is enabled. True only when retentionDays is a valid positive number. */
  enabled: boolean;
  /** Daily execution time in HH:mm format (default: 03:00). */
  cleanTime: string;
  /** L0 directory path under plugin data dir (default: conversations). */
  l0Dir: string;
  /** L1 directory path under plugin data dir (default: records). */
  l1Dir: string;
}

/** Fully resolved plugin configuration (v3). */
export interface MemoryTdaiConfig {
  capture: CaptureConfig;
  extraction: ExtractionConfig;
  persona: PersonaConfig;
  pipeline: PipelineTriggerConfig;
  recall: RecallConfig;
  embedding: EmbeddingConfig;
  /** Local JSONL cleanup settings */
  memoryCleanup: MemoryCleanupConfig;
}

// ============================
// Parser
// ============================

/**
 * Parse plugin config from raw user input.
 * All fields have sensible defaults — minimal config is just {}.
 */
export function parseConfig(raw: Record<string, unknown> | undefined): MemoryTdaiConfig {
  const c = raw ?? {};

  // --- Capture (L0) ---
  const captureGroup = obj(c, "capture");

  // --- Retention days validation (from capture.l0l1RetentionDays) ---
  const rawRetentionDays = num(captureGroup, "l0l1RetentionDays") ?? 0;
  const allowAggressiveCleanup = bool(captureGroup, "allowAggressiveCleanup") ?? false;

  let retentionDays: number | undefined;
  if (rawRetentionDays <= 0) {
    retentionDays = undefined;
  } else if (rawRetentionDays >= 3) {
    retentionDays = rawRetentionDays;
  } else if (allowAggressiveCleanup) {
    retentionDays = rawRetentionDays;
  } else {
    retentionDays = undefined;
  }

  // --- Extraction (L1) ---
  const extractionGroup = obj(c, "extraction");

  // --- Persona (L2/L3) ---
  const personaGroup = obj(c, "persona");

  // --- Pipeline ---
  const pipelineGroup = obj(c, "pipeline");

  // --- Recall ---
  const recallGroup = obj(c, "recall");

  // --- Embedding ---
  const embeddingGroup = obj(c, "embedding");
  let embeddingConfigError: string | undefined;

  // Embedding config: determine provider based on user input and apiKey availability
  const embeddingApiKey = str(embeddingGroup, "apiKey") ?? "";
  const embeddingBaseUrl = str(embeddingGroup, "baseUrl") ?? "";
  const embeddingProviderRaw = str(embeddingGroup, "provider") ?? "none";
  const embeddingModelRaw = str(embeddingGroup, "model") ?? "";
  const embeddingDimensionsRaw = num(embeddingGroup, "dimensions");

  // provider="none" → embedding disabled (default for zero-config users)
  // provider="local" → no longer exposed to users; treated as disabled at entry level
  // Any other value → remote mode (requires apiKey, baseUrl, model, dimensions)
  let embeddingProvider: string;
  let embeddingEnabled = bool(embeddingGroup, "enabled") ?? true;

  if (embeddingProviderRaw === "none") {
    // Explicitly disabled (default): no embedding, no vector search
    embeddingProvider = "none";
    embeddingEnabled = false;
  } else if (embeddingProviderRaw === "local") {
    // Local embedding is not exposed to users; treat as disabled at entry level.
    // Internal LocalEmbeddingService code is preserved but not reachable from config.
    embeddingProvider = "none";
    embeddingEnabled = false;
    embeddingConfigError =
      "Local embedding provider is not available in user config. " +
      "Please configure a remote embedding provider (e.g. openai, deepseek). Embedding has been disabled.";
  } else {
    // Remote mode — validate all required fields
    const missingFields: string[] = [];
    if (!embeddingApiKey) missingFields.push("apiKey");
    if (!embeddingBaseUrl) missingFields.push("baseUrl");
    if (!embeddingModelRaw) missingFields.push("model");
    if (embeddingDimensionsRaw == null || embeddingDimensionsRaw <= 0) missingFields.push("dimensions");

    if (missingFields.length > 0) {
      // Configuration error: disable embedding and log detailed error
      // This does NOT throw — the plugin continues running without vector search
      const errorMsg =
        `Remote embedding provider '${embeddingProviderRaw}' requires 'apiKey', 'baseUrl', 'model', and 'dimensions' to be set. ` +
        `Missing: ${missingFields.join(", ")}. Embedding has been disabled.`;
      // We store the error message so the caller (index.ts) can log it
      embeddingConfigError = errorMsg;
      embeddingEnabled = false;
      embeddingProvider = embeddingProviderRaw; // preserve original for error context
    } else {
      embeddingProvider = embeddingProviderRaw;
    }
  }

  // When provider="none", dimensions=0 signals VectorStore to skip vec0 table
  // creation entirely (deferred until a real embedding provider is configured).
  // This avoids creating vec0 tables with a placeholder dimension that would
  // mismatch if the user later enables a different-dimensional provider.
  const defaultDimensions =
    embeddingProvider === "none" ? 0 :
    embeddingDimensionsRaw ?? 0;
  const defaultModel = embeddingProvider === "none" ? "" : embeddingModelRaw;

  const cleanTime = normalizeCleanTime(str(captureGroup, "cleanTime")) ?? "03:00";
  const l0Dir = str(captureGroup, "l0Dir") ?? "conversations";
  const l1Dir = str(captureGroup, "l1Dir") ?? "records";

  const memoryCleanup: MemoryCleanupConfig = {
    retentionDays,
    enabled: retentionDays != null,
    cleanTime,
    l0Dir,
    l1Dir,
  };

  return {
    capture: {
      enabled: bool(captureGroup, "enabled") ?? true,
      excludeAgents: strArray(captureGroup, "excludeAgents") ?? [],
      l0l1RetentionDays: retentionDays ?? 0,
      allowAggressiveCleanup,
    },
    extraction: {
      enabled: bool(extractionGroup, "enabled") ?? true,
      enableDedup: bool(extractionGroup, "enableDedup") ?? true,
      maxMemoriesPerSession: num(extractionGroup, "maxMemoriesPerSession") ?? 10,
      model: optStr(extractionGroup, "model"),
    },
    persona: {
      triggerEveryN: num(personaGroup, "triggerEveryN") ?? 50,
      maxScenes: num(personaGroup, "maxScenes") ?? 20,
      backupCount: num(personaGroup, "backupCount") ?? 3,
      sceneBackupCount: num(personaGroup, "sceneBackupCount") ?? 10,
      model: optStr(personaGroup, "model"),
    },
    pipeline: {
      everyNConversations: num(pipelineGroup, "everyNConversations") ?? 5,
      enableWarmup: bool(pipelineGroup, "enableWarmup") ?? true,
      l1IdleTimeoutSeconds: num(pipelineGroup, "l1IdleTimeoutSeconds") ?? 60,
      l2DelayAfterL1Seconds: num(pipelineGroup, "l2DelayAfterL1Seconds") ?? 90,
      l2MinIntervalSeconds: num(pipelineGroup, "l2MinIntervalSeconds") ?? 300,
      l2MaxIntervalSeconds: num(pipelineGroup, "l2MaxIntervalSeconds") ?? 1800,
      sessionActiveWindowHours: num(pipelineGroup, "sessionActiveWindowHours") ?? 24,
    },
    recall: {
      enabled: bool(recallGroup, "enabled") ?? true,
      maxResults: num(recallGroup, "maxResults") ?? 5,
      scoreThreshold: num(recallGroup, "scoreThreshold") ?? 0.3,
      strategy: validateStrategy(str(recallGroup, "strategy")) ?? "hybrid",
    },
    embedding: {
      enabled: embeddingEnabled,
      provider: embeddingProvider,
      baseUrl: embeddingBaseUrl,
      apiKey: embeddingApiKey,
      model: str(embeddingGroup, "model") ?? defaultModel,
      dimensions: num(embeddingGroup, "dimensions") ?? defaultDimensions,
      conflictRecallTopK: num(embeddingGroup, "conflictRecallTopK") ?? 5,
      modelCacheDir: optStr(embeddingGroup, "modelCacheDir"),
      configError: embeddingConfigError,
    },
    memoryCleanup,
  };
}

// ============================
// Helper functions
// ============================

/** Get sub-object by key, or empty object if missing. */
function obj(c: Record<string, unknown>, key: string): Record<string, unknown> {
  const v = c[key];
  return v && typeof v === "object" && !Array.isArray(v) ? v as Record<string, unknown> : {};
}

function str(src: Record<string, unknown>, key: string): string | undefined {
  const v = src[key];
  return typeof v === "string" && v.trim() ? v.trim() : undefined;
}

function optStr(src: Record<string, unknown>, key: string): string | undefined {
  const v = src[key];
  return typeof v === "string" ? v : undefined;
}

function num(src: Record<string, unknown>, key: string): number | undefined {
  const v = src[key];
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

function bool(src: Record<string, unknown>, key: string): boolean | undefined {
  const v = src[key];
  return typeof v === "boolean" ? v : undefined;
}

function strArray(src: Record<string, unknown>, key: string): string[] | undefined {
  const v = src[key];
  if (!Array.isArray(v)) return undefined;
  return v.filter((item): item is string => typeof item === "string" && item.trim().length > 0);
}

const VALID_STRATEGIES: RecallConfig["strategy"][] = ["embedding", "keyword", "hybrid"];

/**
 * Validate recall strategy against whitelist.
 * Returns the strategy if valid, undefined otherwise (caller falls back to default).
 */
function validateStrategy(value: string | undefined): RecallConfig["strategy"] | undefined {
  if (!value) return undefined;
  return VALID_STRATEGIES.includes(value as RecallConfig["strategy"])
    ? (value as RecallConfig["strategy"])
    : undefined;
}

/**
 * Normalize a cleanup time string.
 *
 * The input must follow "HH:MM" or "H:MM" format (24-hour clock).
 * If the time is valid, it returns the normalized format "HH:MM"
 * with leading zeros added when necessary.
 * If the format is invalid or the time is out of range
 * (hour: 0–23, minute: 0–59), it returns undefined.
 *
 * Examples:
 * normalizeCleanTime("3:05")  -> "03:05"
 * normalizeCleanTime("03:05") -> "03:05"
 * normalizeCleanTime("23:59") -> "23:59"
 *
 * normalizeCleanTime("24:00") -> undefined   // hour out of range
 * normalizeCleanTime("12:60") -> undefined   // minute out of range
 * normalizeCleanTime("3:5")   -> undefined   // minute must have two digits
 * normalizeCleanTime("abc")   -> undefined   // invalid format
 */
function normalizeCleanTime(input: string | undefined): string | undefined {
  if (!input) return undefined;
  const trimmed = input.trim();
  const m = /^(\d{1,2}):(\d{2})$/.exec(trimmed);
  if (!m) return undefined;

  const hh = Number(m[1]);
  const mm = Number(m[2]);
  if (!Number.isInteger(hh) || !Number.isInteger(mm)) return undefined;
  if (hh < 0 || hh > 23 || mm < 0 || mm > 59) return undefined;

  return `${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}`;
}
