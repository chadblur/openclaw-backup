/**
 * memory-tdai v3: Four-layer memory system plugin for OpenClaw.
 *
 * Provides:
 * - L0: Automatic conversation recording (local JSONL)
 * - L1: Structured memory extraction (LLM + dedup)
 * - L2: Scene block management (LLM scene extraction)
 * - L3: Persona generation (LLM persona synthesis)
 *
 * All processing is local, zero external API dependencies.
 */

import fs from "node:fs";
import path from "node:path";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk/core";
import { parseConfig } from "./src/config.js";
import type { MemoryTdaiConfig } from "./src/config.js";
import { performAutoRecall } from "./src/hooks/auto-recall.js";
import { performAutoCapture } from "./src/hooks/auto-capture.js";
import { MemoryPipelineManager } from "./src/utils/pipeline-manager.js";
import { SceneExtractor } from "./src/scene/scene-extractor.js";
import { CheckpointManager } from "./src/utils/checkpoint.js";
import { PersonaTrigger } from "./src/persona/persona-trigger.js";
import { PersonaGenerator } from "./src/persona/persona-generator.js";
import { prewarmEmbeddedAgent } from "./src/utils/clean-context-runner.js";
import { SessionFilter } from "./src/utils/session-filter.js";
import { extractL1Memories } from "./src/record/l1-extractor.js";
import { readConversationMessagesGroupedBySessionId } from "./src/conversation/l0-recorder.js";
import type { ConversationMessage } from "./src/conversation/l0-recorder.js";
import { VectorStore } from "./src/store/vector-store.js";
import { createEmbeddingService } from "./src/store/embedding.js";
import type { EmbeddingService } from "./src/store/embedding.js";
import { executeMemorySearch, formatSearchResponse } from "./src/tools/memory-search.js";
import { executeConversationSearch, formatConversationSearchResponse } from "./src/tools/conversation-search.js";
import { LocalMemoryCleaner } from "./src/utils/memory-cleaner.js";

const TAG = "[memory-tdai]";

/**
 * Initialize all required data directories under the plugin data root.
 *
 * Called once at plugin registration time so downstream modules
 * (L0 recorder, L1 writer, scene extractor, persona generator, etc.)
 * don't need to lazily mkdir on every write — the directories are
 * guaranteed to exist from startup.
 *
 * Directory layout:
 *   <pluginDataDir>/
 *   ├── conversations/   — L0 daily JSONL shards (one message per line)
 *   ├── records/          — L1 daily JSONL shards (extracted memories)
 *   ├── scene_blocks/     — L2 scene block .md files (LLM-managed)
 *   ├── .metadata/        — checkpoint, scene_index.json
 *   └── .backup/          — rotating backups (persona, scene_blocks)
 */
function initDataDirectories(dataDir: string): void {
  const dirs = [
    "conversations",
    "records",
    "scene_blocks",
    ".metadata",
    ".backup",
  ];
  for (const sub of dirs) {
    fs.mkdirSync(path.join(dataDir, sub), { recursive: true });
  }
}

/**
 * Epoch ms when the plugin was registered (cold-start timestamp).
 * Used as a fallback cursor in performAutoCapture when no checkpoint
 * exists yet — prevents the first agent_end from dumping the entire
 * session history into L0.
 */
let pluginStartTimestamp = 0;

/**
 * Cache original user prompts and message counts across hooks.
 * - text: clean user prompt before prependContext injection
 * - ts: cache creation time (for TTL sweep)
 * - messageCount: session message count at before_prompt_build time,
 *   used as fallback slice offset if timestamp cursor is unreliable
 */
const pendingOriginalPrompts = new Map<string, { text: string; ts: number; messageCount: number }>();
const PROMPT_CACHE_TTL_MS = 10 * 60 * 1000; // 10 minutes
const PROMPT_CACHE_MAX_SIZE = 10_000; // Hard limit to prevent unbounded growth in high-concurrency scenarios

// 进程级单例，避免同一进程重复启动清理器导致并发清理竞态
let sharedMemoryCleaner: LocalMemoryCleaner | undefined;

function sweepStalePromptCache(): void {
  const now = Date.now();
  for (const [key, entry] of pendingOriginalPrompts) {
    if (now - entry.ts > PROMPT_CACHE_TTL_MS) {
      pendingOriginalPrompts.delete(key);
    }
  }
  // Hard limit: if Map is still too large after TTL sweep, evict oldest entries
  if (pendingOriginalPrompts.size > PROMPT_CACHE_MAX_SIZE) {
    const entries = [...pendingOriginalPrompts.entries()].sort((a, b) => a[1].ts - b[1].ts);
    const toEvict = entries.slice(0, entries.length - PROMPT_CACHE_MAX_SIZE);
    for (const [key] of toEvict) {
      pendingOriginalPrompts.delete(key);
    }
  }
}

export default function register(api: OpenClawPluginApi) {
  pluginStartTimestamp = Date.now();
  api.logger.info(
    `${TAG} Registering plugin ... ` +
    `startTimestamp=${pluginStartTimestamp} (${new Date(pluginStartTimestamp).toISOString()})`,
  );

  let cfg: MemoryTdaiConfig;
  try {
    cfg = parseConfig(api.pluginConfig as Record<string, unknown> | undefined);
    api.logger.info(
      `${TAG} Config parsed: ` +
      `capture=${cfg.capture.enabled}, ` +
      `recall=${cfg.recall.enabled}(maxResults=${cfg.recall.maxResults}), ` +
      `extraction=${cfg.extraction.enabled}(dedup=${cfg.extraction.enableDedup}, maxMem=${cfg.extraction.maxMemoriesPerSession}), ` +
      `pipeline=(everyN=${cfg.pipeline.everyNConversations}, warmup=${cfg.pipeline.enableWarmup}, l1Idle=${cfg.pipeline.l1IdleTimeoutSeconds}s, l2DelayAfterL1=${cfg.pipeline.l2DelayAfterL1Seconds}s, l2Min=${cfg.pipeline.l2MinIntervalSeconds}s, l2Max=${cfg.pipeline.l2MaxIntervalSeconds}s, activeWindow=${cfg.pipeline.sessionActiveWindowHours}h), ` +
      `persona(triggerEvery=${cfg.persona.triggerEveryN}, backupCount=${cfg.persona.backupCount}, sceneBackupCount=${cfg.persona.sceneBackupCount}), ` +
      `memoryCleanup(enabled=${cfg.memoryCleanup.enabled}, retentionDays=${cfg.memoryCleanup.retentionDays ?? "(disabled)"}, cleanTime=${cfg.memoryCleanup.cleanTime}, l0Dir=${cfg.memoryCleanup.l0Dir}, l1Dir=${cfg.memoryCleanup.l1Dir})`,
    );
  } catch (err) {
    api.logger.error(`${TAG} Config parsing failed: ${err instanceof Error ? err.message : String(err)}`);
    throw err;
  }

  // If remote embedding config is incomplete, log a prominent error so the user knows
  if (cfg.embedding.configError) {
    api.logger.error(`${TAG} [EMBEDDING CONFIG ERROR] ${cfg.embedding.configError}`);
  }

  // Resolve plugin data directory via runtime API (avoid importing internal paths directly)
  const pluginDataDir = path.join(api.runtime.state.resolveStateDir(), "memory-tdai");
  initDataDirectories(pluginDataDir);
  api.logger.info(`${TAG} Data dir: ${pluginDataDir} (all subdirectories initialized)`);

  // Unified session/agent filter: combines internal-session detection + user-configured excludeAgents
  const sessionFilter = new SessionFilter(cfg.capture.excludeAgents);
  if (cfg.capture.excludeAgents.length > 0) {
    api.logger.info(`${TAG} Agent exclude patterns: ${cfg.capture.excludeAgents.join(", ")}`);
  }

  // Daily local JSONL cleaner (L0/L1), enabled only when retentionDays is configured.
  let memoryCleaner: LocalMemoryCleaner | undefined;
  if (cfg.memoryCleanup.enabled && cfg.memoryCleanup.retentionDays != null) {
    if (!sharedMemoryCleaner) {
      sharedMemoryCleaner = new LocalMemoryCleaner({
        baseDir: pluginDataDir,
        retentionDays: cfg.memoryCleanup.retentionDays,
        cleanTime: cfg.memoryCleanup.cleanTime,
        logger: api.logger,
      });
      sharedMemoryCleaner.start();
      api.logger.info(`${TAG} Memory cleaner started (singleton)`);
    } else {
      api.logger.info(`${TAG} Memory cleaner already started in this process, reusing existing instance`);
    }
    memoryCleaner = sharedMemoryCleaner;
  } else {
    api.logger.info(`${TAG} Memory cleaner disabled (retentionDays not configured)`);
  }

  // Hardcoded actor ID (legacy, to be removed)
  const ACTOR_ID = "default_user";

  const resolveSessionKey = (sessionKey?: string): string | undefined => {
    if (sessionKey) return sessionKey;
    api.logger.warn(`${TAG} sessionKey is empty, skipping capture/recall to avoid unstable fallback key`);
    return undefined;
  };

  // ============================
  // Tool registration
  // ============================

  // Shared references for tools (populated when extraction scheduler creates them)
  let sharedVectorStore: VectorStore | undefined;
  let sharedEmbeddingService: EmbeddingService | undefined;

  /**
   * Whether the local embedding service warmup has been triggered at least once.
   * Tracked separately from schedulerStarted because warmup should also
   * be triggered from before_prompt_build (recall), not only agent_end.
   */
  let embeddingWarmupTriggered = false;

  /**
   * Trigger local embedding model warmup (download + load) on first use.
   * Safe to call multiple times — delegates idempotency to startWarmup() itself.
   *
   * IMPORTANT: If a previous warmup attempt FAILED (e.g. model download
   * network error), this will re-trigger startWarmup() so the service can
   * retry. startWarmup() internally checks its state machine:
   * - "ready" / "initializing" → no-op (already done or in progress)
   * - "idle" / "failed" → starts a new initialization attempt
   *
   * This avoids triggering model download during short-lived CLI commands
   * like `gateway stop` or `agents list` (warmup is still deferred until
   * the first real conversation).
   */
  const ensureEmbeddingWarmup = (): void => {
    if (!sharedEmbeddingService) return;

    if (!embeddingWarmupTriggered) {
      embeddingWarmupTriggered = true;
      api.logger.debug?.(`${TAG} Triggering lazy embedding warmup on first conversation`);
      sharedEmbeddingService.startWarmup();
      return;
    }

    // After first trigger: re-invoke startWarmup() only if the service
    // is not yet ready (covers the "failed" → retry path).
    // startWarmup() is idempotent for "ready" and "initializing" states.
    if (!sharedEmbeddingService.isReady()) {
      api.logger.debug?.(`${TAG} Embedding not ready, re-triggering warmup (retry)`);
      sharedEmbeddingService.startWarmup();
    }
  };

  // tdai_memory_search — Agent-callable L1 memory search tool
  api.registerTool(
    {
      name: "tdai_memory_search",
      label: "Memory Search",
      description:
        "Search through the user's long-term memories. Use this when you need to recall specific information about the user's preferences, past events, instructions, or context from previous conversations. Returns relevant memory records ranked by relevance.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "Search query describing what you want to recall about the user",
          },
          limit: {
            type: "number",
            description: "Maximum number of results to return (default: 5, max: 20)",
          },
          type: {
            type: "string",
            enum: ["persona", "episodic", "instruction"],
            description: "Optional filter by memory type: persona (identity/preferences), episodic (events/activities), instruction (user rules/commands)",
          },
          scene: {
            type: "string",
            description: "Optional filter by scene name",
          },
        },
        required: ["query"],
      },
      async execute(_toolCallId: string, params: Record<string, unknown>) {
        const startMs = Date.now();
        const query = String(params.query ?? "");
        const limit = Math.min(Math.max(Number(params.limit) || 5, 1), 20);
        const typeFilter = typeof params.type === "string" ? params.type : undefined;
        const sceneFilter = typeof params.scene === "string" ? params.scene : undefined;

        api.logger.debug?.(
          `${TAG} [tool] tdai_memory_search called: ` +
          `query="${query.length > 80 ? query.slice(0, 80) + "…" : query}", ` +
          `limit=${limit}, type=${typeFilter ?? "(all)"}, scene=${sceneFilter ?? "(all)"}`,
        );

        try {
          const result = await executeMemorySearch({
            query,
            limit,
            type: typeFilter,
            scene: sceneFilter,
            vectorStore: sharedVectorStore,
            embeddingService: sharedEmbeddingService,
            logger: api.logger,
          });

          const elapsedMs = Date.now() - startMs;
          const responseText = formatSearchResponse(result);
          api.logger.debug?.(
            `${TAG} [tool] tdai_memory_search completed (${elapsedMs}ms): ` +
            `total=${result.total}, strategy=${result.strategy}, ` +
            `responseLength=${responseText.length} chars`,
          );
          return {
            content: [{ type: "text" as const, text: responseText }],
            details: { count: result.total, strategy: result.strategy },
          };
        } catch (err) {
          const elapsedMs = Date.now() - startMs;
          const errMsg = err instanceof Error ? err.message : String(err);
          api.logger.error(`${TAG} [tool] tdai_memory_search failed (${elapsedMs}ms): ${errMsg}`);
          return {
            content: [{ type: "text" as const, text: `Memory search failed: ${errMsg}` }],
            details: { error: errMsg },
          };
        }
      },
    },
    { name: "tdai_memory_search" },
  );

  // tdai_conversation_search — Agent-callable L0 conversation search tool
  api.registerTool(
    {
      name: "tdai_conversation_search",
      label: "Conversation Search",
      description:
        "Search through past conversation history (raw dialogue records). " +
        "Use this when tdai_memory_search (structured memories) doesn't have the information you need, " +
        "or when you want to find specific past conversations, dialogue context, or exact words " +
        "the user said before. Returns relevant individual messages ranked by relevance.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "Search query describing what conversation content you want to find",
          },
          limit: {
            type: "number",
            description: "Maximum number of messages to return (default: 5, max: 20)",
          },
          session_key: {
            type: "string",
            description: "Optional: filter results to a specific session",
          },
        },
        required: ["query"],
      },
      async execute(_toolCallId: string, params: Record<string, unknown>) {
        const startMs = Date.now();
        const query = String(params.query ?? "");
        const limit = Math.min(Math.max(Number(params.limit) || 5, 1), 20);
        const sessionKeyFilter = typeof params.session_key === "string" ? params.session_key : undefined;

        api.logger.debug?.(
          `${TAG} [tool] tdai_conversation_search called: ` +
          `query="${query.length > 80 ? query.slice(0, 80) + "…" : query}", ` +
          `limit=${limit}, session_key=${sessionKeyFilter ?? "(all)"}`,
        );

        try {
          const result = await executeConversationSearch({
            query,
            limit,
            sessionKey: sessionKeyFilter,
            vectorStore: sharedVectorStore,
            embeddingService: sharedEmbeddingService,
            logger: api.logger,
          });

          const elapsedMs = Date.now() - startMs;
          const responseText = formatConversationSearchResponse(result);
          api.logger.debug?.(
            `${TAG} [tool] tdai_conversation_search completed (${elapsedMs}ms): ` +
            `total=${result.total}, responseLength=${responseText.length} chars`,
          );
          return {
            content: [{ type: "text" as const, text: responseText }],
            details: { count: result.total },
          };
        } catch (err) {
          const elapsedMs = Date.now() - startMs;
          const errMsg = err instanceof Error ? err.message : String(err);
          api.logger.error(`${TAG} [tool] tdai_conversation_search failed (${elapsedMs}ms): ${errMsg}`);
          return {
            content: [{ type: "text" as const, text: `Conversation search failed: ${errMsg}` }],
            details: { error: errMsg },
          };
        }
      },
    },
    { name: "tdai_conversation_search" },
  );

  // ============================
  // Lifecycle hooks
  // ============================

  // Before prompt build: auto-recall relevant memories
  // (migrated from legacy before_agent_start to before_prompt_build so that
  //  event.messages is guaranteed to be available — session is already loaded)
  if (cfg.recall.enabled) {
    api.logger.info(`${TAG} Registering before_prompt_build hook (auto-recall)`);
    api.on("before_prompt_build", async (event, ctx) => {
      const startMs = Date.now();
      api.logger.debug?.(`${TAG} [before_prompt_build] Hook triggered`);

      const sessionKey = ctx.sessionKey;

      if (sessionFilter.shouldSkipCtx(ctx)) {
        api.logger.debug?.(`${TAG} [before_prompt_build] Skipping filtered session`);
        return;
      }

      // Trigger embedding warmup on first real conversation (lazy init).
      // This is the earliest point where a real user message arrives,
      // so we start the model download here rather than in register()
      // to avoid triggering it during short-lived CLI commands.
      ensureEmbeddingWarmup();

      // Cache original user prompt for agent_end
      const rawPrompt = event.prompt;
      const messages = Array.isArray(event.messages) ? event.messages : undefined;
      if (sessionKey && rawPrompt) {
        const messageCount = messages?.length ?? 0;
        pendingOriginalPrompts.set(sessionKey, { text: rawPrompt, ts: Date.now(), messageCount });
        api.logger.debug?.(`${TAG} [before_prompt_build] Cached original prompt (${rawPrompt.length} chars, msgCount=${messageCount})`);
      }
      sweepStalePromptCache();

      const userText = rawPrompt;
      api.logger.debug?.(`${TAG} [before_prompt_build] userText length: ${userText?.length}`);
      if (!userText) {
        api.logger.debug?.(`${TAG} [before_prompt_build] No user text found, skipping recall`);
        return;
      }

      const resolvedSessionKey = resolveSessionKey(sessionKey);
      if (!resolvedSessionKey) {
        return;
      }

      try {
        const result = await performAutoRecall({
          userText,
          actorId: ACTOR_ID,
          sessionKey: resolvedSessionKey,
          cfg,
          pluginDataDir,
          logger: api.logger,
          vectorStore: sharedVectorStore,
          embeddingService: sharedEmbeddingService,
        });
        const elapsedMs = Date.now() - startMs;
        if (result?.appendSystemContext) {
          api.logger.info(
            `${TAG} [before_prompt_build] Recall complete (${elapsedMs}ms), ` +
            `appendSystemContext=${result.appendSystemContext.length} chars`,
          );
        } else {
          api.logger.info(`${TAG} [before_prompt_build] Recall complete (${elapsedMs}ms), no context to inject`);
        }
        return result;
      } catch (err) {
        const elapsedMs = Date.now() - startMs;
        api.logger.error(`${TAG} [before_prompt_build] Auto-recall failed after ${elapsedMs}ms: ${err instanceof Error ? err.stack ?? err.message : String(err)}`);
      }
    });
  }

  // After agent end: auto-capture + L0 record + L1/L2/L3 schedule
  if (cfg.capture.enabled) {
    // ============================
    // Create the MemoryPipelineManager (L1→L2→L3 architecture)
    // ============================
    let scheduler: MemoryPipelineManager | undefined;

    // ============================
    // Lazy scheduler startup (Solution C):
    // Defer scheduler.start() until the first agent_end event. This way,
    // short-lived CLI management commands (agents add/list/delete, etc.)
    // never start the scheduler, never recover pending sessions, and
    // therefore never trigger the L1→L2→L3 flush chain on destroy().
    // ============================
    let schedulerStarted = false;

    /**
     * Lazily start the scheduler on first conversation.
     * Reads checkpoint, restores session states, and pre-warms the
     * embedded agent. Subsequent calls are no-ops.
     * No-op when scheduler is undefined (extraction disabled).
     */
    const ensureSchedulerStarted = async (): Promise<void> => {
      if (schedulerStarted || !scheduler) return;
      schedulerStarted = true;

      // Trigger embedding warmup alongside scheduler start — both are
      // deferred until the first real conversation to avoid downloading
      // models during short-lived CLI commands.
      ensureEmbeddingWarmup();

      try {
        const initCheckpoint = new CheckpointManager(pluginDataDir, api.logger);
        const cp = await initCheckpoint.read();
        scheduler.start(initCheckpoint.getAllPipelineStates(cp));
        api.logger.info(
          `${TAG} Scheduler lazy-started on first agent_end ` +
          `(everyN=${cfg.pipeline.everyNConversations}, ` +
          `l1Idle=${cfg.pipeline.l1IdleTimeoutSeconds}s, ` +
          `l2DelayAfterL1=${cfg.pipeline.l2DelayAfterL1Seconds}s, ` +
          `l2MinInterval=${cfg.pipeline.l2MinIntervalSeconds}s, ` +
          `l2MaxInterval=${cfg.pipeline.l2MaxIntervalSeconds}s, ` +
          `sessionActiveWindow=${cfg.pipeline.sessionActiveWindowHours}h)`,
        );
      } catch (err) {
        api.logger.error(
          `${TAG} Failed to restore checkpoint for scheduler: ${err instanceof Error ? err.message : String(err)}`,
        );
        // Start with empty state as fallback
        scheduler.start({});
      }

      // Pre-warm the embedded agent import so the first extraction run doesn't
      // pay the cold-start cost (~35s jiti compile → <50ms with dist/ path).
      prewarmEmbeddedAgent(api.logger);
    };

    if (cfg.extraction.enabled) {
      // === Initialize VectorStore (always) + EmbeddingService (only when embedding enabled) ===
      let vectorStore: VectorStore | undefined;
      let embeddingService: EmbeddingService | undefined;

      // VectorStore is always created as the metadata store for L0/L1 records.
      // It works as a pure SQLite store even without embedding — keyword search,
      // L0/L1 reads, and pipeline queries all use structured SQL, not vectors.
      try {
        const dims = cfg.embedding.dimensions; // 0 when provider="none" → vec0 tables deferred
        const dbPath = path.join(pluginDataDir, "vectors.db");
        vectorStore = new VectorStore(dbPath, dims, api.logger);

        // Create EmbeddingService only when embedding is enabled (remote provider configured)
        if (cfg.embedding.enabled) {
          try {
            if (cfg.embedding.provider !== "local" && cfg.embedding.apiKey) {
              // Remote embedding provider (OpenAI-compatible API: OpenAI, Azure, self-hosted, etc.)
              embeddingService = createEmbeddingService({
                provider: cfg.embedding.provider,
                baseUrl: cfg.embedding.baseUrl,
                apiKey: cfg.embedding.apiKey,
                model: cfg.embedding.model,
                dimensions: cfg.embedding.dimensions,
              }, api.logger);
            } else {
              // Local provider (node-llama-cpp) — preserved internally but not reachable from user config
              embeddingService = createEmbeddingService({
                provider: "local",
                modelPath: cfg.embedding.model || undefined,
                modelCacheDir: cfg.embedding.modelCacheDir,
              }, api.logger);
            }
          } catch (err) {
            api.logger.warn(
              `${TAG} EmbeddingService init failed, continuing with keyword-only mode: ${err instanceof Error ? err.message : String(err)}`,
            );
            embeddingService = undefined;
          }
        } else {
          api.logger.info(`${TAG} Embedding disabled by config, VectorStore will serve as metadata-only store`);
        }

        // Init VectorStore with provider info (undefined when no embedding → skips provider change detection)
        const providerInfo = embeddingService?.getProviderInfo();
        const initResult = vectorStore.init(providerInfo);

        // If VectorStore entered degraded mode (e.g. sqlite-vec load failed),
        // treat it as unavailable and fall back to keyword-only mode.
        if (vectorStore.isDegraded()) {
          api.logger.warn(
            `${TAG} VectorStore is in degraded mode, falling back to keyword dedup`,
          );
          vectorStore = undefined;
          embeddingService = undefined;
        } else {
          api.logger.info(
            `${TAG} VectorStore initialized: ${dbPath} (${dims}D, provider=${cfg.embedding.provider})`,
          );

          // If embedding provider/model/dimensions changed, re-embed all existing texts
          if (initResult.needsReindex && embeddingService) {
            const svc = embeddingService; // capture for async closure
            const vs = vectorStore;       // capture for async closure
            api.logger.info(
              `${TAG} Embedding config changed (${initResult.reason}). ` +
              `Starting background re-embed of all stored texts...`,
            );
            // Run re-embed asynchronously so it doesn't block plugin startup
            vs.reindexAll(
              (text) => svc.embed(text),
              (done, total, layer) => {
                if (done === total || done % 50 === 0) {
                  api.logger.debug?.(`${TAG} Re-embed progress: ${layer} ${done}/${total}`);
                }
              },
            ).then(({ l1Count, l0Count }) => {
              api.logger.info(
                `${TAG} Re-embed complete: L1=${l1Count} records, L0=${l0Count} messages`,
              );
            }).catch((err) => {
              api.logger.error(
                `${TAG} Re-embed failed (non-fatal): ${err instanceof Error ? err.message : String(err)}`,
              );
            });
          }
        }
      } catch (err) {
        api.logger.warn(
          `${TAG} VectorStore init failed; vector/FTS recall and dedup conflict detection will be unavailable: ${err instanceof Error ? err.message : String(err)}`,
        );
        vectorStore = undefined;
        embeddingService = undefined;
      }

      // Share vectorStore/embeddingService with tdai_memory_search tool
      sharedVectorStore = vectorStore;
      sharedEmbeddingService = embeddingService;

      // Keep cleaner's SQLite handle updated (singleton cleaner may start earlier).
      memoryCleaner?.setVectorStore(vectorStore);

      // === Create pipeline manager ===
      scheduler = new MemoryPipelineManager(
        {
          everyNConversations: cfg.pipeline.everyNConversations,
          enableWarmup: cfg.pipeline.enableWarmup,
          l1: { idleTimeoutSeconds: cfg.pipeline.l1IdleTimeoutSeconds },
          l2: {
            delayAfterL1Seconds: cfg.pipeline.l2DelayAfterL1Seconds,
            minIntervalSeconds: cfg.pipeline.l2MinIntervalSeconds,
            maxIntervalSeconds: cfg.pipeline.l2MaxIntervalSeconds,
            sessionActiveWindowHours: cfg.pipeline.sessionActiveWindowHours,
          },
        },
        api.logger,
        sessionFilter,
      );

      // L1 runner: read L0 from DB (primary) or JSONL (fallback) → local LLM extraction → L1 JSONL + VectorStore
      scheduler.setL1Runner(async ({ sessionKey }) => {
      // L1 reads L0 data from VectorStore DB (primary, indexed query).
      // Fallback: read from L0 JSONL files when VectorStore is unavailable.
        if (!api.config) {
          api.logger.debug?.(`${TAG} [pipeline-l1] No OpenClaw config, skipping L1 extraction`);
          return { processedCount: 0 };
        }

        const checkpoint = new CheckpointManager(pluginDataDir, api.logger);
        const cp = await checkpoint.read();
        const runnerState = checkpoint.getRunnerState(cp, sessionKey);

        api.logger.info(
          `${TAG} [pipeline-l1] Session ${sessionKey}: ` +
          `l1_cursor=${runnerState.last_l1_cursor || "(start)"}`,
        );

        try {
          // Read L0 messages since last L1 cursor, grouped by sessionId.
          // Within the same sessionKey, different sessionIds represent different
          // conversation instances (e.g. after /reset). Each group is extracted
          // independently so its sessionId is correctly associated with L1 records.
          //
          // Primary path: read from VectorStore DB (indexed query, fast).
          // Fallback: read from L0 JSONL files (scan + parse, slower).
          let groups: Array<{ sessionId: string; messages: ConversationMessage[] }>;

          if (vectorStore && !vectorStore.isDegraded()) {
            // DB path: fast indexed query
            // NOTE: When last_l1_cursor is 0 (first L1 run), we pass undefined
            // to query all messages — but only those captured AFTER plugin start
            // (L0 capture uses pluginStartTimestamp as floor, so DB won't contain
            // pre-existing messages). This is safe because auto-capture already
            // filters out messages older than pluginStartTimestamp.
            const l1Cursor = runnerState.last_l1_cursor > 0
              ? runnerState.last_l1_cursor
              : undefined;
            const dbGroups = vectorStore.queryL0GroupedBySessionId(
              sessionKey,
              l1Cursor,
            );
            // Cast role from string to "user" | "assistant" (DB stores as string)
            groups = dbGroups.map((g) => ({
              sessionId: g.sessionId,
              messages: g.messages.map((m) => ({
                id: m.id,
                role: m.role as "user" | "assistant",
                content: m.content,
                timestamp: m.timestamp,
              })),
            }));
            api.logger.debug?.(
              `${TAG} [pipeline-l1] L0 data source: VectorStore DB`,
            );
          } else {
            // Fallback: JSONL files
            api.logger.debug?.(
              `${TAG} [pipeline-l1] L0 data source: JSONL files (VectorStore unavailable)`,
            );
            const jsonlGroups = await readConversationMessagesGroupedBySessionId(
              sessionKey,
              pluginDataDir,
              runnerState.last_l1_cursor || undefined,
              api.logger,
              50, // Match DB path limit (queryL0ForL1 default)
            );
            // Convert SessionIdMessageGroup[] to the same shape
            groups = jsonlGroups.map((g) => ({
              sessionId: g.sessionId,
              messages: g.messages,
            }));
          }

          if (groups.length === 0) {
            api.logger.debug?.(`${TAG} [pipeline-l1] No new L0 messages for session ${sessionKey}`);
            return { processedCount: 0 };
          }

          const totalMessages = groups.reduce((sum, g) => sum + g.messages.length, 0);
          api.logger.info(
            `${TAG} [pipeline-l1] Processing ${totalMessages} L0 messages across ${groups.length} sessionId group(s) for session ${sessionKey}`,
          );

          let totalExtracted = 0;
          let totalStored = 0;
          let lastSceneName: string | undefined;
          let maxTimestamp = 0;

          for (const group of groups) {
            api.logger.debug?.(
              `${TAG} [pipeline-l1] Group sessionId=${group.sessionId || "(empty)"}: ${group.messages.length} messages`,
            );

            const l1Result = await extractL1Memories({
              messages: group.messages,
              sessionKey,
              sessionId: group.sessionId,
              baseDir: pluginDataDir,
              config: api.config,
              options: {
                enableDedup: cfg.extraction.enableDedup,
                maxMemoriesPerSession: cfg.extraction.maxMemoriesPerSession,
                model: cfg.extraction.model,
                previousSceneName: lastSceneName ?? (runnerState.last_scene_name || undefined),
                vectorStore,
                embeddingService,
                conflictRecallTopK: cfg.embedding.conflictRecallTopK,
              },
              logger: api.logger,
            });

            totalExtracted += l1Result.extractedCount;
            totalStored += l1Result.storedCount;
            if (l1Result.lastSceneName) {
              lastSceneName = l1Result.lastSceneName;
            }

            const groupMaxTs = Math.max(...group.messages.map((m) => m.timestamp));
            maxTimestamp = Math.max(maxTimestamp, groupMaxTs);
          }

          // Update checkpoint on disk — cursor is the global max timestamp across all groups
          await checkpoint.markL1ExtractionComplete(
            sessionKey,
            totalStored,
            maxTimestamp,
            lastSceneName,
          );

          api.logger.info(
            `${TAG} [pipeline-l1] L1 complete: extracted=${totalExtracted}, stored=${totalStored} (${groups.length} group(s))`,
          );

          return { processedCount: totalMessages };
        } catch (err) {
          api.logger.error(`${TAG} [pipeline-l1] L1 failed: ${err instanceof Error ? err.stack ?? err.message : String(err)}`);
          throw err; // rethrow so pipeline-manager can retry
        }
      });

      // Persister: saves pipeline session states to checkpoint
      scheduler.setPersister(async (states) => {
        const checkpoint = new CheckpointManager(pluginDataDir, api.logger);
        await checkpoint.mergePipelineStates(states);
      });

      // L2 runner: read L1 records (incremental) → SceneExtractor
      scheduler.setL2Runner(async (sessionKey: string, cursor?: string) => {
        try {
          return await runLocalL2Extraction(api, cfg, pluginDataDir, sessionKey, vectorStore, cursor);
        } catch (err) {
          api.logger.error(`${TAG} [pipeline-l2] L2 failed: ${err instanceof Error ? err.stack ?? err.message : String(err)}`);
          throw err; // rethrow so pipeline-manager can handle retry/fallback (consistent with L1 runner)
        }
      });

      // L3 runner: persona trigger + generation
      scheduler.setL3Runner(async () => {
        try {
          const trigger = new PersonaTrigger({
            dataDir: pluginDataDir,
            interval: cfg.persona.triggerEveryN,
            logger: api.logger,
          });

          const { should, reason } = await trigger.shouldGenerate();
          if (!should) {
            api.logger.debug?.(`${TAG} [pipeline-l3] Persona generation not needed`);
            return;
          }

          if (!api.config) {
            api.logger.warn(`${TAG} [pipeline-l3] No OpenClaw config, skipping persona generation`);
            return;
          }

          api.logger.info(`${TAG} [pipeline-l3] Starting persona generation: ${reason}`);
          const generator = new PersonaGenerator({
            dataDir: pluginDataDir,
            config: api.config,
            model: cfg.persona.model,
            backupCount: cfg.persona.backupCount,
            logger: api.logger,
          });
          const genResult = await generator.generate(reason);
          api.logger.info(`${TAG} [pipeline-l3] Persona generation ${genResult ? "succeeded" : "skipped (no changes)"}`);
        } catch (err) {
          api.logger.error(`${TAG} [pipeline-l3] Failed: ${err instanceof Error ? err.stack ?? err.message : String(err)}`);
        }
      });

      // Capture vectorStore reference for cleanup
      const vectorStoreRef = vectorStore;

      // Register a SINGLE gateway_stop hook for ordered shutdown.
      // Order: memoryCleaner → scheduler → vectorStore
      // (memoryCleaner may use VectorStore during cleanup, so it must stop first)
      api.on("gateway_stop", async () => {
        // 1. Stop the memory cleaner first (it may be running deleteL1ExpiredByUpdatedTime)
        if (memoryCleaner) {
          try {
            memoryCleaner.destroy();
            if (sharedMemoryCleaner === memoryCleaner) {
              sharedMemoryCleaner = undefined;
            }
            api.logger.info(`${TAG} [gateway_stop] Memory cleaner destroyed`);
          } catch (error) {
            api.logger.error(`${TAG} [gateway_stop] Error during memory cleaner destruction: ${error instanceof Error ? error.message : String(error)}`);
          }
        }

        // 2. Destroy scheduler (flushes pending L1/L2/L3 work)
        if (scheduler && schedulerStarted) {
          api.logger.info(`${TAG} [gateway_stop] Destroying scheduler...`);
          await scheduler.destroy();
          api.logger.info(`${TAG} [gateway_stop] Scheduler destroyed`);
        } else {
          api.logger.info(`${TAG} [gateway_stop] Scheduler was never started, skipping destroy`);
        }

        // 3. Close VectorStore last (after all consumers are done)
        if (vectorStoreRef) {
          api.logger.info(`${TAG} [gateway_stop] Closing VectorStore`);
          vectorStoreRef.close();
        }

        // 4. Release embedding service resources (model memory, GPU, etc.)
        if (embeddingService?.close) {
          try {
            api.logger.info(`${TAG} [gateway_stop] Closing EmbeddingService`);
            await embeddingService.close();
          } catch (err) {
            api.logger.warn(`${TAG} [gateway_stop] Error closing EmbeddingService: ${err instanceof Error ? err.message : String(err)}`);
          }
        }
      });
    }

    api.logger.info(`${TAG} Registering agent_end hook (auto-capture)`);
    api.on("agent_end", async (event, ctx) => {
      const startMs = Date.now();
      api.logger.debug?.(`${TAG} [agent_end] Hook triggered`);

      const e = event as Record<string, unknown>;
      if (!e.success) {
        api.logger.info(`${TAG} [agent_end] Agent did not succeed, skipping capture`);
        return;
      }

      const sessionKey = ctx.sessionKey;
      const sessionId = ctx.sessionId;

      if (sessionFilter.shouldSkipCtx(ctx)) {
        api.logger.debug?.(`${TAG} [agent_end] Skipping filtered session`);
        return;
      }

      const messages = (e.messages as unknown[]) ?? [];
      const resolvedSessionKey = resolveSessionKey(sessionKey);
      if (!resolvedSessionKey) {
        return;
      }

      // Retrieve cached original prompt (don't delete — retry may trigger multiple agent_end;
      // stale entries are swept by TTL in before_prompt_build)
      const cachedPrompt = sessionKey ? pendingOriginalPrompts.get(sessionKey) : undefined;
      const originalUserText = cachedPrompt?.text;
      const originalUserMessageCount = cachedPrompt?.messageCount;

      try {
        // Lazy-start the scheduler on first real conversation (Solution C).
        // This is a no-op after the first call.
        await ensureSchedulerStarted();

        const captureResult = await performAutoCapture({
          messages,
          sessionKey: resolvedSessionKey,
          sessionId: sessionId || undefined,
          cfg,
          pluginDataDir,
          logger: api.logger,
          scheduler,
          originalUserText,
          originalUserMessageCount,
          pluginStartTimestamp,
          vectorStore: sharedVectorStore,
          embeddingService: sharedEmbeddingService,
        });
        const captureMs = Date.now() - startMs;
        api.logger.info(
          `${TAG} [agent_end] Auto-capture complete (${captureMs}ms), ` +
          `l0Recorded=${captureResult.l0RecordedCount}, ` +
          `schedulerNotified=${captureResult.schedulerNotified}`,
        );
      } catch (err) {
        const elapsedMs = Date.now() - startMs;
        api.logger.error(`${TAG} [agent_end] Auto-capture failed after ${elapsedMs}ms: ${err instanceof Error ? err.stack ?? err.message : String(err)}`);
      }
    });
  } else {
    api.logger.info(`${TAG} Auto-capture disabled`);
  }

  // memoryCleaner gateway_stop is handled in the unified handler above (inside extraction.enabled block).
  // For the case where capture is enabled but extraction is disabled, register cleanup separately.
  if (memoryCleaner && !cfg.extraction.enabled) {
    api.on("gateway_stop", async () => {
      try {
        memoryCleaner?.destroy();
        if (sharedMemoryCleaner === memoryCleaner) {
          sharedMemoryCleaner = undefined;
        }
      } catch (error) {
        api.logger.error(`${TAG} [gateway_stop] Error during memory cleaner destruction: ${error instanceof Error ? error.message : String(error)}`);
      }
    });
  }

  api.logger.info(
    `${TAG} Plugin registration complete (v3). ` +
    `startTimestamp=${pluginStartTimestamp} (${new Date(pluginStartTimestamp).toISOString()})`,
  );
}

// ============================
// L2 extraction implementations
// ============================

/**
 * Local L2 extraction: read L1 records → SceneExtractor.
 *
 * Uses **incremental** reads when VectorStore is available:
 *   1. Receive the pipeline cursor (`last_extraction_updated_time`) from pipeline-manager
 *   2. Query only L1 records updated AFTER that cursor via `queryMemoryRecords`
 *   3. Return the latest `updatedAt` from the batch so pipeline-manager can advance the cursor
 *
 * Falls back to JSONL read (with client-side time filtering) when VectorStore is unavailable.
 */
async function runLocalL2Extraction(
  api: OpenClawPluginApi,
  cfg: MemoryTdaiConfig,
  pluginDataDir: string,
  sessionKey: string,
  vectorStore?: VectorStore,
  updatedAfter?: string,
): Promise<{ latestCursor?: string } | void> {
  api.logger.debug?.(
    `${TAG} [L2-local] session=${sessionKey}, updatedAfter=${updatedAfter ?? "(full)"}`,
  );

  let records: Array<{ content: string; created_at: string; id: string; updatedAt: string }>;

  // Prefer incremental SQLite query when VectorStore is available
  if (vectorStore && !vectorStore.isDegraded()) {
    const { queryMemoryRecords } = await import("./src/record/l1-reader.js");
    const memRecords = queryMemoryRecords(vectorStore, {
      sessionKey,
      updatedAfter,
    }, api.logger);

    if (memRecords.length === 0) {
      api.logger.debug?.(
        `${TAG} [L2-local] No new L1 records since cursor (session=${sessionKey}, updatedAfter=${updatedAfter ?? "(full)"}), skipping scene extraction`,
      );
      return;
    }

    api.logger.debug?.(
      `${TAG} [L2-local] Incremental query returned ${memRecords.length} record(s) (session=${sessionKey})`,
    );

    records = memRecords.map((r) => ({
      content: r.content,
      created_at: r.createdAt,
      id: r.id,
      updatedAt: r.updatedAt,
    }));
  } else {
    // Fallback: read JSONL files with client-side time filtering
    api.logger.debug?.(`${TAG} [L2-local] VectorStore unavailable, falling back to JSONL read`);
    const { readAllMemoryRecords } = await import("./src/record/l1-reader.js");
    let allRecords = await readAllMemoryRecords(pluginDataDir, api.logger);

    // Apply updatedAfter filter on JSONL records (same semantics as SQLite path)
    if (updatedAfter) {
      const beforeCount = allRecords.length;
      allRecords = allRecords.filter((r) => {
        const t = r.updatedAt || r.createdAt || "";
        return t > updatedAfter;
      });
      api.logger.debug?.(
        `${TAG} [L2-local] JSONL time filter: ${beforeCount} → ${allRecords.length} record(s) (updatedAfter=${updatedAfter})`,
      );
    }

    if (allRecords.length === 0) {
      api.logger.debug?.(`${TAG} [L2-local] No new L1 records found (JSONL fallback), skipping scene extraction`);
      return;
    }

    records = allRecords.map((r) => ({
      content: r.content,
      created_at: r.createdAt,
      id: r.id,
      updatedAt: r.updatedAt,
    }));
  }

  const extractor = new SceneExtractor({
    dataDir: pluginDataDir,
    config: api.config!,
    model: cfg.persona.model,
    maxScenes: cfg.persona.maxScenes,
    sceneBackupCount: cfg.persona.sceneBackupCount,
    logger: api.logger,
  });

  const memories = records.map((r) => ({
    content: r.content,
    created_at: r.created_at,
    id: r.id,
  }));

  // ── Checkpoint guard ──────────────────────────────────────────────
  // Snapshot critical counters BEFORE the LLM agent runs.
  // The LLM operates on checkpoint via raw file tools (write_to_file /
  // replace_in_file) which bypass CheckpointManager's file lock.
  // If the LLM accidentally overwrites the entire checkpoint (e.g. via
  // write_to_file), system-managed counters like scenes_processed and
  // memories_since_last_persona can be reset to stale values.
  // After extraction we detect and repair such corruption.
  const preCheckpoint = new CheckpointManager(pluginDataDir, api.logger);
  const preState = await preCheckpoint.read();
  const preScenesProcessed = preState.scenes_processed;
  const preMemoriesSince = preState.memories_since_last_persona;
  const preTotalProcessed = preState.total_processed;

  const extractResult = await extractor.extract(memories);
  if (extractResult.success && extractResult.memoriesProcessed > 0) {
    const checkpoint = new CheckpointManager(pluginDataDir, api.logger);

    // Detect and repair LLM-caused checkpoint corruption.
    // If the LLM wrote the entire checkpoint file (instead of using
    // replace_in_file on specific fields), system-managed counters may
    // have been overwritten with stale/zero values.
    const postState = await checkpoint.read();
    if (
      postState.scenes_processed < preScenesProcessed ||
      postState.total_processed < preTotalProcessed
    ) {
      api.logger.warn(
        `${TAG} [L2-local] ⚠️ Checkpoint corruption detected! ` +
        `scenes_processed: ${preScenesProcessed} → ${postState.scenes_processed}, ` +
        `total_processed: ${preTotalProcessed} → ${postState.total_processed}, ` +
        `memories_since: ${preMemoriesSince} → ${postState.memories_since_last_persona}. ` +
        `Repairing...`,
      );
      await checkpoint.write({
        ...postState,
        scenes_processed: Math.max(postState.scenes_processed, preScenesProcessed),
        total_processed: Math.max(postState.total_processed, preTotalProcessed),
        memories_since_last_persona: Math.max(postState.memories_since_last_persona, preMemoriesSince),
      });
      api.logger.info(`${TAG} [L2-local] Checkpoint repaired`);
    }

    await checkpoint.incrementScenesProcessed();

    // Return the max updatedAt from this batch as the new cursor
    const latestCursor = records.reduce((latest, r) => {
      return r.updatedAt > latest ? r.updatedAt : latest;
    }, "");

    api.logger.debug?.(
      `${TAG} [L2-local] Extraction complete: processed=${extractResult.memoriesProcessed}, latestCursor=${latestCursor}`,
    );

    return { latestCursor: latestCursor || undefined };
  }
}

// ============================
// Helpers
// ============================

