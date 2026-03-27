/**
 * auto-recall hook (v3): injects relevant memories + persona into agent context
 * before the agent starts processing.
 *
 * - Searches L1 memories using configurable strategy (keyword / embedding / hybrid)
 *   - keyword: FTS5 BM25 (requires FTS5; returns empty if unavailable)
 *   - embedding: VectorStore cosine similarity
 *   - hybrid: keyword + embedding merged with RRF
 * - L3 persona injection
 * - L2 scene navigation (full injection, LLM decides relevance)
 */

import fs from "node:fs/promises";
import path from "node:path";
import type { MemoryTdaiConfig } from "../config.js";
import { readSceneIndex } from "../scene/scene-index.js";
import { generateSceneNavigation, stripSceneNavigation } from "../scene/scene-navigation.js";
import type { MemoryRecord } from "../record/l1-reader.js";
import type { VectorStore, VectorSearchResult, FtsSearchResult } from "../store/vector-store.js";
import { buildFtsQuery } from "../store/vector-store.js";
import type { EmbeddingService } from "../store/embedding.js";
import { sanitizeText } from "../utils/sanitize.js";

const TAG = "[memory-tdai] [recall]";

/**
 * Memory tools usage guide — injected at the end of memory context so the
 * main agent knows how to actively retrieve deeper information.
 */
const MEMORY_TOOLS_GUIDE = `<memory-tools-guide>
## 记忆工具调用指南

当上方注入的记忆片段不足以回答用户问题时，可主动调用以下工具获取更多信息：

- **tdai_memory_search**：搜索结构化记忆（L1），适用于回忆用户偏好、历史事件节点、规则等关键信息。
- **tdai_conversation_search**：搜索原始对话（L0），适用于查找具体消息原文、时间线、上下文细节；也可用于补充或校验 memory_search 的结果。
- **read_file**（Scene Navigation 中的路径）：当已定位到相关情境，且需要该场景的完整画像、事件经过或阶段结论时使用。
</memory-tools-guide>`

/**
 * Build the dynamic scene-navigation read_file hint.
 * Tells the agent how to resolve relative paths in scene navigation
 * by prepending the actual pluginDataDir.
 */
function buildScenePathHint(pluginDataDir: string): string {
  return `⚠️ Scene Navigation 路径提示：上方 Scene Navigation 中的 Path（如 \`scene_blocks/xxx.md\`）是相对路径，使用 read_file 读取时需拼接为绝对路径：\`${pluginDataDir}/scene_blocks/xxx.md\``;
}

interface Logger {
  debug?: (message: string) => void;
  info: (message: string) => void;
  warn: (message: string) => void;
  error: (message: string) => void;
}

export interface RecallResult {
  /** Injected before user message (prepended to the user's prompt text by openclaw) */
  prependContext?: string;
  /** Appended to system prompt (all memory context: persona, scene navigation, relevant memories) */
  appendSystemContext?: string;
}

export async function performAutoRecall(params: {
  userText: string;
  actorId: string;
  sessionKey: string;
  cfg: MemoryTdaiConfig;
  pluginDataDir: string;
  logger?: Logger;
  vectorStore?: VectorStore;
  embeddingService?: EmbeddingService;
}): Promise<RecallResult | undefined> {
  const { userText, cfg, pluginDataDir, logger, vectorStore, embeddingService } = params;

  // Search relevant memories (L1 layer) — skip only when userText is empty/undefined
  let memoryLines: string[] = [];
  if (!userText || userText.length === 0) {
    logger?.debug?.(`${TAG} User text empty/undefined, skipping memory search (persona/scene still injected)`);
  } else {
    const strategy = cfg.recall.strategy ?? "hybrid";
    memoryLines = await searchMemories(userText, pluginDataDir, cfg, logger, strategy, vectorStore, embeddingService);
  }

  // Read persona (L3 layer)
  let personaContent: string | undefined;
  try {
    const personaPath = path.join(pluginDataDir, "persona.md");
    const raw = await fs.readFile(personaPath, "utf-8");
    personaContent = stripSceneNavigation(raw).trim();
    if (!personaContent) personaContent = undefined;
    logger?.debug?.(`${TAG} Persona loaded: ${personaContent ? `${personaContent.length} chars` : "empty"}`);
  } catch {
    logger?.debug?.(`${TAG} No persona file found (expected for new users)`);
  }

  // Load full scene navigation (L2 layer)
  let sceneNavigation: string | undefined;
  try {
    const sceneIndex = await readSceneIndex(pluginDataDir);
    if (sceneIndex.length > 0) {
      sceneNavigation = generateSceneNavigation(sceneIndex);
      logger?.debug?.(`${TAG} Scene navigation generated: ${sceneIndex.length} scenes`);
    }
  } catch {
    logger?.debug?.(`${TAG} No scene index found`);
  }

  if (memoryLines.length === 0 && !personaContent && !sceneNavigation) {
    logger?.debug?.(`${TAG} No memories/persona/scenes to inject`);
    return undefined;
  }

  // All memory context → appendSystemContext (system prompt end)
  // Order: user-persona → scene-navigation → relevant-memories → tools-guide
  const systemParts: string[] = [];
  if (personaContent) {
    systemParts.push(`<user-persona>\n${personaContent}\n</user-persona>`);
  }
  if (sceneNavigation) {
    const pathHint = buildScenePathHint(pluginDataDir);
    systemParts.push(`<scene-navigation>\n${sceneNavigation}\n\n${pathHint}\n</scene-navigation>`);
  }
  if (memoryLines.length > 0) {
    systemParts.push(`<relevant-memories>\n${memoryLines.join("\n")}\n</relevant-memories>`);
  }

  // Append memory tools usage guide so the agent knows how to actively
  // retrieve deeper context when the injected snippets are not enough.
  if (systemParts.length > 0) {
    systemParts.push(MEMORY_TOOLS_GUIDE);
  }

  const appendSystemContext = systemParts.length > 0 ? systemParts.join("\n\n") : undefined;

  if (!appendSystemContext) {
    return undefined;
  }

  return { appendSystemContext };
}

// ============================
// Multi-strategy search dispatcher
// ============================

interface ScoredRecord {
  record: MemoryRecord;
  score: number;
}

/**
 * Search memories using the configured strategy.
 *
 * - "keyword": JSONL keyword-based (Jaccard similarity) — no embedding needed
 * - "embedding": VectorStore cosine similarity — requires vectorStore + embeddingService
 * - "hybrid": merge both keyword and embedding results with RRF (Reciprocal Rank Fusion)
 *
 * Falls back to keyword if embedding resources are unavailable.
 */
async function searchMemories(
  userText: string,
  pluginDataDir: string,
  cfg: MemoryTdaiConfig,
  logger: Logger | undefined,
  strategy: "keyword" | "embedding" | "hybrid",
  vectorStore?: VectorStore,
  embeddingService?: EmbeddingService,
): Promise<string[]> {
  // Strip gateway-injected inbound metadata (Sender, timestamps, media markers,
  // base64 image data, etc.) so FTS / embedding queries are based on pure user intent.
  const cleanText = sanitizeText(userText);

  if (cleanText.length < 2) {
    logger?.debug?.(`${TAG} Query too short for memory search (raw=${userText.length}, clean=${cleanText.length})`);
    return [];
  }

  if (cleanText.length !== userText.length) {
    logger?.debug?.(
      `${TAG} userText sanitized: ${userText.length} → ${cleanText.length} chars`,
    );
  }

  const maxResults = cfg.recall.maxResults ?? 5;
  const threshold = cfg.recall.scoreThreshold ?? 0.1;

  const embeddingAvailable = !!vectorStore && !!embeddingService;

  logger?.debug?.(
    `${TAG} [searchMemories] strategy=${strategy}, embeddingAvailable=${embeddingAvailable}, ` +
    `vectorStore=${vectorStore ? "available" : "UNAVAILABLE"}, ` +
    `embeddingService=${embeddingService ? "available" : "UNAVAILABLE"}, ` +
    `maxResults=${maxResults}, threshold=${threshold}`,
  );

  // Determine effective strategy (fall back to keyword if embedding not available)
  let effectiveStrategy = strategy;
  if ((strategy === "embedding" || strategy === "hybrid") && !embeddingAvailable) {
    logger?.warn?.(
      `${TAG} Strategy "${strategy}" requested but EmbeddingService not available, falling back to keyword`,
    );
    effectiveStrategy = "keyword";
  }

  logger?.debug?.(`${TAG} Search strategy: ${effectiveStrategy} (configured: ${strategy})`);

  try {
    if (effectiveStrategy === "keyword") {
      return await searchByKeyword(cleanText, pluginDataDir, maxResults, threshold, logger, vectorStore);
    }

    if (effectiveStrategy === "embedding") {
      return await searchByEmbedding(cleanText, maxResults, threshold, vectorStore!, embeddingService!, logger);
    }

    // Hybrid: run both keyword and embedding, merge with RRF
    return await searchHybrid(cleanText, pluginDataDir, maxResults, threshold, vectorStore!, embeddingService!, logger);
  } catch (err) {
    logger?.warn?.(`${TAG} Memory search failed (strategy=${effectiveStrategy}): ${err instanceof Error ? err.message : String(err)}`);
    return [];
  }
}

// ============================
// Strategy: Keyword (FTS5 BM25, no in-memory fallback)
// ============================

async function searchByKeyword(
  userText: string,
  _pluginDataDir: string,
  maxResults: number,
  threshold: number,
  logger?: Logger,
  vectorStore?: VectorStore,
): Promise<string[]> {
  // Prefer FTS5 if available
  if (vectorStore?.isFtsAvailable()) {
    const ftsQuery = buildFtsQuery(userText);
    if (ftsQuery) {
      logger?.debug?.(`${TAG} [keyword-fts] Using FTS5 BM25 search: query="${ftsQuery}"`);
      const ftsResults = vectorStore.ftsSearchL1(ftsQuery, maxResults * 2);
      if (ftsResults.length > 0) {
        logger?.debug?.(
          `${TAG} [keyword-fts] FTS5 raw results (${ftsResults.length}): ` +
          ftsResults.map((r) => `id=${r.record_id} score=${r.score.toFixed(6)}`).join(", "),
        );
        const filtered = ftsResults
          .filter((r) => r.score >= threshold)
          .slice(0, maxResults);

        if (filtered.length > 0) {
          logger?.debug?.(`${TAG} [keyword-fts] FTS5 found ${filtered.length} results (from ${ftsResults.length} raw, threshold=${threshold})`);
          return filtered.map((r) => formatMemoryLine(ftsResultToFormatable(r)));
        }

        // BM25 absolute scores are unreliable when the document set is very
        // small (e.g. 1–3 records) because IDF approaches 0.  In that case,
        // trust FTS5's MATCH + rank ordering and return the top results anyway.
        if (ftsResults.length <= maxResults) {
          logger?.debug?.(
            `${TAG} [keyword-fts] All ${ftsResults.length} results below threshold=${threshold} ` +
            `but document set is small — returning all matched results`,
          );
          return ftsResults.slice(0, maxResults).map((r) => formatMemoryLine(ftsResultToFormatable(r)));
        }
        logger?.debug?.(`${TAG} [keyword-fts] FTS5 returned 0 results above threshold (from ${ftsResults.length} raw)`);
      }
    }
  }

  // FTS5 not available or returned no results — skip in-memory fallback to avoid O(N) full scan
  logger?.debug?.(`${TAG} [keyword] FTS5 unavailable or no results, skipping keyword search`);
  return [];
}

// ============================
// Strategy: Embedding (VectorStore cosine)
// ============================

async function searchByEmbedding(
  userText: string,
  maxResults: number,
  threshold: number,
  vectorStore: VectorStore,
  embeddingService: EmbeddingService,
  logger?: Logger,
): Promise<string[]> {
  logger?.debug?.(
    `${TAG} [embedding-search] START query="${userText.slice(0, 80)}...", maxResults=${maxResults}, threshold=${threshold}`,
  );
  const queryEmbedding = await embeddingService.embed(userText);
  logger?.debug?.(
    `${TAG} [embedding-search] Query embedding OK: dims=${queryEmbedding.length}, ` +
    `norm=${Math.sqrt(Array.from(queryEmbedding).reduce((s, v) => s + v * v, 0)).toFixed(4)}, ` +
    `searching top-${maxResults * 2}...`,
  );
  // Retrieve more candidates for subsequent filtering
  const vecResults: VectorSearchResult[] = vectorStore.search(queryEmbedding, maxResults * 2);

  if (vecResults.length === 0) {
    logger?.debug?.(`${TAG} [embedding-search] Returned 0 results`);
    return [];
  }

  logger?.debug?.(`${TAG} [embedding-search] Got ${vecResults.length} candidates, filtering by threshold=${threshold}`);
  for (const r of vecResults) {
    logger?.debug?.(
      `${TAG} [embedding-search] candidate id=${r.record_id}, score=${r.score.toFixed(4)}, ` +
      `type=${r.type}, content="${r.content.slice(0, 60)}..."`,
    );
  }

  const filtered = vecResults
    .filter((r) => r.score >= threshold)
    .slice(0, maxResults);

  if (filtered.length > 0) {
    logger?.debug?.(`${TAG} [embedding-search] Found ${filtered.length} relevant memories above threshold (from ${vecResults.length} candidates)`);
    return filtered.map((r) => formatMemoryLine(vectorResultToFormatable(r)));
  }

  logger?.debug?.(`${TAG} [embedding-search] No results above threshold ${threshold}`);
  return [];
}

// ============================
// Strategy: Hybrid (Keyword + Embedding + RRF)
// ============================

/**
 * Hybrid search: run keyword (FTS5) and embedding in parallel, merge with
 * Reciprocal Rank Fusion (RRF) to combine rank lists.
 *
 * RRF score for a record at rank r = 1 / (k + r), where k=60 is a constant.
 * If a record appears in both lists, its RRF scores are summed.
 *
 * If FTS5 is unavailable, the keyword side returns empty and RRF uses
 * embedding results only.
 */
async function searchHybrid(
  userText: string,
  _pluginDataDir: string,
  maxResults: number,
  _threshold: number,
  vectorStore: VectorStore,
  embeddingService: EmbeddingService,
  logger?: Logger,
): Promise<string[]> {
  // Run keyword and embedding searches in parallel
  const candidateK = maxResults * 3; // retrieve more for merging

  const [keywordResults, embeddingResults] = await Promise.all([
    // Keyword search: FTS5 only (no in-memory fallback)
    (async () => {
      try {
        // Try FTS5 first
        if (vectorStore.isFtsAvailable()) {
          const ftsQuery = buildFtsQuery(userText);
          if (ftsQuery) {
            const ftsResults = vectorStore.ftsSearchL1(ftsQuery, candidateK);
            if (ftsResults.length > 0) {
              logger?.debug?.(`${TAG} [hybrid-keyword-fts] FTS5 found ${ftsResults.length} candidates`);
              // Convert FtsSearchResult to ScoredRecord for RRF merge
              return ftsResults.map((r): ScoredRecord => ({
                record: {
                  id: r.record_id,
                  content: r.content,
                  type: r.type as MemoryRecord["type"],
                  priority: r.priority,
                  scene_name: r.scene_name,
                  source_message_ids: [],
                  metadata: r.metadata_json ? (() => { try { return JSON.parse(r.metadata_json); } catch { return {}; } })() : {},
                  timestamps: [r.timestamp_str].filter(Boolean),
                  createdAt: "",
                  updatedAt: "",
                  sessionKey: r.session_key,
                  sessionId: r.session_id,
                },
                score: r.score,
              }));
            }
          }
        }
        // FTS5 not available or returned no results — skip in-memory fallback
        logger?.debug?.(`${TAG} [hybrid-keyword] FTS5 unavailable or no results, skipping keyword part`);
        return [] as ScoredRecord[];
      } catch (err) {
        logger?.warn?.(`${TAG} Hybrid: keyword part failed: ${err instanceof Error ? err.message : String(err)}`);
        return [] as ScoredRecord[];
      }
    })(),
    // Embedding search
    (async () => {
      try {
        logger?.debug?.(`${TAG} [hybrid-embedding] Generating query embedding...`);
        const queryEmbedding = await embeddingService.embed(userText);
        logger?.debug?.(
          `${TAG} [hybrid-embedding] Embedding OK, dims=${queryEmbedding.length}, searching top-${candidateK}...`,
        );
        const results = vectorStore.search(queryEmbedding, candidateK);
        logger?.debug?.(`${TAG} [hybrid-embedding] Got ${results.length} candidates`);
        return results;
      } catch (err) {
        logger?.warn?.(`${TAG} Hybrid: embedding part failed: ${err instanceof Error ? err.message : String(err)}`);
        return [] as VectorSearchResult[];
      }
    })(),
  ]);

  if (keywordResults.length === 0 && embeddingResults.length === 0) {
    logger?.debug?.(`${TAG} Hybrid search: both strategies returned 0 results`);
    return [];
  }

  // RRF merge: k=60 is a standard constant from the RRF paper
  const RRF_K = 60;

  // Map: record_id → { rrfScore, formatable }
  const mergedMap = new Map<string, { rrfScore: number; formatable: FormatableMemory }>();

  // Process keyword results
  for (let rank = 0; rank < keywordResults.length; rank++) {
    const r = keywordResults[rank];
    const id = r.record.id;
    const rrfScore = 1 / (RRF_K + rank + 1);
    const existing = mergedMap.get(id);
    if (existing) {
      existing.rrfScore += rrfScore;
    } else {
      mergedMap.set(id, { rrfScore, formatable: recordToFormatable(r.record) });
    }
  }

  // Process embedding results
  for (let rank = 0; rank < embeddingResults.length; rank++) {
    const r = embeddingResults[rank];
    const id = r.record_id;
    const rrfScore = 1 / (RRF_K + rank + 1);
    const existing = mergedMap.get(id);
    if (existing) {
      existing.rrfScore += rrfScore;
    } else {
      mergedMap.set(id, { rrfScore, formatable: vectorResultToFormatable(r) });
    }
  }

  // Sort by combined RRF score and take top results
  const sorted = [...mergedMap.entries()]
    .sort((a, b) => b[1].rrfScore - a[1].rrfScore)
    .slice(0, maxResults);

  if (sorted.length > 0) {
    logger?.debug?.(
      `${TAG} Hybrid search found ${sorted.length} results ` +
      `(keyword=${keywordResults.length}, embedding=${embeddingResults.length})`,
    );
    return sorted.map(([, { formatable }]) => formatMemoryLine(formatable));
  }

  logger?.debug?.(`${TAG} Hybrid search: no results after merge`);
  return [];
}

// ============================
// Unified memory line formatter
// ============================

/**
 * Format a single memory record into a rich natural-language line for prompt injection.
 *
 * Time semantics:
 *   - timestamp (点时间): when the activity/event happened, e.g. "2025-03-01 mentioned something"
 *   - activity_start_time / activity_end_time (段时间): activity time range, e.g. "trip from 2025-05-01 to 2025-05-10"
 *   - All three time fields may be empty/undefined — handled gracefully.
 *
 * Output examples:
 *   - [persona] 用户叫王小明，30岁，是一名软件工程师。
 *   - [episodic|旅行计划] 用户计划五月去日本旅行。(活动时间: 2025-05-01 ~ 2025-05-10)
 *   - [episodic] 用户今天加班到很晚。(活动时间: 2025-03-01)
 *   - [instruction] 用户要求回答时使用中文，保持简洁。
 */
interface FormatableMemory {
  type: string;
  content: string;
  scene_name?: string;
  /** Activity time range start (段时间 start), may be empty */
  activity_start_time?: string;
  /** Activity time range end (段时间 end), may be empty */
  activity_end_time?: string;
  /** Activity point-in-time (点时间: when it happened), may be empty */
  timestamp?: string;
}

function formatMemoryLine(m: FormatableMemory): string {
  // 1. Type tag + optional scene name
  const tag = m.scene_name ? `${m.type}|${m.scene_name}` : m.type;

  // 2. Content (core)
  let line = `- [${tag}] ${m.content}`;

  // 3. Time info — prefer activity_start/end range; fall back to timestamp as point-in-time
  const start = formatTimestamp(m.activity_start_time);
  const end = formatTimestamp(m.activity_end_time);
  const point = formatTimestamp(m.timestamp);

  if (start && end) {
    // 段时间: both start and end
    line += ` (活动时间: ${start} ~ ${end})`;
  } else if (start) {
    // 段时间: only start
    line += ` (活动时间: ${start}起)`;
  } else if (end) {
    // 段时间: only end
    line += ` (活动时间: 至${end})`;
  } else if (point) {
    // 点时间: single timestamp
    line += ` (活动时间: ${point})`;
  }
  // If all three are empty → no time info appended (graceful)

  return line;
}

/**
 * Format an ISO 8601 timestamp to a concise date or datetime string.
 * - If the time part is 00:00:00 → show date only (e.g. "2025-03-01")
 * - Otherwise → show date + time (e.g. "2025-03-01 14:30")
 * - Returns undefined for empty/invalid inputs.
 */
function formatTimestamp(ts: string | undefined): string | undefined {
  if (!ts) return undefined;
  // Try to parse ISO format: "2025-03-01T14:30:00.000Z" or "2025-03-01"
  const match = ts.match(/^(\d{4}-\d{2}-\d{2})(?:T(\d{2}:\d{2})(?::\d{2})?)?/);
  if (!match) return undefined;
  const datePart = match[1];
  const timePart = match[2];
  if (!timePart || timePart === "00:00") {
    return datePart;
  }
  return `${datePart} ${timePart}`;
}

/**
 * Build a FormatableMemory from a full MemoryRecord (keyword search path).
 * Handles empty metadata, empty timestamps array gracefully.
 */
function recordToFormatable(record: MemoryRecord): FormatableMemory {
  const meta = record.metadata as { activity_start_time?: string; activity_end_time?: string } | undefined;
  return {
    type: record.type,
    content: record.content,
    scene_name: record.scene_name || undefined,
    activity_start_time: meta?.activity_start_time || undefined,
    activity_end_time: meta?.activity_end_time || undefined,
    timestamp: (record.timestamps && record.timestamps.length > 0) ? record.timestamps[0] : undefined,
  };
}

/**
 * Build a FormatableMemory from a VectorSearchResult (embedding search path).
 * Handles empty/invalid metadata_json, empty timestamp_str gracefully.
 */
function vectorResultToFormatable(r: VectorSearchResult): FormatableMemory {
  let activityStart: string | undefined;
  let activityEnd: string | undefined;
  if (r.metadata_json && r.metadata_json !== "{}") {
    try {
      const meta = typeof r.metadata_json === "string" ? JSON.parse(r.metadata_json) : r.metadata_json;
      activityStart = meta?.activity_start_time || undefined;
      activityEnd = meta?.activity_end_time || undefined;
    } catch { /* ignore parse errors — treat as no metadata */ }
  }
  return {
    type: r.type,
    content: r.content,
    scene_name: r.scene_name || undefined,
    activity_start_time: activityStart,
    activity_end_time: activityEnd,
    timestamp: r.timestamp_str || undefined,
  };
}

/**
 * Build a FormatableMemory from an FtsSearchResult (FTS5 keyword search path).
 * Handles empty/invalid metadata_json, empty timestamp_str gracefully.
 */
function ftsResultToFormatable(r: FtsSearchResult): FormatableMemory {
  let activityStart: string | undefined;
  let activityEnd: string | undefined;
  if (r.metadata_json && r.metadata_json !== "{}") {
    try {
      const meta = typeof r.metadata_json === "string" ? JSON.parse(r.metadata_json) : r.metadata_json;
      activityStart = meta?.activity_start_time || undefined;
      activityEnd = meta?.activity_end_time || undefined;
    } catch { /* ignore parse errors — treat as no metadata */ }
  }
  return {
    type: r.type,
    content: r.content,
    scene_name: r.scene_name || undefined,
    activity_start_time: activityStart,
    activity_end_time: activityEnd,
    timestamp: r.timestamp_str || undefined,
  };
}


