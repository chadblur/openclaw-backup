/**
 * Unit tests for auto-recall hook (performAutoRecall).
 *
 * Covers:
 * - AR-01: before_prompt_build injects memory into appendSystemContext
 * - AR-02: maxResults + scoreThreshold filtering and limiting
 * - AR-03: VectorStore empty → returns empty, no error
 * - AR-04: Strategy branches: "keyword" / "embedding" / "hybrid"
 * - Additional: short text skip, persona loading, scene navigation, formatting, fallback
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { performAutoRecall } from "./auto-recall.js";
import type { RecallResult } from "./auto-recall.js";
import type { MemoryTdaiConfig } from "../config.js";
import type { VectorStore, VectorSearchResult } from "../store/vector-store.js";
import type { EmbeddingService } from "../store/embedding.js";
import { parseConfig } from "../config.js";

// ============================
// Mock modules: fs, scene-index, scene-navigation, l1-reader
// ============================

vi.mock("node:fs/promises", () => ({
  default: {
    readFile: vi.fn().mockRejectedValue(new Error("no file")),
    readdir: vi.fn().mockRejectedValue(new Error("no dir")),
  },
}));

vi.mock("../scene/scene-index.js", () => ({
  readSceneIndex: vi.fn().mockResolvedValue([]),
}));

vi.mock("../scene/scene-navigation.js", () => ({
  generateSceneNavigation: vi.fn().mockReturnValue(""),
  stripSceneNavigation: vi.fn().mockImplementation((s: string) => s),
}));

vi.mock("../record/l1-reader.js", () => ({
  queryMemoryRecords: vi.fn().mockReturnValue([]),
}));

// ============================
// Helpers
// ============================

function makeConfig(overrides?: Partial<MemoryTdaiConfig["recall"]>): MemoryTdaiConfig {
  const base = parseConfig({});
  return {
    ...base,
    recall: {
      ...base.recall,
      ...overrides,
    },
  };
}

function createMockLogger() {
  return {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  };
}

function createMockVectorStore(results: VectorSearchResult[] = []): VectorStore {
  return {
    search: vi.fn().mockReturnValue(results),
    searchL0: vi.fn().mockReturnValue([]),
    upsert: vi.fn().mockReturnValue(true),
    upsertL0: vi.fn().mockReturnValue(true),
    delete: vi.fn().mockReturnValue(true),
    deleteBatch: vi.fn().mockReturnValue(true),
    deleteL0: vi.fn().mockReturnValue(true),
    count: vi.fn().mockReturnValue(0),
    countL0: vi.fn().mockReturnValue(0),
    init: vi.fn().mockReturnValue({ needsReindex: false }),
    close: vi.fn(),
    isDegraded: vi.fn().mockReturnValue(false),
    getAllL1Texts: vi.fn().mockReturnValue([]),
    getAllL0Texts: vi.fn().mockReturnValue([]),
    reindexAll: vi.fn().mockResolvedValue({ l1Count: 0, l0Count: 0 }),
    queryL1Records: vi.fn().mockReturnValue([]),
    queryL0ForL1: vi.fn().mockReturnValue([]),
    queryL0GroupedBySessionId: vi.fn().mockReturnValue([]),
    deleteL1ExpiredByUpdatedTime: vi.fn().mockReturnValue(0),
    isFtsAvailable: vi.fn().mockReturnValue(false),
    ftsSearchL1: vi.fn().mockReturnValue([]),
    ftsSearchL0: vi.fn().mockReturnValue([]),
  } as unknown as VectorStore;
}

function createMockEmbeddingService(): EmbeddingService {
  return {
    embed: vi.fn().mockResolvedValue(new Float32Array([0.1, 0.2, 0.3])),
    embedBatch: vi.fn().mockResolvedValue([new Float32Array([0.1, 0.2, 0.3])]),
    getDimensions: vi.fn().mockReturnValue(768),
    getProviderInfo: vi.fn().mockReturnValue({ provider: "mock", model: "mock-model" }),
    isReady: vi.fn().mockReturnValue(true),
    startWarmup: vi.fn(),
  };
}

// ============================
// Tests
// ============================

describe("performAutoRecall", () => {
  beforeEach(async () => {
    vi.clearAllMocks();

    // Re-apply default mock implementations after clearAllMocks.
    // clearAllMocks only resets call history / return values, NOT the
    // implementation set via mockResolvedValue / mockReturnValue / mockImplementation.
    // Tests that override these (e.g. AR-01 "order context") leak state into later tests.
    const fs = await import("node:fs/promises");
    (fs.default.readFile as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("no file"));
    (fs.default.readdir as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("no dir"));

    const { readSceneIndex } = await import("../scene/scene-index.js");
    (readSceneIndex as ReturnType<typeof vi.fn>).mockResolvedValue([]);

    const { generateSceneNavigation, stripSceneNavigation } = await import("../scene/scene-navigation.js");
    (generateSceneNavigation as ReturnType<typeof vi.fn>).mockReturnValue("");
    (stripSceneNavigation as ReturnType<typeof vi.fn>).mockImplementation((s: string) => s);

    const { queryMemoryRecords } = await import("../record/l1-reader.js");
    (queryMemoryRecords as ReturnType<typeof vi.fn>).mockReturnValue([]);
  });

  // ─────────────────────────────────────
  // AR-01: Injects memory into appendSystemContext
  // ─────────────────────────────────────

  describe("AR-01: injects memory into appendSystemContext", () => {
    it("should return appendSystemContext with relevant-memories when keyword search finds matches", async () => {
      const mockVectorStore = createMockVectorStore();
      (mockVectorStore.isFtsAvailable as ReturnType<typeof vi.fn>).mockReturnValue(true);
      (mockVectorStore.ftsSearchL1 as ReturnType<typeof vi.fn>).mockReturnValue([
        {
          record_id: "mem-1",
          content: "用户喜欢编程和TypeScript",
          type: "persona",
          priority: 80,
          scene_name: "",
          score: 0.9,
          timestamp_str: "2026-03-01T00:00:00Z",
          timestamp_start: "",
          timestamp_end: "",
          session_key: "s1",
          session_id: "sid1",
          metadata_json: "{}",
        },
      ]);

      const cfg = makeConfig({ strategy: "keyword", maxResults: 5, scoreThreshold: 0.01 });
      const result = await performAutoRecall({
        userText: "我喜欢编程TypeScript开发",
        actorId: "actor-1",
        sessionKey: "s1",
        cfg,
        pluginDataDir: "/tmp/test-data",
        logger: createMockLogger(),
        vectorStore: mockVectorStore,
      });

      expect(result).toBeDefined();
      expect(result!.appendSystemContext).toBeDefined();
      expect(result!.appendSystemContext).toContain("<relevant-memories>");
      expect(result!.appendSystemContext).toContain("用户喜欢编程和TypeScript");
    });

    it("should include persona in appendSystemContext when persona file exists", async () => {
      const fs = await import("node:fs/promises");
      (fs.default.readFile as ReturnType<typeof vi.fn>).mockResolvedValue("用户是一名资深工程师，擅长后端开发。");

      const { stripSceneNavigation } = await import("../scene/scene-navigation.js");
      (stripSceneNavigation as ReturnType<typeof vi.fn>).mockImplementation((s: string) => s);

      const cfg = makeConfig({ strategy: "keyword" });
      const result = await performAutoRecall({
        userText: "告诉我关于用户的信息",
        actorId: "a1",
        sessionKey: "s1",
        cfg,
        pluginDataDir: "/tmp/test-data",
        logger: createMockLogger(),
      });

      expect(result).toBeDefined();
      expect(result!.appendSystemContext).toContain("<user-persona>");
      expect(result!.appendSystemContext).toContain("用户是一名资深工程师");
    });

    it("should include scene navigation when scene index has entries", async () => {
      const { readSceneIndex } = await import("../scene/scene-index.js");
      (readSceneIndex as ReturnType<typeof vi.fn>).mockResolvedValue([
        { filename: "scene-1.md", summary: "Work projects", heat: 50, created: "2026-01-01", updated: "2026-03-01" },
      ]);

      const { generateSceneNavigation } = await import("../scene/scene-navigation.js");
      (generateSceneNavigation as ReturnType<typeof vi.fn>).mockReturnValue("## Scene Nav\n- Work projects");

      const cfg = makeConfig({ strategy: "keyword" });
      const result = await performAutoRecall({
        userText: "关于工作项目的记忆",
        actorId: "a1",
        sessionKey: "s1",
        cfg,
        pluginDataDir: "/tmp/test-data",
        logger: createMockLogger(),
      });

      expect(result).toBeDefined();
      expect(result!.appendSystemContext).toContain("<scene-navigation>");
    });

    it("should order context: persona → scene-navigation → relevant-memories", async () => {
      const fs = await import("node:fs/promises");
      (fs.default.readFile as ReturnType<typeof vi.fn>).mockResolvedValue("Persona content");

      const { stripSceneNavigation } = await import("../scene/scene-navigation.js");
      (stripSceneNavigation as ReturnType<typeof vi.fn>).mockImplementation((s: string) => s);

      const { readSceneIndex } = await import("../scene/scene-index.js");
      (readSceneIndex as ReturnType<typeof vi.fn>).mockResolvedValue([
        { filename: "f.md", summary: "s", heat: 1, created: "", updated: "" },
      ]);

      const { generateSceneNavigation } = await import("../scene/scene-navigation.js");
      (generateSceneNavigation as ReturnType<typeof vi.fn>).mockReturnValue("Scene nav content");

      const mockVectorStore = createMockVectorStore();
      (mockVectorStore.isFtsAvailable as ReturnType<typeof vi.fn>).mockReturnValue(true);
      (mockVectorStore.ftsSearchL1 as ReturnType<typeof vi.fn>).mockReturnValue([
        {
          record_id: "m1",
          content: "Memory content here",
          type: "persona",
          priority: 50,
          scene_name: "",
          score: 0.9,
          timestamp_str: "",
          timestamp_start: "",
          timestamp_end: "",
          session_key: "",
          session_id: "",
          metadata_json: "{}",
        },
      ]);

      const cfg = makeConfig({ strategy: "keyword", scoreThreshold: 0.01 });
      const result = await performAutoRecall({
        userText: "Memory content Persona Scene nav",
        actorId: "a1",
        sessionKey: "s1",
        cfg,
        pluginDataDir: "/tmp/test-data",
        logger: createMockLogger(),
        vectorStore: mockVectorStore,
      });

      expect(result).toBeDefined();
      const ctx = result!.appendSystemContext!;
      const personaIdx = ctx.indexOf("<user-persona>");
      const sceneIdx = ctx.indexOf("<scene-navigation>");
      const memoryIdx = ctx.indexOf("<relevant-memories>");

      expect(personaIdx).toBeLessThan(sceneIdx);
      expect(sceneIdx).toBeLessThan(memoryIdx);
    });
  });

  // ─────────────────────────────────────
  // AR-02: maxResults + scoreThreshold
  // ─────────────────────────────────────

  describe("AR-02: maxResults + scoreThreshold filtering and limiting", () => {
    it("should limit results to maxResults", async () => {
      const mockVectorStore = createMockVectorStore();
      (mockVectorStore.isFtsAvailable as ReturnType<typeof vi.fn>).mockReturnValue(true);
      (mockVectorStore.ftsSearchL1 as ReturnType<typeof vi.fn>).mockReturnValue(
        Array.from({ length: 20 }, (_, i) => ({
          record_id: `mem-${i}`,
          content: `记忆编程TypeScript开发内容${i}`,
          type: "persona",
          priority: 50,
          scene_name: "",
          score: 0.9 - i * 0.01,
          timestamp_str: "",
          timestamp_start: "",
          timestamp_end: "",
          session_key: "",
          session_id: "",
          metadata_json: "{}",
        })),
      );

      const cfg = makeConfig({ strategy: "keyword", maxResults: 3, scoreThreshold: 0.01 });
      const result = await performAutoRecall({
        userText: "编程TypeScript开发",
        actorId: "a1",
        sessionKey: "s1",
        cfg,
        pluginDataDir: "/tmp/test-data",
        logger: createMockLogger(),
        vectorStore: mockVectorStore,
      });

      expect(result).toBeDefined();
      const lines = result!.appendSystemContext!
        .split("\n")
        .filter((l) => l.startsWith("- ["));
      expect(lines.length).toBeLessThanOrEqual(3);
    });

    it("should filter out results below scoreThreshold", async () => {
      const { queryMemoryRecords } = await import("../record/l1-reader.js");
      (queryMemoryRecords as ReturnType<typeof vi.fn>).mockReturnValue([
        {
          id: "mem-no-match",
          content: "完全不相关的内容ZZZZZ",
          type: "persona",
          priority: 50,
          scene_name: "",
          source_message_ids: [],
          metadata: {},
          timestamps: [],
          createdAt: "",
          updatedAt: "",
          sessionKey: "",
          sessionId: "",
        },
      ]);

      const cfg = makeConfig({ strategy: "keyword", scoreThreshold: 0.5 });
      const result = await performAutoRecall({
        userText: "编程TypeScript",
        actorId: "a1",
        sessionKey: "s1",
        cfg,
        pluginDataDir: "/tmp/test-data",
        logger: createMockLogger(),
      });

      // No match above threshold → undefined
      expect(result).toBeUndefined();
    });

    it("embedding search should respect maxResults and scoreThreshold", async () => {
      const mockVectorStore = createMockVectorStore([
        {
          record_id: "r1",
          content: "Memory 1",
          type: "persona",
          priority: 50,
          scene_name: "",
          score: 0.9,
          timestamp_str: "",
          timestamp_start: "",
          timestamp_end: "",
          session_key: "",
          session_id: "",
          metadata_json: "{}",
        },
        {
          record_id: "r2",
          content: "Memory 2",
          type: "episodic",
          priority: 50,
          scene_name: "",
          score: 0.8,
          timestamp_str: "",
          timestamp_start: "",
          timestamp_end: "",
          session_key: "",
          session_id: "",
          metadata_json: "{}",
        },
        {
          record_id: "r3",
          content: "Memory 3 low score",
          type: "persona",
          priority: 50,
          scene_name: "",
          score: 0.1,
          timestamp_str: "",
          timestamp_start: "",
          timestamp_end: "",
          session_key: "",
          session_id: "",
          metadata_json: "{}",
        },
      ]);
      const mockEmbedding = createMockEmbeddingService();

      const cfg = makeConfig({ strategy: "embedding", maxResults: 2, scoreThreshold: 0.3 });
      const result = await performAutoRecall({
        userText: "查找相关的记忆内容",
        actorId: "a1",
        sessionKey: "s1",
        cfg,
        pluginDataDir: "/tmp/test-data",
        logger: createMockLogger(),
        vectorStore: mockVectorStore,
        embeddingService: mockEmbedding,
      });

      expect(result).toBeDefined();
      // Should contain Memory 1 and 2 (above threshold), not Memory 3 (below)
      const ctx = result!.appendSystemContext!;
      expect(ctx).toContain("Memory 1");
      expect(ctx).toContain("Memory 2");
      expect(ctx).not.toContain("Memory 3 low score");
    });
  });

  // ─────────────────────────────────────
  // AR-03: VectorStore empty → returns empty
  // ─────────────────────────────────────

  describe("AR-03: VectorStore empty → returns empty, no error", () => {
    it("should return undefined when vectorStore returns no results", async () => {
      const mockVectorStore = createMockVectorStore([]);
      const mockEmbedding = createMockEmbeddingService();

      const cfg = makeConfig({ strategy: "embedding" });
      const result = await performAutoRecall({
        userText: "搜索一些内容试试看",
        actorId: "a1",
        sessionKey: "s1",
        cfg,
        pluginDataDir: "/tmp/test-data",
        logger: createMockLogger(),
        vectorStore: mockVectorStore,
        embeddingService: mockEmbedding,
      });

      expect(result).toBeUndefined();
    });

    it("should return undefined when no memories, persona, or scenes exist", async () => {
      const cfg = makeConfig({ strategy: "keyword" });
      const result = await performAutoRecall({
        userText: "这是一个测试查询内容",
        actorId: "a1",
        sessionKey: "s1",
        cfg,
        pluginDataDir: "/tmp/test-data",
        logger: createMockLogger(),
      });

      expect(result).toBeUndefined();
    });
  });

  // ─────────────────────────────────────
  // AR-04: Strategy branches
  // ─────────────────────────────────────

  describe("AR-04: strategy branches (keyword / embedding / hybrid)", () => {
    it("keyword strategy: uses FTS5 search (not VectorStore cosine)", async () => {
      const mockVectorStore = createMockVectorStore();
      (mockVectorStore.isFtsAvailable as ReturnType<typeof vi.fn>).mockReturnValue(true);
      (mockVectorStore.ftsSearchL1 as ReturnType<typeof vi.fn>).mockReturnValue([
        {
          record_id: "m1",
          content: "Keyword match test 编程",
          type: "instruction",
          priority: 50,
          scene_name: "",
          score: 0.9,
          timestamp_str: "",
          timestamp_start: "",
          timestamp_end: "",
          session_key: "",
          session_id: "",
          metadata_json: "{}",
        },
      ]);

      const cfg = makeConfig({ strategy: "keyword", scoreThreshold: 0.01 });
      const result = await performAutoRecall({
        userText: "Keyword match test 编程",
        actorId: "a1",
        sessionKey: "s1",
        cfg,
        pluginDataDir: "/tmp/test-data",
        logger: createMockLogger(),
        vectorStore: mockVectorStore,
      });

      expect(result).toBeDefined();
      // VectorStore.search (cosine) should NOT be called for keyword strategy
      expect(mockVectorStore.search).not.toHaveBeenCalled();
      // FTS5 should be used
      expect(mockVectorStore.ftsSearchL1).toHaveBeenCalled();
    });

    it("embedding strategy: uses VectorStore search", async () => {
      const mockVectorStore = createMockVectorStore([
        {
          record_id: "r1",
          content: "Embedding result",
          type: "persona",
          priority: 50,
          scene_name: "",
          score: 0.9,
          timestamp_str: "",
          timestamp_start: "",
          timestamp_end: "",
          session_key: "",
          session_id: "",
          metadata_json: "{}",
        },
      ]);
      const mockEmbedding = createMockEmbeddingService();

      const cfg = makeConfig({ strategy: "embedding" });
      const result = await performAutoRecall({
        userText: "搜索嵌入向量记忆内容",
        actorId: "a1",
        sessionKey: "s1",
        cfg,
        pluginDataDir: "/tmp/test-data",
        logger: createMockLogger(),
        vectorStore: mockVectorStore,
        embeddingService: mockEmbedding,
      });

      expect(result).toBeDefined();
      expect(mockVectorStore.search).toHaveBeenCalled();
      expect(mockEmbedding.embed).toHaveBeenCalled();
    });

    it("hybrid strategy: merges keyword and embedding results with RRF", async () => {
      const { queryMemoryRecords } = await import("../record/l1-reader.js");
      (queryMemoryRecords as ReturnType<typeof vi.fn>).mockReturnValue([
        {
          id: "keyword-only",
          content: "Only in keyword search 编程开发",
          type: "persona",
          priority: 50,
          scene_name: "",
          source_message_ids: [],
          metadata: {},
          timestamps: [],
          createdAt: "",
          updatedAt: "",
          sessionKey: "",
          sessionId: "",
        },
      ]);

      const mockVectorStore = createMockVectorStore([
        {
          record_id: "embedding-only",
          content: "Only in embedding search",
          type: "episodic",
          priority: 50,
          scene_name: "",
          score: 0.85,
          timestamp_str: "",
          timestamp_start: "",
          timestamp_end: "",
          session_key: "",
          session_id: "",
          metadata_json: "{}",
        },
      ]);
      const mockEmbedding = createMockEmbeddingService();

      const cfg = makeConfig({ strategy: "hybrid", maxResults: 10, scoreThreshold: 0.01 });
      const result = await performAutoRecall({
        userText: "编程开发 hybrid search test",
        actorId: "a1",
        sessionKey: "s1",
        cfg,
        pluginDataDir: "/tmp/test-data",
        logger: createMockLogger(),
        vectorStore: mockVectorStore,
        embeddingService: mockEmbedding,
      });

      expect(result).toBeDefined();
      // Both sources contribute
      expect(mockVectorStore.search).toHaveBeenCalled();
      expect(mockEmbedding.embed).toHaveBeenCalled();
    });

    it("embedding/hybrid falls back to keyword when vectorStore is unavailable (returns empty without FTS5)", async () => {
      const logger = createMockLogger();
      const cfg = makeConfig({ strategy: "embedding", scoreThreshold: 0.01 });
      const result = await performAutoRecall({
        userText: "Fallback keyword result 测试回退",
        actorId: "a1",
        sessionKey: "s1",
        cfg,
        pluginDataDir: "/tmp/test-data",
        logger,
        // No vectorStore or embeddingService provided
      });

      // Without vectorStore, strategy falls back to keyword, but without FTS5 keyword returns empty
      expect(result).toBeUndefined();
      expect(logger.warn).toHaveBeenCalledWith(
        expect.stringContaining("falling back to keyword"),
      );
    });
  });

  // ─────────────────────────────────────
  // Edge cases
  // ─────────────────────────────────────

  describe("edge cases", () => {
    it("should still inject persona/scene when user text is short (skips memory search only)", async () => {
      const fs = await import("node:fs/promises");
      (fs.default.readFile as ReturnType<typeof vi.fn>).mockResolvedValue("用户是一名工程师。");

      const { stripSceneNavigation } = await import("../scene/scene-navigation.js");
      (stripSceneNavigation as ReturnType<typeof vi.fn>).mockImplementation((s: string) => s);

      const cfg = makeConfig();
      const result = await performAutoRecall({
        userText: "hi",
        actorId: "a1",
        sessionKey: "s1",
        cfg,
        pluginDataDir: "/tmp/test-data",
        logger: createMockLogger(),
      });

      // Short text skips memory search, but persona is still injected
      expect(result).toBeDefined();
      expect(result!.appendSystemContext).toContain("<user-persona>");
      expect(result!.appendSystemContext).toContain("用户是一名工程师");
    });

    it("should still inject persona/scene when user text is empty (skips memory search only)", async () => {
      const fs = await import("node:fs/promises");
      (fs.default.readFile as ReturnType<typeof vi.fn>).mockResolvedValue("用户喜欢简洁的回答。");

      const { stripSceneNavigation } = await import("../scene/scene-navigation.js");
      (stripSceneNavigation as ReturnType<typeof vi.fn>).mockImplementation((s: string) => s);

      const cfg = makeConfig();
      const result = await performAutoRecall({
        userText: "",
        actorId: "a1",
        sessionKey: "s1",
        cfg,
        pluginDataDir: "/tmp/test-data",
        logger: createMockLogger(),
      });

      // Empty text skips memory search, but persona is still injected
      expect(result).toBeDefined();
      expect(result!.appendSystemContext).toContain("<user-persona>");
      expect(result!.appendSystemContext).toContain("用户喜欢简洁的回答");
      // No relevant-memories section
      expect(result!.appendSystemContext).not.toContain("<relevant-memories>");
    });

    it("should return undefined when user text is empty AND no persona/scene exist", async () => {
      const cfg = makeConfig();
      const result = await performAutoRecall({
        userText: "",
        actorId: "a1",
        sessionKey: "s1",
        cfg,
        pluginDataDir: "/tmp/test-data",
        logger: createMockLogger(),
      });

      // No memories, no persona, no scenes → undefined
      expect(result).toBeUndefined();
    });

    it("should handle persona file read failure gracefully", async () => {
      const fs = await import("node:fs/promises");
      (fs.default.readFile as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("ENOENT"));

      const cfg = makeConfig({ strategy: "keyword" });
      // Should not throw
      const result = await performAutoRecall({
        userText: "这是一个正常的查询文本",
        actorId: "a1",
        sessionKey: "s1",
        cfg,
        pluginDataDir: "/tmp/test-data",
        logger: createMockLogger(),
      });

      // No error thrown, just no persona
      expect(result === undefined || result?.appendSystemContext?.indexOf("<user-persona>") === -1).toBeTruthy();
    });

    it("should handle scene index read failure gracefully", async () => {
      const { readSceneIndex } = await import("../scene/scene-index.js");
      (readSceneIndex as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("fail"));

      const cfg = makeConfig({ strategy: "keyword" });
      // Should not throw
      const result = await performAutoRecall({
        userText: "这是一个正常的查询文本",
        actorId: "a1",
        sessionKey: "s1",
        cfg,
        pluginDataDir: "/tmp/test-data",
        logger: createMockLogger(),
      });

      // Graceful — no scene navigation, but no crash
      expect(true).toBe(true);
    });

    it("should handle memory search failure gracefully and return empty", async () => {
      const { queryMemoryRecords } = await import("../record/l1-reader.js");
      (queryMemoryRecords as ReturnType<typeof vi.fn>).mockImplementation(() => { throw new Error("disk fail"); });

      const cfg = makeConfig({ strategy: "keyword" });
      const result = await performAutoRecall({
        userText: "搜索失败场景测试内容",
        actorId: "a1",
        sessionKey: "s1",
        cfg,
        pluginDataDir: "/tmp/test-data",
        logger: createMockLogger(),
      });

      // Should degrade gracefully — not throw
      expect(result).toBeUndefined();
    });

    it("should format memory lines with activity time ranges correctly", async () => {
      const mockVectorStore = createMockVectorStore();
      (mockVectorStore.isFtsAvailable as ReturnType<typeof vi.fn>).mockReturnValue(true);
      (mockVectorStore.ftsSearchL1 as ReturnType<typeof vi.fn>).mockReturnValue([
        {
          record_id: "m-time",
          content: "用户五月去日本旅行",
          type: "episodic",
          priority: 80,
          scene_name: "旅行计划",
          score: 0.9,
          timestamp_str: "2026-03-01T14:30:00Z",
          timestamp_start: "",
          timestamp_end: "",
          session_key: "",
          session_id: "",
          metadata_json: JSON.stringify({
            activity_start_time: "2026-05-01T00:00:00Z",
            activity_end_time: "2026-05-10T00:00:00Z",
          }),
        },
      ]);

      const cfg = makeConfig({ strategy: "keyword", scoreThreshold: 0.01 });
      const result = await performAutoRecall({
        userText: "用户旅行计划日本五月",
        actorId: "a1",
        sessionKey: "s1",
        cfg,
        pluginDataDir: "/tmp/test-data",
        logger: createMockLogger(),
        vectorStore: mockVectorStore,
      });

      expect(result).toBeDefined();
      const ctx = result!.appendSystemContext!;
      // Should contain scene name in tag
      expect(ctx).toContain("episodic|旅行计划");
      // Should contain activity time range
      expect(ctx).toContain("活动时间: 2026-05-01 ~ 2026-05-10");
    });

    it("should handle malformed metadata_json in vector results gracefully", async () => {
      const mockVectorStore = createMockVectorStore([
        {
          record_id: "r-bad-meta",
          content: "Bad metadata record 测试",
          type: "persona",
          priority: 50,
          scene_name: "",
          score: 0.9,
          timestamp_str: "",
          timestamp_start: "",
          timestamp_end: "",
          session_key: "",
          session_id: "",
          metadata_json: "NOT_VALID_JSON",
        },
      ]);
      const mockEmbedding = createMockEmbeddingService();

      const cfg = makeConfig({ strategy: "embedding" });
      const result = await performAutoRecall({
        userText: "Bad metadata record 测试查询",
        actorId: "a1",
        sessionKey: "s1",
        cfg,
        pluginDataDir: "/tmp/test-data",
        logger: createMockLogger(),
        vectorStore: mockVectorStore,
        embeddingService: mockEmbedding,
      });

      // Should not throw — just no activity time info
      expect(result).toBeDefined();
    });
  });
});
