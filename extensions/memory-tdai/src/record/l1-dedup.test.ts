/**
 * Unit tests for L1 Memory Conflict Detection / Dedup.
 *
 * v4: Updated to match the new 3-tier degradation strategy:
 *   Tier 1: Vector recall (vectorStore + embeddingService)
 *   Tier 2: FTS5 keyword recall (vectorStore with FTS)
 *   Tier 3: Skip conflict detection entirely (all → store)
 *
 * The old JSONL-based Jaccard fallback and loadExistingRecords have been removed.
 *
 * DD-01: Candidates found → LLM judgment (mocked as fail → fallback store)
 * DD-02: No candidates → all store (fast path)
 * DD-04: Empty memory list → empty result
 */
import { describe, it, expect, vi } from "vitest";
import { batchDedup } from "./l1-dedup.js";
import type { ExtractedMemory, MemoryRecord } from "./l1-writer.js";

function makeMemory(content: string, opts?: Partial<ExtractedMemory>): ExtractedMemory & { record_id: string } {
  return {
    content,
    type: opts?.type ?? "persona",
    priority: opts?.priority ?? 50,
    source_message_ids: opts?.source_message_ids ?? [],
    metadata: opts?.metadata ?? {},
    scene_name: opts?.scene_name ?? "",
    record_id: `m_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
  };
}

// ── DD-04: Empty memory list ──

describe("DD-04: empty memory list returns empty", () => {
  it("should return empty array for empty memories", async () => {
    const result = await batchDedup({ memories: [], config: {} });
    expect(result).toEqual([]);
  });
});

// ── Fast paths (no LLM involved) ──

describe("fast path: no recall capability → skip dedup, all store", () => {
  it("should store all when no vectorStore provided at all", async () => {
    const m1 = makeMemory("User prefers dark mode IDE");
    const m2 = makeMemory("User likes TypeScript language");

    const result = await batchDedup({ memories: [m1, m2], config: {} });

    expect(result).toHaveLength(2);
    expect(result[0]).toMatchObject({ record_id: m1.record_id, action: "store", target_ids: [] });
    expect(result[1]).toMatchObject({ record_id: m2.record_id, action: "store", target_ids: [] });
  });

  it("should store all when vectorStore has count=0 and no FTS", async () => {
    const mockVS = {
      count: vi.fn().mockReturnValue(0),
      search: vi.fn(), upsert: vi.fn(), deleteBatch: vi.fn(),
      queryL1Records: vi.fn().mockReturnValue([]),
      close: vi.fn(),
      isFtsAvailable: vi.fn().mockReturnValue(false),
      ftsSearchL1: vi.fn().mockReturnValue([]),
      ftsSearchL0: vi.fn().mockReturnValue([]),
    };

    const m = makeMemory("User prefers tea over coffee");
    const result = await batchDedup({
      memories: [m],
      config: {},
      vectorStore: mockVS as any,
    });

    // count=0, no FTS → skip dedup → store
    expect(result).toHaveLength(1);
    expect(result[0].action).toBe("store");
    expect(mockVS.search).not.toHaveBeenCalled();
  });

  it("should store all when vectorStore has count=0 and FTS returns empty", async () => {
    const mockVS = {
      count: vi.fn().mockReturnValue(0),
      search: vi.fn(), upsert: vi.fn(), deleteBatch: vi.fn(),
      queryL1Records: vi.fn().mockReturnValue([]),
      close: vi.fn(),
      isFtsAvailable: vi.fn().mockReturnValue(true),
      ftsSearchL1: vi.fn().mockReturnValue([]),
      ftsSearchL0: vi.fn().mockReturnValue([]),
    };

    const m = makeMemory("User enjoys hiking outdoors");
    const result = await batchDedup({
      memories: [m],
      config: {},
      vectorStore: mockVS as any,
    });

    // count=0 but FTS available → use FTS → empty results → store
    expect(result).toHaveLength(1);
    expect(result[0].action).toBe("store");
    expect(mockVS.ftsSearchL1).toHaveBeenCalled();
  });
});

// ── Tier 1: Vector recall ──

describe("Tier 1: vector recall", () => {
  it("should store all when vector search returns empty", async () => {
    const m = makeMemory("User prefers dark theme for coding");

    const mockVS = {
      count: vi.fn().mockReturnValue(5),
      search: vi.fn().mockReturnValue([]),
      upsert: vi.fn(), deleteBatch: vi.fn(),
      queryL1Records: vi.fn().mockReturnValue([]),
      close: vi.fn(),
      isFtsAvailable: vi.fn().mockReturnValue(false),
      ftsSearchL1: vi.fn().mockReturnValue([]),
      ftsSearchL0: vi.fn().mockReturnValue([]),
    };
    const mockES = {
      embed: vi.fn().mockResolvedValue(new Float32Array([0.1, 0.2, 0.3])),
      embedBatch: vi.fn().mockResolvedValue([new Float32Array([0.1, 0.2, 0.3])]),
      getDimensions: vi.fn().mockReturnValue(3),
      getProviderInfo: vi.fn().mockReturnValue({ provider: "mock", model: "mock" }),
    };

    const result = await batchDedup({
      memories: [m],
      config: {},
      vectorStore: mockVS as any,
      embeddingService: mockES as any,
    });

    expect(mockES.embedBatch).toHaveBeenCalledWith(["User prefers dark theme for coding"]);
    expect(mockVS.search).toHaveBeenCalled();
    expect(result).toHaveLength(1);
    expect(result[0].action).toBe("store");
  });

  it("should exclude current batch IDs from vector search results", async () => {
    const m1 = makeMemory("User prefers coding in TypeScript");
    const m2 = makeMemory("User likes working with React");

    const mockVS = {
      count: vi.fn().mockReturnValue(10),
      search: vi.fn().mockImplementation(() => {
        // Only return self-batch matches → after filtering, no candidates
        return [
          { record_id: m1.record_id, content: "self", type: "persona", priority: 50, scene_name: "", score: 0.99, timestamp_str: "", timestamp_start: "", timestamp_end: "", session_key: "s", session_id: "", metadata_json: "{}" },
          { record_id: m2.record_id, content: "self", type: "persona", priority: 50, scene_name: "", score: 0.98, timestamp_str: "", timestamp_start: "", timestamp_end: "", session_key: "s", session_id: "", metadata_json: "{}" },
        ];
      }),
      upsert: vi.fn(), deleteBatch: vi.fn(),
      queryL1Records: vi.fn().mockReturnValue([]),
      close: vi.fn(),
      isFtsAvailable: vi.fn().mockReturnValue(false),
      ftsSearchL1: vi.fn().mockReturnValue([]),
      ftsSearchL0: vi.fn().mockReturnValue([]),
    };
    const mockES = {
      embed: vi.fn(),
      embedBatch: vi.fn().mockResolvedValue([new Float32Array([0.1, 0.2, 0.3]), new Float32Array([0.4, 0.5, 0.6])]),
      getDimensions: vi.fn().mockReturnValue(3),
      getProviderInfo: vi.fn().mockReturnValue({ provider: "mock", model: "mock" }),
    };

    const result = await batchDedup({
      memories: [m1, m2],
      config: {},
      vectorStore: mockVS as any,
      embeddingService: mockES as any,
    });

    // After excluding self-batch, no candidates → all store
    expect(result).toHaveLength(2);
    expect(result.every((d) => d.action === "store")).toBe(true);
  });

  it("should pass custom conflictRecallTopK to vector search", async () => {
    const m = makeMemory("User likes morning running exercise");

    const mockVS = {
      count: vi.fn().mockReturnValue(20),
      search: vi.fn().mockReturnValue([]),
      upsert: vi.fn(), deleteBatch: vi.fn(),
      queryL1Records: vi.fn().mockReturnValue([]),
      close: vi.fn(),
      isFtsAvailable: vi.fn().mockReturnValue(false),
      ftsSearchL1: vi.fn().mockReturnValue([]),
      ftsSearchL0: vi.fn().mockReturnValue([]),
    };
    const mockES = {
      embed: vi.fn(),
      embedBatch: vi.fn().mockResolvedValue([new Float32Array([0.1, 0.2, 0.3])]),
      getDimensions: vi.fn().mockReturnValue(3),
      getProviderInfo: vi.fn().mockReturnValue({ provider: "mock", model: "mock" }),
    };

    await batchDedup({
      memories: [m],
      config: {},
      vectorStore: mockVS as any,
      embeddingService: mockES as any,
      conflictRecallTopK: 3,
    });

    // search should be called with topK + memories.length = 3 + 1 = 4
    expect(mockVS.search).toHaveBeenCalledWith(expect.any(Float32Array), 4);
  });

  it("should attempt LLM when vector candidates are found (→ fails → store)", async () => {
    const log = { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() };
    const m = makeMemory("User prefers coding in dark theme");

    const mockVS = {
      count: vi.fn().mockReturnValue(10),
      search: vi.fn().mockReturnValue([
        { record_id: "existing_1", content: "User likes dark mode", type: "persona", priority: 50, scene_name: "", score: 0.92, timestamp_str: "", timestamp_start: "", timestamp_end: "", session_key: "s", session_id: "", metadata_json: "{}" },
      ]),
      upsert: vi.fn(), deleteBatch: vi.fn(),
      queryL1Records: vi.fn().mockReturnValue([]),
      close: vi.fn(),
      isFtsAvailable: vi.fn().mockReturnValue(false),
      ftsSearchL1: vi.fn().mockReturnValue([]),
      ftsSearchL0: vi.fn().mockReturnValue([]),
    };
    const mockES = {
      embed: vi.fn(),
      embedBatch: vi.fn().mockResolvedValue([new Float32Array([0.1, 0.2, 0.3])]),
      getDimensions: vi.fn().mockReturnValue(3),
      getProviderInfo: vi.fn().mockReturnValue({ provider: "mock", model: "mock" }),
    };

    const result = await batchDedup({
      memories: [m],
      config: {},
      logger: log,
      vectorStore: mockVS as any,
      embeddingService: mockES as any,
    });

    // Candidates found → LLM judgment attempted → fails → fallback store
    expect(result).toHaveLength(1);
    expect(result[0].action).toBe("store");
    expect(log.warn).toHaveBeenCalled();
  }, 75_000);
});

// ── Tier 1 → Tier 2 degradation: vector fails → FTS ──

describe("Tier 1 → Tier 2: vector recall fails → FTS fallback", () => {
  it("should fallback to FTS when embedBatch fails and FTS available", async () => {
    const log = { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() };
    const m = makeMemory("Completely unique content xyz abc");

    const mockVS = {
      count: vi.fn().mockReturnValue(5),
      search: vi.fn(), upsert: vi.fn(), deleteBatch: vi.fn(),
      queryL1Records: vi.fn().mockReturnValue([]),
      close: vi.fn(),
      isFtsAvailable: vi.fn().mockReturnValue(true),
      ftsSearchL1: vi.fn().mockReturnValue([]),
      ftsSearchL0: vi.fn().mockReturnValue([]),
    };
    const mockES = {
      embed: vi.fn(),
      embedBatch: vi.fn().mockRejectedValue(new Error("embedding service down")),
      getDimensions: vi.fn().mockReturnValue(3),
      getProviderInfo: vi.fn().mockReturnValue({ provider: "mock", model: "mock" }),
    };

    const result = await batchDedup({
      memories: [m],
      config: {},
      logger: log,
      vectorStore: mockVS as any,
      embeddingService: mockES as any,
    });

    expect(log.warn).toHaveBeenCalledWith(expect.stringContaining("Vector recall failed"));
    // FTS called as fallback
    expect(mockVS.ftsSearchL1).toHaveBeenCalled();
    expect(result).toHaveLength(1);
    expect(result[0].action).toBe("store");
  });

  it("should skip dedup when embedBatch fails and FTS not available", async () => {
    const log = { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() };
    const m = makeMemory("Completely unique content xyz abc");

    const mockVS = {
      count: vi.fn().mockReturnValue(5),
      search: vi.fn(), upsert: vi.fn(), deleteBatch: vi.fn(),
      queryL1Records: vi.fn().mockReturnValue([]),
      close: vi.fn(),
      isFtsAvailable: vi.fn().mockReturnValue(false),
      ftsSearchL1: vi.fn().mockReturnValue([]),
      ftsSearchL0: vi.fn().mockReturnValue([]),
    };
    const mockES = {
      embed: vi.fn(),
      embedBatch: vi.fn().mockRejectedValue(new Error("embedding service down")),
      getDimensions: vi.fn().mockReturnValue(3),
      getProviderInfo: vi.fn().mockReturnValue({ provider: "mock", model: "mock" }),
    };

    const result = await batchDedup({
      memories: [m],
      config: {},
      logger: log,
      vectorStore: mockVS as any,
      embeddingService: mockES as any,
    });

    expect(log.warn).toHaveBeenCalledWith(expect.stringContaining("Vector recall failed"));
    // FTS not available → skip dedup → store
    expect(mockVS.ftsSearchL1).not.toHaveBeenCalled();
    expect(result).toHaveLength(1);
    expect(result[0].action).toBe("store");
  });

  it("should fallback to FTS when vector search throws", async () => {
    const log = { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() };
    const m = makeMemory("User prefers dark theme coding IDE");

    const mockVS = {
      count: vi.fn().mockReturnValue(10),
      search: vi.fn().mockImplementation(() => { throw new Error("SQLite corrupt"); }),
      upsert: vi.fn(), deleteBatch: vi.fn(),
      queryL1Records: vi.fn().mockReturnValue([]),
      close: vi.fn(),
      isFtsAvailable: vi.fn().mockReturnValue(true),
      ftsSearchL1: vi.fn().mockReturnValue([]),
      ftsSearchL0: vi.fn().mockReturnValue([]),
    };
    const mockES = {
      embed: vi.fn(),
      embedBatch: vi.fn().mockResolvedValue([new Float32Array([0.1, 0.2, 0.3])]),
      getDimensions: vi.fn().mockReturnValue(3),
      getProviderInfo: vi.fn().mockReturnValue({ provider: "mock", model: "mock" }),
    };

    const result = await batchDedup({
      memories: [m],
      config: {},
      logger: log,
      vectorStore: mockVS as any,
      embeddingService: mockES as any,
    });

    // embedBatch succeeded but search threw → fallback to FTS → no results → store
    expect(log.warn).toHaveBeenCalledWith(expect.stringContaining("Vector recall failed"));
    expect(mockVS.ftsSearchL1).toHaveBeenCalled();
    expect(result).toHaveLength(1);
    expect(result[0].action).toBe("store");
  });
});

// ── Tier 2: FTS keyword recall (no embedding service) ──

describe("Tier 2: FTS keyword recall (no embedding service)", () => {
  it("should use FTS when vectorStore has data but no embeddingService", async () => {
    const log = { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() };
    const m = makeMemory("User enjoys hiking outdoors");

    const mockVS = {
      count: vi.fn().mockReturnValue(10),
      search: vi.fn(), upsert: vi.fn(), deleteBatch: vi.fn(),
      queryL1Records: vi.fn().mockReturnValue([]),
      close: vi.fn(),
      isFtsAvailable: vi.fn().mockReturnValue(true),
      ftsSearchL1: vi.fn().mockReturnValue([]),
      ftsSearchL0: vi.fn().mockReturnValue([]),
    };

    const result = await batchDedup({
      memories: [m],
      config: {},
      logger: log,
      vectorStore: mockVS as any,
      // No embeddingService
    });

    // FTS available → use FTS → no results → store
    expect(mockVS.ftsSearchL1).toHaveBeenCalled();
    expect(mockVS.search).not.toHaveBeenCalled();
    expect(result).toHaveLength(1);
    expect(result[0].action).toBe("store");
  });

  it("should skip dedup when vectorStore has data but no embeddingService and no FTS", async () => {
    const log = { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() };
    const m = makeMemory("User enjoys hiking outdoors");

    const mockVS = {
      count: vi.fn().mockReturnValue(10),
      search: vi.fn(), upsert: vi.fn(), deleteBatch: vi.fn(),
      queryL1Records: vi.fn().mockReturnValue([]),
      close: vi.fn(),
      isFtsAvailable: vi.fn().mockReturnValue(false),
      ftsSearchL1: vi.fn().mockReturnValue([]),
      ftsSearchL0: vi.fn().mockReturnValue([]),
    };

    const result = await batchDedup({
      memories: [m],
      config: {},
      logger: log,
      vectorStore: mockVS as any,
      // No embeddingService
    });

    // No embedding, no FTS → skip dedup → store
    expect(mockVS.ftsSearchL1).not.toHaveBeenCalled();
    expect(mockVS.search).not.toHaveBeenCalled();
    expect(result).toHaveLength(1);
    expect(result[0].action).toBe("store");
  });

  it("should attempt LLM when FTS finds candidates (→ fails → store)", async () => {
    const log = { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() };
    const m = makeMemory("用户喜欢在晚上弹钢琴");

    const mockVS = {
      count: vi.fn().mockReturnValue(0), // no vector data
      search: vi.fn(), upsert: vi.fn(), deleteBatch: vi.fn(),
      queryL1Records: vi.fn().mockReturnValue([]),
      close: vi.fn(),
      isFtsAvailable: vi.fn().mockReturnValue(true),
      ftsSearchL1: vi.fn().mockReturnValue([
        { record_id: "r1", content: "用户经常在晚上弹钢琴练习", type: "persona", priority: 50, scene_name: "", score: 0.85, timestamp_str: "", session_key: "s", session_id: "", metadata_json: "{}" },
      ]),
      ftsSearchL0: vi.fn().mockReturnValue([]),
    };

    const result = await batchDedup({
      memories: [m],
      config: {},
      logger: log,
      vectorStore: mockVS as any,
    });

    // FTS found candidate → LLM judgment attempted → fails (no config) → fallback store
    expect(mockVS.ftsSearchL1).toHaveBeenCalled();
    expect(result).toHaveLength(1);
    expect(result[0].action).toBe("store");
    expect(log.warn).toHaveBeenCalled();
  }, 75_000);

  it("should exclude current batch IDs from FTS results", async () => {
    const m1 = makeMemory("User prefers coding in TypeScript");
    const m2 = makeMemory("User likes working with React");

    const mockVS = {
      count: vi.fn().mockReturnValue(0),
      search: vi.fn(), upsert: vi.fn(), deleteBatch: vi.fn(),
      queryL1Records: vi.fn().mockReturnValue([]),
      close: vi.fn(),
      isFtsAvailable: vi.fn().mockReturnValue(true),
      ftsSearchL1: vi.fn().mockImplementation(() => {
        // Only return self-batch matches → after filtering, no candidates
        return [
          { record_id: m1.record_id, content: "self", type: "persona", priority: 50, scene_name: "", score: 0.9, timestamp_str: "", session_key: "s", session_id: "", metadata_json: "{}" },
          { record_id: m2.record_id, content: "self", type: "persona", priority: 50, scene_name: "", score: 0.8, timestamp_str: "", session_key: "s", session_id: "", metadata_json: "{}" },
        ];
      }),
      ftsSearchL0: vi.fn().mockReturnValue([]),
    };

    const result = await batchDedup({
      memories: [m1, m2],
      config: {},
      vectorStore: mockVS as any,
    });

    // After excluding self-batch, no candidates → all store
    expect(result).toHaveLength(2);
    expect(result.every((d) => d.action === "store")).toBe(true);
  });
});

// ── Batch: one decision per memory ──

describe("batch: multiple memories get individual decisions", () => {
  it("should return one decision per memory", async () => {
    const m1 = makeMemory("Memory content number one here");
    const m2 = makeMemory("Memory content number two here");
    const m3 = makeMemory("Memory content number three");

    const result = await batchDedup({ memories: [m1, m2, m3], config: {} });

    expect(result).toHaveLength(3);
    const ids = new Set(result.map((d) => d.record_id));
    expect(ids.has(m1.record_id)).toBe(true);
    expect(ids.has(m2.record_id)).toBe(true);
    expect(ids.has(m3.record_id)).toBe(true);
  });
});
