/**
 * Unit tests for memory_search tool (hybrid search with RRF).
 *
 * T-01: executeMemorySearch parameter validation + correct VectorStore call
 * T-03: Empty / special character queries → safe handling
 *
 * Additional edge cases:
 * - VectorStore or EmbeddingService unavailable
 * - Embedding fails → degrades to FTS-only
 * - FTS fails → degrades to embedding-only
 * - Both available → hybrid (RRF merge)
 * - Type filter
 * - Scene filter (case-insensitive partial match)
 * - Limit trimming
 * - formatSearchResponse formatting
 */
import { describe, it, expect, vi } from "vitest";
import {
  executeMemorySearch,
  formatSearchResponse,
} from "./memory-search.js";
import type { MemorySearchResult } from "./memory-search.js";

// ── Helpers ──

const logger = () => ({
  debug: vi.fn(),
  info: vi.fn(),
  warn: vi.fn(),
  error: vi.fn(),
});

function createMockVectorStore(searchResults: Array<{
  record_id: string;
  content: string;
  type: string;
  priority: number;
  scene_name: string;
  score: number;
  timestamp_start: string;
  timestamp_end: string;
}> = []) {
  return {
    search: vi.fn().mockReturnValue(
      searchResults.map((r) => ({
        ...r,
        timestamp_str: r.timestamp_start,
        session_key: "sk",
        session_id: "",
        metadata_json: "{}",
      })),
    ),
    upsert: vi.fn(),
    delete: vi.fn(),
    deleteBatch: vi.fn(),
    count: vi.fn(),
    countL0: vi.fn(),
    searchL0: vi.fn(),
    queryL1Records: vi.fn().mockReturnValue([]),
    close: vi.fn(),
    isFtsAvailable: vi.fn().mockReturnValue(false),
    ftsSearchL1: vi.fn().mockReturnValue([]),
    ftsSearchL0: vi.fn().mockReturnValue([]),
  };
}

function createMockEmbeddingService() {
  return {
    embed: vi.fn().mockResolvedValue(new Float32Array([0.1, 0.2, 0.3])),
    embedBatch: vi.fn().mockResolvedValue([new Float32Array([0.1, 0.2, 0.3])]),
    getDimensions: vi.fn().mockReturnValue(3),
    getProviderInfo: vi.fn().mockReturnValue({ provider: "mock", model: "mock" }),
  };
}

// ── T-01: Parameter validation + correct VectorStore call ──

describe("T-01: executeMemorySearch parameter validation", () => {
  it("should call embed() with query and search with candidateK", async () => {
    const log = logger();
    const vs = createMockVectorStore([
      { record_id: "r1", content: "User likes coffee", type: "persona", priority: 50, scene_name: "food", score: 0.85, timestamp_start: "2026-03-17", timestamp_end: "2026-03-17" },
    ]);
    const es = createMockEmbeddingService();

    const result = await executeMemorySearch({
      query: "coffee preference",
      limit: 5,
      vectorStore: vs as any,
      embeddingService: es as any,
      logger: log,
    });

    expect(es.embed).toHaveBeenCalledWith("coffee preference");
    // candidateK = limit * 3 = 15
    expect(vs.search).toHaveBeenCalledWith(expect.any(Float32Array), 15);
    expect(result.results).toHaveLength(1);
    expect(result.results[0].id).toBe("r1");
    expect(result.results[0].content).toBe("User likes coffee");
    // FTS not available → embedding-only
    expect(result.strategy).toBe("embedding");
  });

  it("should map VectorSearchResult to MemorySearchResultItem correctly", async () => {
    const vs = createMockVectorStore([
      { record_id: "r1", content: "Test memory", type: "episodic", priority: 70, scene_name: "work", score: 0.92, timestamp_start: "2026-03-15", timestamp_end: "2026-03-17" },
    ]);
    const es = createMockEmbeddingService();

    const result = await executeMemorySearch({
      query: "test",
      limit: 10,
      vectorStore: vs as any,
      embeddingService: es as any,
    });

    const item = result.results[0];
    expect(item.id).toBe("r1");
    expect(item.content).toBe("Test memory");
    expect(item.type).toBe("episodic");
    expect(item.priority).toBe(70);
    expect(item.scene_name).toBe("work");
    expect(item.created_at).toBe("2026-03-15");
    expect(item.updated_at).toBe("2026-03-17");
  });
});

// ── T-03: Empty / special character queries ──

describe("T-03: empty / special character queries", () => {
  it("should return empty for empty string query", async () => {
    const result = await executeMemorySearch({
      query: "",
      limit: 5,
    });
    expect(result.results).toEqual([]);
    expect(result.total).toBe(0);
  });

  it("should return empty for whitespace-only query", async () => {
    const result = await executeMemorySearch({
      query: "   \n  ",
      limit: 5,
    });
    expect(result.results).toEqual([]);
    expect(result.total).toBe(0);
  });

  it("should handle special characters in query without crashing", async () => {
    const vs = createMockVectorStore([]);
    const es = createMockEmbeddingService();

    const result = await executeMemorySearch({
      query: "User's <preference> & \"dark\" mode; DROP TABLE;--",
      limit: 5,
      vectorStore: vs as any,
      embeddingService: es as any,
    });

    expect(es.embed).toHaveBeenCalled();
    expect(result.results).toEqual([]);
  });

  it("should handle CJK characters in query", async () => {
    const vs = createMockVectorStore([
      { record_id: "r1", content: "用户喜欢喝咖啡", type: "persona", priority: 50, scene_name: "饮食", score: 0.88, timestamp_start: "", timestamp_end: "" },
    ]);
    const es = createMockEmbeddingService();

    const result = await executeMemorySearch({
      query: "咖啡偏好",
      limit: 5,
      vectorStore: vs as any,
      embeddingService: es as any,
    });

    expect(result.results).toHaveLength(1);
    expect(result.results[0].content).toBe("用户喜欢喝咖啡");
  });
});

// ── VectorStore / EmbeddingService unavailable ──

describe("missing dependencies", () => {
  it("should return empty when vectorStore is undefined", async () => {
    const log = logger();
    const es = createMockEmbeddingService();

    const result = await executeMemorySearch({
      query: "test",
      limit: 5,
      embeddingService: es as any,
      logger: log,
    });

    expect(result.results).toEqual([]);
    expect(log.warn).toHaveBeenCalledWith(
      expect.stringContaining("not available"),
    );
  });

  it("should return empty when embeddingService is undefined and FTS unavailable", async () => {
    const log = logger();
    const vs = createMockVectorStore();
    // isFtsAvailable defaults to false in mock

    const result = await executeMemorySearch({
      query: "test",
      limit: 5,
      vectorStore: vs as any,
      logger: log,
    });

    expect(result.results).toEqual([]);
    expect(result.strategy).toBe("none");
    expect(result.message).toBeDefined();
  });

  it("should return empty when both are undefined", async () => {
    const result = await executeMemorySearch({
      query: "test",
      limit: 5,
    });
    expect(result.results).toEqual([]);
  });
});

// ── Embedding failure (degrades to FTS-only) ──

describe("embedding failure", () => {
  it("should return empty when embed() throws and FTS is not available", async () => {
    const log = logger();
    const vs = createMockVectorStore();
    const es = createMockEmbeddingService();
    es.embed.mockRejectedValue(new Error("API timeout"));

    const result = await executeMemorySearch({
      query: "test query",
      limit: 5,
      vectorStore: vs as any,
      embeddingService: es as any,
      logger: log,
    });

    expect(result.results).toEqual([]);
    expect(log.warn).toHaveBeenCalledWith(
      expect.stringContaining("Embedding search failed"),
    );
  });

  it("should degrade to FTS-only when embed() throws but FTS is available", async () => {
    const log = logger();
    const vs = createMockVectorStore();
    vs.isFtsAvailable.mockReturnValue(true);
    vs.ftsSearchL1.mockReturnValue([
      {
        record_id: "fts-r1", content: "FTS result", type: "persona",
        priority: 50, scene_name: "", score: 0.7,
        timestamp_str: "", timestamp_start: "2026-03-17",
        timestamp_end: "2026-03-17", session_key: "", session_id: "",
        metadata_json: "{}",
      },
    ]);
    const es = createMockEmbeddingService();
    es.embed.mockRejectedValue(new Error("API timeout"));

    const result = await executeMemorySearch({
      query: "test query",
      limit: 5,
      vectorStore: vs as any,
      embeddingService: es as any,
      logger: log,
    });

    expect(result.strategy).toBe("fts");
    expect(result.results).toHaveLength(1);
    expect(result.results[0].id).toBe("fts-r1");
  });
});

// ── Hybrid search (both FTS + embedding available) ──

describe("hybrid search (RRF merge)", () => {
  it("should use strategy='hybrid' when both FTS and embedding return results", async () => {
    const vs = createMockVectorStore([
      { record_id: "vec-1", content: "Vector result", type: "persona", priority: 50, scene_name: "", score: 0.9, timestamp_start: "", timestamp_end: "" },
    ]);
    vs.isFtsAvailable.mockReturnValue(true);
    vs.ftsSearchL1.mockReturnValue([
      {
        record_id: "fts-1", content: "FTS result", type: "persona",
        priority: 50, scene_name: "", score: 0.8,
        timestamp_str: "", timestamp_start: "", timestamp_end: "",
        session_key: "", session_id: "", metadata_json: "{}",
      },
    ]);
    const es = createMockEmbeddingService();

    const result = await executeMemorySearch({
      query: "test",
      limit: 10,
      vectorStore: vs as any,
      embeddingService: es as any,
    });

    expect(result.strategy).toBe("hybrid");
    expect(result.results.length).toBeGreaterThanOrEqual(1);
    // Both results should be present
    const ids = result.results.map((r) => r.id);
    expect(ids).toContain("vec-1");
    expect(ids).toContain("fts-1");
  });

  it("should boost records that appear in both FTS and embedding results via RRF", async () => {
    // "shared-1" appears in both lists → highest combined RRF score
    const vs = createMockVectorStore([
      { record_id: "shared-1", content: "Shared result", type: "persona", priority: 50, scene_name: "", score: 0.9, timestamp_start: "", timestamp_end: "" },
      { record_id: "vec-only", content: "Vec only", type: "persona", priority: 50, scene_name: "", score: 0.8, timestamp_start: "", timestamp_end: "" },
    ]);
    vs.isFtsAvailable.mockReturnValue(true);
    vs.ftsSearchL1.mockReturnValue([
      {
        record_id: "shared-1", content: "Shared result", type: "persona",
        priority: 50, scene_name: "", score: 0.85,
        timestamp_str: "", timestamp_start: "", timestamp_end: "",
        session_key: "", session_id: "", metadata_json: "{}",
      },
      {
        record_id: "fts-only", content: "FTS only", type: "persona",
        priority: 50, scene_name: "", score: 0.7,
        timestamp_str: "", timestamp_start: "", timestamp_end: "",
        session_key: "", session_id: "", metadata_json: "{}",
      },
    ]);
    const es = createMockEmbeddingService();

    const result = await executeMemorySearch({
      query: "test",
      limit: 10,
      vectorStore: vs as any,
      embeddingService: es as any,
    });

    expect(result.strategy).toBe("hybrid");
    // "shared-1" should be ranked first due to RRF boost from appearing in both lists
    expect(result.results[0].id).toBe("shared-1");
    expect(result.results).toHaveLength(3);
  });

  it("should degrade to embedding-only when FTS is available but returns no results", async () => {
    const vs = createMockVectorStore([
      { record_id: "vec-1", content: "Vec result", type: "persona", priority: 50, scene_name: "", score: 0.9, timestamp_start: "", timestamp_end: "" },
    ]);
    vs.isFtsAvailable.mockReturnValue(true);
    vs.ftsSearchL1.mockReturnValue([]); // FTS returns nothing
    const es = createMockEmbeddingService();

    const result = await executeMemorySearch({
      query: "test",
      limit: 10,
      vectorStore: vs as any,
      embeddingService: es as any,
    });

    expect(result.strategy).toBe("embedding");
    expect(result.results).toHaveLength(1);
    expect(result.results[0].id).toBe("vec-1");
  });

  it("should degrade to FTS-only when FTS throws are caught and only FTS results remain", async () => {
    const vs = createMockVectorStore();
    vs.isFtsAvailable.mockReturnValue(true);
    vs.ftsSearchL1.mockImplementation(() => { throw new Error("FTS corrupt"); });
    const es = createMockEmbeddingService();

    const result = await executeMemorySearch({
      query: "test",
      limit: 5,
      vectorStore: vs as any,
      embeddingService: es as any,
    });

    // FTS threw, vec returned 0 results from empty mock → both empty
    expect(result.results).toEqual([]);
  });
});

// ── Type filter ──

describe("type filter", () => {
  it("should filter results by exact type match", async () => {
    const vs = createMockVectorStore([
      { record_id: "r1", content: "Persona memory", type: "persona", priority: 50, scene_name: "", score: 0.9, timestamp_start: "", timestamp_end: "" },
      { record_id: "r2", content: "Episodic memory", type: "episodic", priority: 50, scene_name: "", score: 0.8, timestamp_start: "", timestamp_end: "" },
      { record_id: "r3", content: "Instruction memory", type: "instruction", priority: 50, scene_name: "", score: 0.7, timestamp_start: "", timestamp_end: "" },
    ]);
    const es = createMockEmbeddingService();

    const result = await executeMemorySearch({
      query: "test",
      limit: 10,
      type: "persona",
      vectorStore: vs as any,
      embeddingService: es as any,
    });

    expect(result.results).toHaveLength(1);
    expect(result.results[0].type).toBe("persona");
  });

  it("should return empty when no results match type filter", async () => {
    const vs = createMockVectorStore([
      { record_id: "r1", content: "Persona", type: "persona", priority: 50, scene_name: "", score: 0.9, timestamp_start: "", timestamp_end: "" },
    ]);
    const es = createMockEmbeddingService();

    const result = await executeMemorySearch({
      query: "test",
      limit: 10,
      type: "episodic",
      vectorStore: vs as any,
      embeddingService: es as any,
    });

    expect(result.results).toEqual([]);
  });
});

// ── Scene filter ──

describe("scene filter", () => {
  it("should filter by case-insensitive partial scene name match", async () => {
    const vs = createMockVectorStore([
      { record_id: "r1", content: "M1", type: "persona", priority: 50, scene_name: "Work-Habits", score: 0.9, timestamp_start: "", timestamp_end: "" },
      { record_id: "r2", content: "M2", type: "persona", priority: 50, scene_name: "Food-Preferences", score: 0.8, timestamp_start: "", timestamp_end: "" },
    ]);
    const es = createMockEmbeddingService();

    const result = await executeMemorySearch({
      query: "test",
      limit: 10,
      scene: "work",
      vectorStore: vs as any,
      embeddingService: es as any,
    });

    expect(result.results).toHaveLength(1);
    expect(result.results[0].scene_name).toBe("Work-Habits");
  });
});

// ── Limit trimming ──

describe("limit trimming", () => {
  it("should trim results to requested limit after filtering", async () => {
    const items = Array.from({ length: 10 }, (_, i) => ({
      record_id: `r${i}`,
      content: `Memory ${i}`,
      type: "persona" as const,
      priority: 50,
      scene_name: "",
      score: 0.9 - i * 0.01,
      timestamp_start: "",
      timestamp_end: "",
    }));
    const vs = createMockVectorStore(items);
    const es = createMockEmbeddingService();

    const result = await executeMemorySearch({
      query: "test",
      limit: 3,
      vectorStore: vs as any,
      embeddingService: es as any,
    });

    expect(result.results).toHaveLength(3);
    expect(result.total).toBe(3);
  });
});

// ── formatSearchResponse ──

describe("formatSearchResponse", () => {
  it("should return 'no matching' message for empty results", () => {
    const result: MemorySearchResult = { results: [], total: 0, strategy: "embedding" };
    expect(formatSearchResponse(result)).toBe("No matching memories found.");
  });

  it("should format results with type, score, scene, priority", () => {
    const result: MemorySearchResult = {
      results: [
        { id: "r1", content: "User likes dark mode", type: "persona", priority: 80, scene_name: "coding", score: 0.92, created_at: "", updated_at: "" },
      ],
      total: 1,
      strategy: "embedding",
    };

    const text = formatSearchResponse(result);
    expect(text).toContain("1 matching memories");
    expect(text).toContain("[persona]");
    expect(text).toContain("(priority: 80)");
    expect(text).toContain("[scene: coding]");
    expect(text).toContain("(score: 0.920)");
    expect(text).toContain("User likes dark mode");
  });

  it("should show 'global instruction' for negative priority", () => {
    const result: MemorySearchResult = {
      results: [
        { id: "r1", content: "Always respond in English", type: "instruction", priority: -1, scene_name: "", score: 0.99, created_at: "", updated_at: "" },
      ],
      total: 1,
      strategy: "embedding",
    };

    const text = formatSearchResponse(result);
    expect(text).toContain("(global instruction)");
    expect(text).not.toContain("(priority:");
  });

  it("should omit scene when scene_name is empty", () => {
    const result: MemorySearchResult = {
      results: [
        { id: "r1", content: "Test", type: "persona", priority: 50, scene_name: "", score: 0.8, created_at: "", updated_at: "" },
      ],
      total: 1,
      strategy: "embedding",
    };

    const text = formatSearchResponse(result);
    expect(text).not.toContain("[scene:");
  });

  it("should format multiple results", () => {
    const result: MemorySearchResult = {
      results: [
        { id: "r1", content: "First memory", type: "persona", priority: 80, scene_name: "", score: 0.9, created_at: "", updated_at: "" },
        { id: "r2", content: "Second memory", type: "episodic", priority: 60, scene_name: "work", score: 0.7, created_at: "", updated_at: "" },
      ],
      total: 2,
      strategy: "embedding",
    };

    const text = formatSearchResponse(result);
    expect(text).toContain("2 matching memories");
    expect(text).toContain("First memory");
    expect(text).toContain("Second memory");
  });
});

// ── FTS-only search (when embeddingService is unavailable) ──

describe("FTS-only search", () => {
  it("should use FTS5 when embeddingService is undefined and FTS is available", async () => {
    const log = logger();
    const vs = createMockVectorStore();
    vs.isFtsAvailable.mockReturnValue(true);
    vs.ftsSearchL1.mockReturnValue([
      {
        record_id: "fts-r1",
        content: "FTS matched memory",
        type: "persona",
        priority: 60,
        scene_name: "coding",
        score: 0.75,
        timestamp_str: "2026-03-17",
        timestamp_start: "2026-03-17",
        timestamp_end: "2026-03-17",
        session_key: "sk",
        session_id: "",
        metadata_json: "{}",
      },
    ]);

    const result = await executeMemorySearch({
      query: "FTS matched",
      limit: 5,
      vectorStore: vs as any,
      logger: log,
    });

    expect(result.strategy).toBe("fts");
    expect(result.results).toHaveLength(1);
    expect(result.results[0].id).toBe("fts-r1");
    expect(result.results[0].content).toBe("FTS matched memory");
    expect(result.results[0].scene_name).toBe("coding");
    expect(vs.ftsSearchL1).toHaveBeenCalled();
  });

  it("should apply type filter in FTS-only mode", async () => {
    const vs = createMockVectorStore();
    vs.isFtsAvailable.mockReturnValue(true);
    vs.ftsSearchL1.mockReturnValue([
      { record_id: "f1", content: "Persona mem", type: "persona", priority: 50, scene_name: "", score: 0.8, timestamp_str: "", timestamp_start: "", timestamp_end: "", session_key: "", session_id: "", metadata_json: "{}" },
      { record_id: "f2", content: "Episodic mem", type: "episodic", priority: 50, scene_name: "", score: 0.7, timestamp_str: "", timestamp_start: "", timestamp_end: "", session_key: "", session_id: "", metadata_json: "{}" },
    ]);

    const result = await executeMemorySearch({
      query: "mem",
      limit: 10,
      type: "persona",
      vectorStore: vs as any,
    });

    expect(result.strategy).toBe("fts");
    expect(result.results).toHaveLength(1);
    expect(result.results[0].type).toBe("persona");
  });

  it("should apply scene filter in FTS-only mode", async () => {
    const vs = createMockVectorStore();
    vs.isFtsAvailable.mockReturnValue(true);
    vs.ftsSearchL1.mockReturnValue([
      { record_id: "f1", content: "Work stuff", type: "persona", priority: 50, scene_name: "Work-Habits", score: 0.8, timestamp_str: "", timestamp_start: "", timestamp_end: "", session_key: "", session_id: "", metadata_json: "{}" },
      { record_id: "f2", content: "Food stuff", type: "persona", priority: 50, scene_name: "Food", score: 0.7, timestamp_str: "", timestamp_start: "", timestamp_end: "", session_key: "", session_id: "", metadata_json: "{}" },
    ]);

    const result = await executeMemorySearch({
      query: "stuff",
      limit: 10,
      scene: "work",
      vectorStore: vs as any,
    });

    expect(result.strategy).toBe("fts");
    expect(result.results).toHaveLength(1);
    expect(result.results[0].scene_name).toBe("Work-Habits");
  });

  it("should trim FTS results to limit", async () => {
    const vs = createMockVectorStore();
    vs.isFtsAvailable.mockReturnValue(true);
    vs.ftsSearchL1.mockReturnValue(
      Array.from({ length: 10 }, (_, i) => ({
        record_id: `f${i}`, content: `Memory ${i}`, type: "persona", priority: 50,
        scene_name: "", score: 0.9 - i * 0.05, timestamp_str: "", timestamp_start: "",
        timestamp_end: "", session_key: "", session_id: "", metadata_json: "{}",
      })),
    );

    const result = await executeMemorySearch({
      query: "Memory",
      limit: 3,
      vectorStore: vs as any,
    });

    expect(result.strategy).toBe("fts");
    expect(result.results).toHaveLength(3);
  });

  it("should return empty when FTS query produces no usable tokens", async () => {
    const vs = createMockVectorStore();
    vs.isFtsAvailable.mockReturnValue(true);

    const result = await executeMemorySearch({
      query: "!@#$%^&*()",
      limit: 5,
      vectorStore: vs as any,
    });

    // FTS produced no tokens → no results
    expect(result.results).toEqual([]);
  });

  it("should return empty gracefully when ftsSearchL1 throws", async () => {
    const log = logger();
    const vs = createMockVectorStore();
    vs.isFtsAvailable.mockReturnValue(true);
    vs.ftsSearchL1.mockImplementation(() => { throw new Error("FTS corrupt"); });

    const result = await executeMemorySearch({
      query: "test query",
      limit: 5,
      vectorStore: vs as any,
      logger: log,
    });

    expect(result.results).toEqual([]);
  });

  it("should show message when embedding and FTS are both unavailable", async () => {
    const vs = createMockVectorStore();
    // isFtsAvailable defaults to false

    const result = await executeMemorySearch({
      query: "test",
      limit: 5,
      vectorStore: vs as any,
    });

    expect(result.strategy).toBe("none");
    expect(result.message).toContain("Embedding service is not configured");
    expect(result.results).toEqual([]);
  });

  it("should display message via formatSearchResponse when message is set", () => {
    const result: MemorySearchResult = {
      results: [],
      total: 0,
      strategy: "none",
      message: "Custom warning message",
    };
    expect(formatSearchResponse(result)).toBe("Custom warning message");
  });
});
