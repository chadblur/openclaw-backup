/**
 * Unit tests for conversation_search tool (hybrid search with RRF).
 *
 * T-02: executeConversationSearch parameter validation + correct L0 search
 * T-03: Empty / special character queries → safe handling
 *
 * Additional edge cases:
 * - VectorStore or EmbeddingService unavailable
 * - Embedding fails → degrades to FTS-only
 * - FTS fails → degrades to embedding-only
 * - Both available → hybrid (RRF merge)
 * - Session key filter
 * - Limit trimming
 * - formatConversationSearchResponse formatting
 */
import { describe, it, expect, vi } from "vitest";
import {
  executeConversationSearch,
  formatConversationSearchResponse,
} from "./conversation-search.js";
import type { ConversationSearchResult } from "./conversation-search.js";

// ── Helpers ──

const logger = () => ({
  debug: vi.fn(),
  info: vi.fn(),
  warn: vi.fn(),
  error: vi.fn(),
});

function createMockVectorStore(l0Results: Array<{
  record_id: string;
  session_key: string;
  session_id?: string;
  role: string;
  message_text: string;
  score: number;
  recorded_at: string;
  timestamp?: number;
}> = []) {
  return {
    searchL0: vi.fn().mockReturnValue(
      l0Results.map((r) => ({
        record_id: r.record_id,
        session_key: r.session_key,
        session_id: r.session_id ?? "",
        role: r.role,
        message_text: r.message_text,
        score: r.score,
        recorded_at: r.recorded_at,
        timestamp: r.timestamp ?? 0,
      })),
    ),
    search: vi.fn().mockReturnValue([]),
    upsert: vi.fn(),
    delete: vi.fn(),
    deleteBatch: vi.fn(),
    count: vi.fn(),
    countL0: vi.fn(),
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

// ── T-02: Parameter validation + correct L0 search ──

describe("T-02: executeConversationSearch parameter validation", () => {
  it("should call embed() with query and searchL0 with candidateK", async () => {
    const log = logger();
    const vs = createMockVectorStore([
      { record_id: "l0_1", session_key: "sk1", role: "user", message_text: "Hello world", score: 0.88, recorded_at: "2026-03-17T10:00:00Z" },
    ]);
    const es = createMockEmbeddingService();

    const result = await executeConversationSearch({
      query: "hello",
      limit: 5,
      vectorStore: vs as any,
      embeddingService: es as any,
      logger: log,
    });

    expect(es.embed).toHaveBeenCalledWith("hello");
    // Without sessionFilter, candidateK = limit * 3 = 15
    expect(vs.searchL0).toHaveBeenCalledWith(expect.any(Float32Array), 15);
    expect(result.results).toHaveLength(1);
    expect(result.results[0].id).toBe("l0_1");
    expect(result.results[0].content).toBe("Hello world");
    expect(result.results[0].role).toBe("user");
    expect(result.total).toBe(1);
    expect(result.strategy).toBe("embedding");
  });

  it("should map L0VectorSearchResult to ConversationSearchResultItem correctly", async () => {
    const vs = createMockVectorStore([
      { record_id: "l0_x", session_key: "sk_abc", role: "assistant", message_text: "I can help you", score: 0.75, recorded_at: "2026-03-17T12:30:00Z" },
    ]);
    const es = createMockEmbeddingService();

    const result = await executeConversationSearch({
      query: "help",
      limit: 10,
      vectorStore: vs as any,
      embeddingService: es as any,
    });

    const item = result.results[0];
    expect(item.id).toBe("l0_x");
    expect(item.session_key).toBe("sk_abc");
    expect(item.role).toBe("assistant");
    expect(item.content).toBe("I can help you");
    expect(item.recorded_at).toBe("2026-03-17T12:30:00Z");
  });
});

// ── T-03: Empty / special character queries ──

describe("T-03: empty / special character queries", () => {
  it("should return empty for empty string query", async () => {
    const result = await executeConversationSearch({
      query: "",
      limit: 5,
    });
    expect(result.results).toEqual([]);
    expect(result.total).toBe(0);
  });

  it("should return empty for whitespace-only query", async () => {
    const result = await executeConversationSearch({
      query: "   \t  ",
      limit: 5,
    });
    expect(result.results).toEqual([]);
  });

  it("should handle special characters in query", async () => {
    const vs = createMockVectorStore([]);
    const es = createMockEmbeddingService();

    const result = await executeConversationSearch({
      query: "<script>alert('xss')</script> OR 1=1; --",
      limit: 5,
      vectorStore: vs as any,
      embeddingService: es as any,
    });

    expect(es.embed).toHaveBeenCalled();
    expect(result.results).toEqual([]);
  });

  it("should handle CJK characters in query", async () => {
    const vs = createMockVectorStore([
      { record_id: "l0_cjk", session_key: "sk", role: "user", message_text: "今天天气怎么样", score: 0.82, recorded_at: "" },
    ]);
    const es = createMockEmbeddingService();

    const result = await executeConversationSearch({
      query: "天气",
      limit: 5,
      vectorStore: vs as any,
      embeddingService: es as any,
    });

    expect(result.results).toHaveLength(1);
    expect(result.results[0].content).toBe("今天天气怎么样");
  });
});

// ── Missing dependencies ──

describe("missing dependencies", () => {
  it("should return empty when vectorStore is undefined", async () => {
    const log = logger();
    const es = createMockEmbeddingService();

    const result = await executeConversationSearch({
      query: "test",
      limit: 5,
      embeddingService: es as any,
      logger: log,
    });

    expect(result.results).toEqual([]);
    expect(log.warn).toHaveBeenCalledWith(expect.stringContaining("not available"));
  });

  it("should return empty with message when embeddingService is undefined and FTS unavailable", async () => {
    const log = logger();
    const vs = createMockVectorStore();
    // isFtsAvailable defaults to false

    const result = await executeConversationSearch({
      query: "test",
      limit: 5,
      vectorStore: vs as any,
      logger: log,
    });

    expect(result.results).toEqual([]);
    expect(result.strategy).toBe("none");
    expect(result.message).toBeDefined();
    expect(result.message).toContain("Embedding service is not configured");
  });

  it("should return empty when both are undefined", async () => {
    const result = await executeConversationSearch({
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
    es.embed.mockRejectedValue(new Error("network timeout"));

    const result = await executeConversationSearch({
      query: "test query",
      limit: 5,
      vectorStore: vs as any,
      embeddingService: es as any,
      logger: log,
    });

    expect(result.results).toEqual([]);
    expect(log.warn).toHaveBeenCalledWith(expect.stringContaining("Embedding search failed"));
  });

  it("should degrade to FTS-only when embed() throws but FTS is available", async () => {
    const log = logger();
    const vs = createMockVectorStore();
    vs.isFtsAvailable.mockReturnValue(true);
    vs.ftsSearchL0.mockReturnValue([
      {
        record_id: "fts-l0-1", session_key: "sk_fts", session_id: "",
        role: "user", message_text: "FTS L0 result", score: 0.72,
        recorded_at: "2026-03-17T12:00:00Z", timestamp: 0,
      },
    ]);
    const es = createMockEmbeddingService();
    es.embed.mockRejectedValue(new Error("network timeout"));

    const result = await executeConversationSearch({
      query: "test query",
      limit: 5,
      vectorStore: vs as any,
      embeddingService: es as any,
      logger: log,
    });

    expect(result.strategy).toBe("fts");
    expect(result.results).toHaveLength(1);
    expect(result.results[0].id).toBe("fts-l0-1");
  });
});

// ── Hybrid search (both FTS + embedding available) ──

describe("hybrid search (RRF merge)", () => {
  it("should use strategy='hybrid' when both FTS and embedding return results", async () => {
    const vs = createMockVectorStore([
      { record_id: "vec-l0-1", session_key: "sk", role: "user", message_text: "Vector result", score: 0.9, recorded_at: "" },
    ]);
    vs.isFtsAvailable.mockReturnValue(true);
    vs.ftsSearchL0.mockReturnValue([
      {
        record_id: "fts-l0-1", session_key: "sk", session_id: "",
        role: "user", message_text: "FTS result", score: 0.8,
        recorded_at: "", timestamp: 0,
      },
    ]);
    const es = createMockEmbeddingService();

    const result = await executeConversationSearch({
      query: "test",
      limit: 10,
      vectorStore: vs as any,
      embeddingService: es as any,
    });

    expect(result.strategy).toBe("hybrid");
    expect(result.results.length).toBeGreaterThanOrEqual(1);
    const ids = result.results.map((r) => r.id);
    expect(ids).toContain("vec-l0-1");
    expect(ids).toContain("fts-l0-1");
  });

  it("should boost records that appear in both FTS and embedding results via RRF", async () => {
    const vs = createMockVectorStore([
      { record_id: "shared-1", session_key: "sk", role: "user", message_text: "Shared msg", score: 0.9, recorded_at: "" },
      { record_id: "vec-only", session_key: "sk", role: "user", message_text: "Vec only", score: 0.8, recorded_at: "" },
    ]);
    vs.isFtsAvailable.mockReturnValue(true);
    vs.ftsSearchL0.mockReturnValue([
      { record_id: "shared-1", session_key: "sk", session_id: "", role: "user", message_text: "Shared msg", score: 0.85, recorded_at: "", timestamp: 0 },
      { record_id: "fts-only", session_key: "sk", session_id: "", role: "assistant", message_text: "FTS only", score: 0.7, recorded_at: "", timestamp: 0 },
    ]);
    const es = createMockEmbeddingService();

    const result = await executeConversationSearch({
      query: "test",
      limit: 10,
      vectorStore: vs as any,
      embeddingService: es as any,
    });

    expect(result.strategy).toBe("hybrid");
    // "shared-1" should be ranked first due to RRF boost
    expect(result.results[0].id).toBe("shared-1");
    expect(result.results).toHaveLength(3);
  });

  it("should degrade to embedding-only when FTS returns no results", async () => {
    const vs = createMockVectorStore([
      { record_id: "vec-1", session_key: "sk", role: "user", message_text: "Vec result", score: 0.9, recorded_at: "" },
    ]);
    vs.isFtsAvailable.mockReturnValue(true);
    vs.ftsSearchL0.mockReturnValue([]);
    const es = createMockEmbeddingService();

    const result = await executeConversationSearch({
      query: "test",
      limit: 10,
      vectorStore: vs as any,
      embeddingService: es as any,
    });

    expect(result.strategy).toBe("embedding");
    expect(result.results).toHaveLength(1);
    expect(result.results[0].id).toBe("vec-1");
  });

  it("should use candidateK = limit*4 when sessionFilter is set", async () => {
    const vs = createMockVectorStore([]);
    vs.isFtsAvailable.mockReturnValue(true);
    vs.ftsSearchL0.mockReturnValue([]);
    const es = createMockEmbeddingService();

    await executeConversationSearch({
      query: "test",
      limit: 5,
      sessionKey: "sk_filter",
      vectorStore: vs as any,
      embeddingService: es as any,
    });

    // With sessionFilter, candidateK = limit * 4 = 20
    expect(vs.searchL0).toHaveBeenCalledWith(expect.any(Float32Array), 20);
    expect(vs.ftsSearchL0).toHaveBeenCalledWith(expect.any(String), 20);
  });
});

// ── Session key filter ──

describe("session key filter", () => {
  it("should filter results by exact session_key match", async () => {
    const vs = createMockVectorStore([
      { record_id: "l0_1", session_key: "sk_a", role: "user", message_text: "Msg A", score: 0.9, recorded_at: "" },
      { record_id: "l0_2", session_key: "sk_b", role: "user", message_text: "Msg B", score: 0.8, recorded_at: "" },
      { record_id: "l0_3", session_key: "sk_a", role: "assistant", message_text: "Reply A", score: 0.7, recorded_at: "" },
    ]);
    const es = createMockEmbeddingService();

    const result = await executeConversationSearch({
      query: "test",
      limit: 10,
      sessionKey: "sk_a",
      vectorStore: vs as any,
      embeddingService: es as any,
    });

    expect(result.results).toHaveLength(2);
    expect(result.results.every((r) => r.session_key === "sk_a")).toBe(true);
  });

  it("should return empty when no results match session filter", async () => {
    const vs = createMockVectorStore([
      { record_id: "l0_1", session_key: "sk_other", role: "user", message_text: "Msg", score: 0.9, recorded_at: "" },
    ]);
    const es = createMockEmbeddingService();

    const result = await executeConversationSearch({
      query: "test",
      limit: 10,
      sessionKey: "sk_nonexistent",
      vectorStore: vs as any,
      embeddingService: es as any,
    });

    expect(result.results).toEqual([]);
  });
});

// ── Limit trimming ──

describe("limit trimming", () => {
  it("should trim results to requested limit", async () => {
    const items = Array.from({ length: 8 }, (_, i) => ({
      record_id: `l0_${i}`,
      session_key: "sk",
      role: "user" as const,
      message_text: `Message ${i}`,
      score: 0.9 - i * 0.01,
      recorded_at: "",
    }));
    const vs = createMockVectorStore(items);
    const es = createMockEmbeddingService();

    const result = await executeConversationSearch({
      query: "test",
      limit: 3,
      vectorStore: vs as any,
      embeddingService: es as any,
    });

    expect(result.results).toHaveLength(3);
    expect(result.total).toBe(3);
  });
});

// ── formatConversationSearchResponse ──

describe("formatConversationSearchResponse", () => {
  it("should return 'no matching' message for empty results", () => {
    const result: ConversationSearchResult = { results: [], total: 0, strategy: "none" };
    expect(formatConversationSearchResponse(result)).toBe("No matching conversation messages found.");
  });

  it("should format results with role, session, date, score", () => {
    const result: ConversationSearchResult = {
      results: [
        { id: "l0_1", session_key: "sk_main", role: "user", content: "What is TypeScript?", score: 0.88, recorded_at: "2026-03-17T10:00:00Z" },
      ],
      total: 1,
      strategy: "hybrid",
    };

    const text = formatConversationSearchResponse(result);
    expect(text).toContain("1 matching message");
    expect(text).toContain("[user]");
    expect(text).toContain("Session: sk_main");
    expect(text).toContain("[2026-03-17T10:00:00Z]");
    expect(text).toContain("(score: 0.880)");
    expect(text).toContain("What is TypeScript?");
  });

  it("should omit date when recorded_at is empty", () => {
    const result: ConversationSearchResult = {
      results: [
        { id: "l0_1", session_key: "sk", role: "assistant", content: "Hello", score: 0.5, recorded_at: "" },
      ],
      total: 1,
      strategy: "embedding",
    };

    const text = formatConversationSearchResponse(result);
    expect(text).not.toContain("[]"); // No empty brackets
    expect(text).toContain("[assistant]");
  });

  it("should format multiple results with separators", () => {
    const result: ConversationSearchResult = {
      results: [
        { id: "l0_1", session_key: "sk", role: "user", content: "First message", score: 0.9, recorded_at: "" },
        { id: "l0_2", session_key: "sk", role: "assistant", content: "Second message", score: 0.8, recorded_at: "" },
      ],
      total: 2,
      strategy: "hybrid",
    };

    const text = formatConversationSearchResponse(result);
    expect(text).toContain("2 matching message");
    expect(text).toContain("---"); // separator
    expect(text).toContain("First message");
    expect(text).toContain("Second message");
  });
});

// ── FTS-only search on L0 (when embeddingService is unavailable) ──

describe("FTS-only L0 search", () => {
  it("should use FTS5 on L0 when embeddingService is undefined and FTS is available", async () => {
    const log = logger();
    const vs = createMockVectorStore();
    vs.isFtsAvailable.mockReturnValue(true);
    vs.ftsSearchL0.mockReturnValue([
      {
        record_id: "fts-l0-1",
        session_key: "sk_fts",
        session_id: "",
        role: "user",
        message_text: "FTS matched conversation",
        score: 0.72,
        recorded_at: "2026-03-17T12:00:00Z",
        timestamp: 0,
      },
    ]);

    const result = await executeConversationSearch({
      query: "FTS matched",
      limit: 5,
      vectorStore: vs as any,
      logger: log,
    });

    expect(result.strategy).toBe("fts");
    expect(result.results).toHaveLength(1);
    expect(result.results[0].id).toBe("fts-l0-1");
    expect(result.results[0].content).toBe("FTS matched conversation");
    expect(result.results[0].session_key).toBe("sk_fts");
    expect(vs.ftsSearchL0).toHaveBeenCalled();
  });

  it("should apply session key filter in FTS-only mode", async () => {
    const vs = createMockVectorStore();
    vs.isFtsAvailable.mockReturnValue(true);
    vs.ftsSearchL0.mockReturnValue([
      { record_id: "f1", session_key: "sk_a", session_id: "", role: "user", message_text: "Msg A", score: 0.9, recorded_at: "", timestamp: 0 },
      { record_id: "f2", session_key: "sk_b", session_id: "", role: "user", message_text: "Msg B", score: 0.8, recorded_at: "", timestamp: 0 },
      { record_id: "f3", session_key: "sk_a", session_id: "", role: "assistant", message_text: "Reply A", score: 0.7, recorded_at: "", timestamp: 0 },
    ]);

    const result = await executeConversationSearch({
      query: "Msg",
      limit: 10,
      sessionKey: "sk_a",
      vectorStore: vs as any,
    });

    expect(result.strategy).toBe("fts");
    expect(result.results).toHaveLength(2);
    expect(result.results.every((r) => r.session_key === "sk_a")).toBe(true);
  });

  it("should trim FTS L0 results to limit", async () => {
    const vs = createMockVectorStore();
    vs.isFtsAvailable.mockReturnValue(true);
    vs.ftsSearchL0.mockReturnValue(
      Array.from({ length: 8 }, (_, i) => ({
        record_id: `f${i}`, session_key: "sk", session_id: "", role: "user",
        message_text: `Message ${i}`, score: 0.9 - i * 0.05,
        recorded_at: "", timestamp: 0,
      })),
    );

    const result = await executeConversationSearch({
      query: "Message",
      limit: 3,
      vectorStore: vs as any,
    });

    expect(result.results).toHaveLength(3);
    expect(result.total).toBe(3);
  });

  it("should return empty when FTS query produces no usable tokens", async () => {
    const vs = createMockVectorStore();
    vs.isFtsAvailable.mockReturnValue(true);

    const result = await executeConversationSearch({
      query: "!@#$%^",
      limit: 5,
      vectorStore: vs as any,
    });

    expect(result.results).toEqual([]);
  });

  it("should return empty gracefully when ftsSearchL0 throws", async () => {
    const log = logger();
    const vs = createMockVectorStore();
    vs.isFtsAvailable.mockReturnValue(true);
    vs.ftsSearchL0.mockImplementation(() => { throw new Error("FTS L0 corrupt"); });

    const result = await executeConversationSearch({
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

    const result = await executeConversationSearch({
      query: "test",
      limit: 5,
      vectorStore: vs as any,
    });

    expect(result.strategy).toBe("none");
    expect(result.message).toContain("Embedding service is not configured");
    expect(result.results).toEqual([]);
  });

  it("should display message via formatConversationSearchResponse when message is set", () => {
    const result: ConversationSearchResult = {
      results: [],
      total: 0,
      strategy: "none",
      message: "Custom L0 warning message",
    };
    expect(formatConversationSearchResponse(result)).toBe("Custom L0 warning message");
  });
});
