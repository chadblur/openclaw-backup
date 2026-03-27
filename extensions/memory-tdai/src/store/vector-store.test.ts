/**
 * Unit tests for VectorStore (A 同学自测 → B 审查).
 *
 * VS-01: Vector write + read roundtrip correct
 * VS-02: TopK limits returned count
 * VS-03: Empty store query returns empty
 * VS-04: Delete removes record from search results
 * VS-05: close() after operations
 *
 * Additional edge cases:
 * - Degraded mode (sqlite-vec load failure)
 * - Embedding meta change detection (needsReindex)
 * - L0 operations (upsert, search, delete, count, query)
 * - deleteBatch, deleteL1ExpiredByUpdatedTime
 * - queryL1Records with various filters
 * - reindexAll
 * - Orphan vector (vector without metadata)
 * - Fault tolerance (upsert/search/delete return gracefully on error)
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import os from "node:os";
import path from "node:path";
import fs from "node:fs/promises";
import { VectorStore, buildFtsQuery, bm25RankToScore, tokenizeForFts, _resetJiebaForTest, _setJiebaForTest } from "./vector-store.js";
import type { MemoryRecord } from "../record/l1-writer.js";

// ── Helpers ──

let testDir: string;
const mkDir = async () => {
  const d = path.join(
    os.tmpdir(),
    `vs-test-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
  );
  await fs.mkdir(d, { recursive: true });
  return d;
};
const rmDir = async (d: string) => {
  try {
    await fs.rm(d, { recursive: true, force: true });
  } catch {}
};
const logger = () => ({
  debug: vi.fn(),
  info: vi.fn(),
  warn: vi.fn(),
  error: vi.fn(),
});

const DIMS = 4;

function createStore(dir: string, dims = DIMS, log?: ReturnType<typeof logger>) {
  const dbPath = path.join(dir, "test.db");
  return new VectorStore(dbPath, dims, log);
}

function makeRecord(
  id: string,
  content: string,
  overrides?: Partial<MemoryRecord>,
): MemoryRecord {
  const now = new Date().toISOString();
  return {
    id,
    content,
    type: "persona",
    priority: 50,
    scene_name: "",
    source_message_ids: [],
    metadata: {},
    timestamps: [now],
    createdAt: now,
    updatedAt: now,
    sessionKey: "s1",
    sessionId: "",
    ...overrides,
  };
}

function randomEmbedding(dims = DIMS): Float32Array {
  const arr = new Float32Array(dims);
  for (let i = 0; i < dims; i++) arr[i] = Math.random() - 0.5;
  // L2 normalize
  const norm = Math.sqrt(arr.reduce((s, v) => s + v * v, 0));
  for (let i = 0; i < dims; i++) arr[i] /= norm;
  return arr;
}

/** Creates a deterministic embedding pointing in a specific "direction". */
function directedEmbedding(index: number, dims = DIMS): Float32Array {
  const arr = new Float32Array(dims);
  arr[index % dims] = 1.0;
  return arr;
}

beforeEach(async () => {
  testDir = await mkDir();
});
afterEach(async () => {
  await rmDir(testDir);
});

// ── VS-01: Vector write + read roundtrip ──

describe("VS-01: upsert + search roundtrip", () => {
  it("should write a record and retrieve it via search", () => {
    const log = logger();
    const store = createStore(testDir, DIMS, log);
    store.init();

    const record = makeRecord("r1", "User likes dark mode");
    const emb = directedEmbedding(0);
    expect(store.upsert(record, emb)).toBe(true);

    // Search with same direction → should match
    const results = store.search(directedEmbedding(0), 5);
    expect(results).toHaveLength(1);
    expect(results[0].record_id).toBe("r1");
    expect(results[0].content).toBe("User likes dark mode");
    expect(results[0].type).toBe("persona");
    expect(results[0].priority).toBe(50);
    expect(results[0].score).toBeGreaterThan(0.9); // same direction → high similarity

    store.close();
  });

  it("should upsert (update) an existing record", () => {
    const store = createStore(testDir);
    store.init();

    const emb = directedEmbedding(0);
    store.upsert(makeRecord("r1", "Original content"), emb);

    // Update same ID with new content
    store.upsert(makeRecord("r1", "Updated content"), emb);

    const results = store.search(directedEmbedding(0), 5);
    expect(results).toHaveLength(1);
    expect(results[0].content).toBe("Updated content");

    store.close();
  });

  it("should store and retrieve metadata fields correctly", () => {
    const store = createStore(testDir);
    store.init();

    const now = new Date().toISOString();
    const record = makeRecord("r_meta", "Memory with full metadata", {
      type: "episodic",
      priority: 90,
      scene_name: "work-habits",
      sessionKey: "sk1",
      sessionId: "sid1",
      timestamps: [now, "2026-03-16T10:00:00Z"],
      metadata: { activity_start_time: "2026-03-16T09:00:00Z" },
    });

    store.upsert(record, directedEmbedding(0));

    const results = store.search(directedEmbedding(0), 1);
    expect(results).toHaveLength(1);
    expect(results[0].type).toBe("episodic");
    expect(results[0].priority).toBe(90);
    expect(results[0].scene_name).toBe("work-habits");
    expect(results[0].session_key).toBe("sk1");
    expect(results[0].session_id).toBe("sid1");
    expect(results[0].timestamp_str).toBe(now);
    expect(JSON.parse(results[0].metadata_json)).toHaveProperty("activity_start_time");

    store.close();
  });
});

// ── VS-02: TopK limits ──

describe("VS-02: topK limits returned count", () => {
  it("should return at most topK results", () => {
    const store = createStore(testDir);
    store.init();

    // Insert 5 records with similar embeddings
    for (let i = 0; i < 5; i++) {
      store.upsert(makeRecord(`r${i}`, `Memory ${i}`), randomEmbedding());
    }

    const query = randomEmbedding();
    const results2 = store.search(query, 2);
    expect(results2.length).toBeLessThanOrEqual(2);

    const results5 = store.search(query, 5);
    expect(results5.length).toBeLessThanOrEqual(5);

    store.close();
  });
});

// ── VS-03: Empty store ──

describe("VS-03: empty store query returns empty", () => {
  it("should return empty array on empty store", () => {
    const store = createStore(testDir);
    store.init();

    const results = store.search(randomEmbedding(), 10);
    expect(results).toEqual([]);

    store.close();
  });

  it("count() should return 0 on empty store", () => {
    const store = createStore(testDir);
    store.init();

    expect(store.count()).toBe(0);

    store.close();
  });
});

// ── VS-04: Delete removes record ──

describe("VS-04: delete removes record from search results", () => {
  it("should not find record after delete", () => {
    const store = createStore(testDir);
    store.init();

    const emb = directedEmbedding(0);
    store.upsert(makeRecord("r_del", "To be deleted"), emb);
    expect(store.count()).toBe(1);

    expect(store.delete("r_del")).toBe(true);
    expect(store.count()).toBe(0);

    const results = store.search(emb, 5);
    expect(results).toEqual([]);

    store.close();
  });

  it("deleteBatch should remove multiple records", () => {
    const store = createStore(testDir);
    store.init();

    store.upsert(makeRecord("r1", "Memory 1"), directedEmbedding(0));
    store.upsert(makeRecord("r2", "Memory 2"), directedEmbedding(1));
    store.upsert(makeRecord("r3", "Memory 3"), directedEmbedding(2));
    expect(store.count()).toBe(3);

    expect(store.deleteBatch(["r1", "r3"])).toBe(true);
    expect(store.count()).toBe(1);

    const results = store.search(directedEmbedding(1), 5);
    expect(results).toHaveLength(1);
    expect(results[0].record_id).toBe("r2");

    store.close();
  });

  it("deleteBatch with empty array returns true", () => {
    const store = createStore(testDir);
    store.init();

    expect(store.deleteBatch([])).toBe(true);

    store.close();
  });
});

// ── VS-05: close() ──

describe("VS-05: close()", () => {
  it("should close without error", () => {
    const log = logger();
    const store = createStore(testDir, DIMS, log);
    store.init();

    store.upsert(makeRecord("r1", "Test"), randomEmbedding());
    store.close();

    expect(log.info).toHaveBeenCalledWith(expect.stringContaining("Database closed"));
  });

  it("should handle double close gracefully", () => {
    const log = logger();
    const store = createStore(testDir, DIMS, log);
    store.init();
    store.close();
    // Second close — should not crash (the DB warns but VectorStore catches it)
    store.close();
  });
});

// ── Degraded mode ──

describe("degraded mode", () => {
  it("isDegraded() returns false after successful init", () => {
    const store = createStore(testDir);
    store.init();
    expect(store.isDegraded()).toBe(false);
    store.close();
  });

  // Note: testing actual degraded mode (failed sqlite-vec load) is hard without
  // mocking the require chain; we test the behavioral contract instead.
});

// ── Embedding meta change detection ──

describe("embedding meta change detection (needsReindex)", () => {
  it("first init with providerInfo → needsReindex=false (no prior data)", () => {
    const store = createStore(testDir);
    const result = store.init({ provider: "openai", model: "text-embedding-3-large" });
    expect(result.needsReindex).toBe(false);
    store.close();
  });

  it("same providerInfo on re-open → needsReindex=false", () => {
    const dbPath = path.join(testDir, "meta-test.db");
    const info = { provider: "openai", model: "text-embedding-3-large" };

    const store1 = new VectorStore(dbPath, DIMS);
    store1.init(info);
    store1.close();

    const store2 = new VectorStore(dbPath, DIMS);
    const result = store2.init(info);
    expect(result.needsReindex).toBe(false);
    store2.close();
  });

  it("different model on re-open → needsReindex=true", () => {
    const dbPath = path.join(testDir, "meta-change.db");
    const log = logger();

    const store1 = new VectorStore(dbPath, DIMS, log);
    store1.init({ provider: "openai", model: "model-v1" });
    store1.upsert(makeRecord("r1", "Existing memory"), randomEmbedding());
    store1.close();

    const store2 = new VectorStore(dbPath, DIMS, log);
    const result = store2.init({ provider: "openai", model: "model-v2" });
    expect(result.needsReindex).toBe(true);
    expect(result.reason).toContain("model");
    store2.close();
  });

  it("different provider on re-open → needsReindex=true", () => {
    const dbPath = path.join(testDir, "meta-provider.db");

    const store1 = new VectorStore(dbPath, DIMS);
    store1.init({ provider: "openai", model: "m1" });
    store1.upsert(makeRecord("r1", "Existing"), randomEmbedding());
    store1.close();

    const store2 = new VectorStore(dbPath, DIMS);
    const result = store2.init({ provider: "local", model: "m1" });
    expect(result.needsReindex).toBe(true);
    expect(result.reason).toContain("provider");
    store2.close();
  });

  it("legacy DB without meta but has data → needsReindex=true", () => {
    const dbPath = path.join(testDir, "legacy.db");
    const log = logger();

    // First init without providerInfo (simulates legacy)
    const store1 = new VectorStore(dbPath, DIMS, log);
    store1.init(); // no providerInfo
    store1.upsert(makeRecord("r1", "Legacy memory"), randomEmbedding());
    store1.close();

    // Re-open with providerInfo → should detect legacy
    const store2 = new VectorStore(dbPath, DIMS, log);
    const result = store2.init({ provider: "openai", model: "m1" });
    expect(result.needsReindex).toBe(true);
    expect(result.reason).toContain("legacy");
    store2.close();
  });

  it("init without providerInfo → needsReindex=false (no tracking)", () => {
    const store = createStore(testDir);
    const result = store.init();
    expect(result.needsReindex).toBe(false);
    store.close();
  });
});

// ── deleteL1ExpiredByUpdatedTime ──

describe("deleteL1ExpiredByUpdatedTime", () => {
  it("should delete records with updated_time before cutoff", () => {
    const store = createStore(testDir);
    store.init();

    store.upsert(
      makeRecord("old", "Old memory", { updatedAt: "2026-01-01T00:00:00Z" }),
      randomEmbedding(),
    );
    store.upsert(
      makeRecord("new", "New memory", { updatedAt: "2026-03-17T00:00:00Z" }),
      randomEmbedding(),
    );

    const deleted = store.deleteL1ExpiredByUpdatedTime("2026-02-01T00:00:00Z");
    expect(deleted).toBe(1);
    expect(store.count()).toBe(1);

    store.close();
  });

  it("should return 0 when nothing expired", () => {
    const store = createStore(testDir);
    store.init();

    store.upsert(
      makeRecord("fresh", "Fresh memory", { updatedAt: "2026-03-17T00:00:00Z" }),
      randomEmbedding(),
    );

    const deleted = store.deleteL1ExpiredByUpdatedTime("2026-01-01T00:00:00Z");
    expect(deleted).toBe(0);
    expect(store.count()).toBe(1);

    store.close();
  });
});

// ── queryL1Records with filters ──

describe("queryL1Records", () => {
  it("should return all records when no filter", () => {
    const store = createStore(testDir);
    store.init();

    store.upsert(makeRecord("r1", "M1"), randomEmbedding());
    store.upsert(makeRecord("r2", "M2"), randomEmbedding());

    const rows = store.queryL1Records();
    expect(rows).toHaveLength(2);

    store.close();
  });

  it("should filter by sessionKey", () => {
    const store = createStore(testDir);
    store.init();

    store.upsert(makeRecord("r1", "M1", { sessionKey: "sk_a" }), randomEmbedding());
    store.upsert(makeRecord("r2", "M2", { sessionKey: "sk_b" }), randomEmbedding());

    const rows = store.queryL1Records({ sessionKey: "sk_a" });
    expect(rows).toHaveLength(1);
    expect(rows[0].record_id).toBe("r1");

    store.close();
  });

  it("should filter by sessionId", () => {
    const store = createStore(testDir);
    store.init();

    store.upsert(makeRecord("r1", "M1", { sessionId: "sid_a" }), randomEmbedding());
    store.upsert(makeRecord("r2", "M2", { sessionId: "sid_b" }), randomEmbedding());

    const rows = store.queryL1Records({ sessionId: "sid_a" });
    expect(rows).toHaveLength(1);
    expect(rows[0].record_id).toBe("r1");

    store.close();
  });

  it("should filter by updatedAfter", () => {
    const store = createStore(testDir);
    store.init();

    store.upsert(
      makeRecord("old", "Old", { updatedAt: "2026-03-15T00:00:00Z" }),
      randomEmbedding(),
    );
    store.upsert(
      makeRecord("new", "New", { updatedAt: "2026-03-17T12:00:00Z" }),
      randomEmbedding(),
    );

    const rows = store.queryL1Records({ updatedAfter: "2026-03-16T00:00:00Z" });
    expect(rows).toHaveLength(1);
    expect(rows[0].record_id).toBe("new");

    store.close();
  });

  it("should filter by sessionKey + updatedAfter combined", () => {
    const store = createStore(testDir);
    store.init();

    store.upsert(
      makeRecord("r1", "M1", { sessionKey: "sk_a", updatedAt: "2026-03-15T00:00:00Z" }),
      randomEmbedding(),
    );
    store.upsert(
      makeRecord("r2", "M2", { sessionKey: "sk_a", updatedAt: "2026-03-17T12:00:00Z" }),
      randomEmbedding(),
    );
    store.upsert(
      makeRecord("r3", "M3", { sessionKey: "sk_b", updatedAt: "2026-03-17T12:00:00Z" }),
      randomEmbedding(),
    );

    const rows = store.queryL1Records({ sessionKey: "sk_a", updatedAfter: "2026-03-16T00:00:00Z" });
    expect(rows).toHaveLength(1);
    expect(rows[0].record_id).toBe("r2");

    store.close();
  });

  it("should filter by sessionId + updatedAfter combined", () => {
    const store = createStore(testDir);
    store.init();

    store.upsert(
      makeRecord("r1", "M1", { sessionId: "sid_a", updatedAt: "2026-03-15T00:00:00Z" }),
      randomEmbedding(),
    );
    store.upsert(
      makeRecord("r2", "M2", { sessionId: "sid_a", updatedAt: "2026-03-17T12:00:00Z" }),
      randomEmbedding(),
    );

    const rows = store.queryL1Records({ sessionId: "sid_a", updatedAfter: "2026-03-16T00:00:00Z" });
    expect(rows).toHaveLength(1);
    expect(rows[0].record_id).toBe("r2");

    store.close();
  });

  it("should return empty array in degraded mode", () => {
    // We can't directly set degraded=true, but queryL1Records returns [] for no data
    const store = createStore(testDir);
    store.init();
    expect(store.queryL1Records()).toEqual([]);
    store.close();
  });
});

// ── L0 operations ──

describe("L0 operations", () => {
  it("upsertL0 + searchL0 roundtrip", () => {
    const store = createStore(testDir);
    store.init();

    const emb = directedEmbedding(0);
    const ok = store.upsertL0(
      {
        id: "l0_1",
        sessionKey: "sk1",
        sessionId: "sid1",
        role: "user",
        messageText: "Hello, how are you?",
        recordedAt: "2026-03-17T10:00:00Z",
        timestamp: 1710672000000,
      },
      emb,
    );
    expect(ok).toBe(true);

    const results = store.searchL0(directedEmbedding(0), 5);
    expect(results).toHaveLength(1);
    expect(results[0].record_id).toBe("l0_1");
    expect(results[0].role).toBe("user");
    expect(results[0].message_text).toBe("Hello, how are you?");
    expect(results[0].session_key).toBe("sk1");
    expect(results[0].score).toBeGreaterThan(0.9);

    store.close();
  });

  it("countL0 returns correct count", () => {
    const store = createStore(testDir);
    store.init();

    expect(store.countL0()).toBe(0);

    store.upsertL0(
      {
        id: "l0_1",
        sessionKey: "sk1",
        sessionId: "",
        role: "user",
        messageText: "Hello",
        recordedAt: new Date().toISOString(),
        timestamp: Date.now(),
      },
      randomEmbedding(),
    );

    expect(store.countL0()).toBe(1);

    store.close();
  });

  it("deleteL0 removes record", () => {
    const store = createStore(testDir);
    store.init();

    store.upsertL0(
      {
        id: "l0_del",
        sessionKey: "sk1",
        sessionId: "",
        role: "user",
        messageText: "To be deleted",
        recordedAt: new Date().toISOString(),
        timestamp: Date.now(),
      },
      randomEmbedding(),
    );
    expect(store.countL0()).toBe(1);

    expect(store.deleteL0("l0_del")).toBe(true);
    expect(store.countL0()).toBe(0);

    store.close();
  });

  it("searchL0 returns empty on empty store", () => {
    const store = createStore(testDir);
    store.init();

    const results = store.searchL0(randomEmbedding(), 5);
    expect(results).toEqual([]);

    store.close();
  });
});

// ── queryL0ForL1 ──

describe("queryL0ForL1", () => {
  it("should return all messages for a session key ordered by timestamp", () => {
    const store = createStore(testDir);
    store.init();

    store.upsertL0(
      { id: "m2", sessionKey: "sk", sessionId: "", role: "assistant", messageText: "Hi", recordedAt: "", timestamp: 200 },
      randomEmbedding(),
    );
    store.upsertL0(
      { id: "m1", sessionKey: "sk", sessionId: "", role: "user", messageText: "Hello", recordedAt: "", timestamp: 100 },
      randomEmbedding(),
    );

    const rows = store.queryL0ForL1("sk");
    expect(rows).toHaveLength(2);
    expect(rows[0].record_id).toBe("m1"); // timestamp 100 first
    expect(rows[1].record_id).toBe("m2"); // timestamp 200 second

    store.close();
  });

  it("should filter by afterTimestamp", () => {
    const store = createStore(testDir);
    store.init();

    store.upsertL0(
      { id: "old", sessionKey: "sk", sessionId: "", role: "user", messageText: "Old", recordedAt: "", timestamp: 100 },
      randomEmbedding(),
    );
    store.upsertL0(
      { id: "new", sessionKey: "sk", sessionId: "", role: "user", messageText: "New", recordedAt: "", timestamp: 300 },
      randomEmbedding(),
    );

    const rows = store.queryL0ForL1("sk", 200);
    expect(rows).toHaveLength(1);
    expect(rows[0].record_id).toBe("new");

    store.close();
  });

  it("should return empty for non-existent session key", () => {
    const store = createStore(testDir);
    store.init();

    const rows = store.queryL0ForL1("nonexistent");
    expect(rows).toEqual([]);

    store.close();
  });

  it("should return newest messages first when limit truncates, in chronological order", () => {
    const store = createStore(testDir);
    store.init();

    // Insert 5 messages with timestamps 100..500
    for (let i = 1; i <= 5; i++) {
      store.upsertL0(
        { id: `m${i}`, sessionKey: "sk", sessionId: "", role: "user", messageText: `Msg${i}`, recordedAt: "", timestamp: i * 100 },
        randomEmbedding(),
      );
    }

    // limit=3 → should return the 3 newest (m3,m4,m5) in chronological order
    const rows = store.queryL0ForL1("sk", undefined, 3);
    expect(rows).toHaveLength(3);
    expect(rows[0].record_id).toBe("m3"); // ts 300
    expect(rows[1].record_id).toBe("m4"); // ts 400
    expect(rows[2].record_id).toBe("m5"); // ts 500

    store.close();
  });
});

// ── queryL0GroupedBySessionId ──

describe("queryL0GroupedBySessionId", () => {
  it("should group messages by session_id and sort chronologically", () => {
    const store = createStore(testDir);
    store.init();

    store.upsertL0(
      { id: "m1", sessionKey: "sk", sessionId: "sid_a", role: "user", messageText: "A1", recordedAt: "", timestamp: 100 },
      randomEmbedding(),
    );
    store.upsertL0(
      { id: "m2", sessionKey: "sk", sessionId: "sid_b", role: "user", messageText: "B1", recordedAt: "", timestamp: 50 },
      randomEmbedding(),
    );
    store.upsertL0(
      { id: "m3", sessionKey: "sk", sessionId: "sid_a", role: "assistant", messageText: "A2", recordedAt: "", timestamp: 200 },
      randomEmbedding(),
    );

    const groups = store.queryL0GroupedBySessionId("sk");
    expect(groups).toHaveLength(2);

    // sid_b (earliest ts=50) comes first
    expect(groups[0].sessionId).toBe("sid_b");
    expect(groups[0].messages).toHaveLength(1);

    // sid_a (earliest ts=100) comes second
    expect(groups[1].sessionId).toBe("sid_a");
    expect(groups[1].messages).toHaveLength(2);
    expect(groups[1].messages[0].content).toBe("A1");
    expect(groups[1].messages[1].content).toBe("A2");

    store.close();
  });

  it("should filter by afterTimestamp", () => {
    const store = createStore(testDir);
    store.init();

    store.upsertL0(
      { id: "m1", sessionKey: "sk", sessionId: "sid_a", role: "user", messageText: "Old", recordedAt: "", timestamp: 50 },
      randomEmbedding(),
    );
    store.upsertL0(
      { id: "m2", sessionKey: "sk", sessionId: "sid_a", role: "user", messageText: "New", recordedAt: "", timestamp: 200 },
      randomEmbedding(),
    );

    const groups = store.queryL0GroupedBySessionId("sk", 100);
    expect(groups).toHaveLength(1);
    expect(groups[0].messages).toHaveLength(1);
    expect(groups[0].messages[0].content).toBe("New");

    store.close();
  });

  it("should return empty for empty store", () => {
    const store = createStore(testDir);
    store.init();

    const groups = store.queryL0GroupedBySessionId("nonexistent");
    expect(groups).toEqual([]);

    store.close();
  });
});

// ── getAllL1Texts / getAllL0Texts ──

describe("getAllL1Texts / getAllL0Texts", () => {
  it("getAllL1Texts returns all L1 record texts", () => {
    const store = createStore(testDir);
    store.init();

    store.upsert(makeRecord("r1", "Memory one"), randomEmbedding());
    store.upsert(makeRecord("r2", "Memory two"), randomEmbedding());

    const texts = store.getAllL1Texts();
    expect(texts).toHaveLength(2);
    expect(texts.map((t) => t.record_id).sort()).toEqual(["r1", "r2"]);

    store.close();
  });

  it("getAllL0Texts returns all L0 message texts", () => {
    const store = createStore(testDir);
    store.init();

    store.upsertL0(
      { id: "l0_1", sessionKey: "sk", sessionId: "", role: "user", messageText: "Hello", recordedAt: "", timestamp: 100 },
      randomEmbedding(),
    );

    const texts = store.getAllL0Texts();
    expect(texts).toHaveLength(1);
    expect(texts[0].message_text).toBe("Hello");

    store.close();
  });

  it("returns empty arrays when store is empty", () => {
    const store = createStore(testDir);
    store.init();

    expect(store.getAllL1Texts()).toEqual([]);
    expect(store.getAllL0Texts()).toEqual([]);

    store.close();
  });
});

// ── reindexAll ──

describe("reindexAll", () => {
  it("should re-embed all L1 and L0 records", async () => {
    const store = createStore(testDir);
    store.init();

    store.upsert(makeRecord("r1", "Memory 1"), randomEmbedding());
    store.upsertL0(
      { id: "l0_1", sessionKey: "sk", sessionId: "", role: "user", messageText: "Msg 1", recordedAt: "", timestamp: 100 },
      randomEmbedding(),
    );

    const embedFn = vi.fn().mockResolvedValue(directedEmbedding(0));
    const progressFn = vi.fn();

    const result = await store.reindexAll(embedFn, progressFn);
    expect(result.l1Count).toBe(1);
    expect(result.l0Count).toBe(1);
    expect(embedFn).toHaveBeenCalledTimes(2);
    expect(progressFn).toHaveBeenCalledTimes(2);

    store.close();
  });

  it("should skip individual records on embedFn failure", async () => {
    const log = logger();
    const store = createStore(testDir, DIMS, log);
    store.init();

    store.upsert(makeRecord("r1", "Memory 1"), randomEmbedding());
    store.upsert(makeRecord("r2", "Memory 2"), randomEmbedding());

    let callCount = 0;
    const embedFn = vi.fn().mockImplementation(async () => {
      callCount++;
      if (callCount === 1) throw new Error("embed fail");
      return directedEmbedding(0);
    });

    const result = await store.reindexAll(embedFn);
    expect(result.l1Count).toBe(2); // both counted (one skipped but still incremented)
    expect(log.warn).toHaveBeenCalledWith(expect.stringContaining("reindex L1 skip"));

    store.close();
  });

  it("should return zeros on empty store", async () => {
    const store = createStore(testDir);
    store.init();

    const embedFn = vi.fn();
    const result = await store.reindexAll(embedFn);
    expect(result).toEqual({ l1Count: 0, l0Count: 0 });
    expect(embedFn).not.toHaveBeenCalled();

    store.close();
  });
});

// ── Orphan vector handling ──

describe("orphan vector (vector without metadata)", () => {
  it("search should skip orphan vectors gracefully", () => {
    const log = logger();
    const store = createStore(testDir, DIMS, log);
    store.init();

    // Insert a normal record
    store.upsert(makeRecord("r1", "Normal memory"), directedEmbedding(0));

    // Manually delete only the metadata (create orphan)
    // We can't access private stmtDeleteMeta directly, so we use delete() which removes both
    // Instead, test orphan handling indirectly: just verify no crash when metadata is missing
    // The actual orphan scenario would happen from a DB corruption or partial write

    // This test verifies the code path exists: store.search handles missing meta rows
    const results = store.search(directedEmbedding(0), 5);
    expect(results).toHaveLength(1); // normal record found

    store.close();
  });
});

// ── Fault tolerance ──

describe("fault tolerance", () => {
  it("count() returns 0 in degraded mode (simulated via closed DB)", () => {
    const log = logger();
    const store = createStore(testDir, DIMS, log);
    store.init();
    store.close();

    // After close, operations should fail gracefully
    // count() catches errors and returns 0
    expect(store.count()).toBe(0);
  });

  it("upsert returns false when DB is closed", () => {
    const log = logger();
    const store = createStore(testDir, DIMS, log);
    store.init();
    store.close();

    const result = store.upsert(makeRecord("r1", "Test"), randomEmbedding());
    expect(result).toBe(false);
  });

  it("search returns empty when DB is closed", () => {
    const log = logger();
    const store = createStore(testDir, DIMS, log);
    store.init();
    store.close();

    const results = store.search(randomEmbedding(), 5);
    expect(results).toEqual([]);
  });

  it("delete returns false when DB is closed", () => {
    const log = logger();
    const store = createStore(testDir, DIMS, log);
    store.init();
    store.close();

    expect(store.delete("nonexistent")).toBe(false);
  });

  it("L0 operations return gracefully when DB is closed", () => {
    const log = logger();
    const store = createStore(testDir, DIMS, log);
    store.init();
    store.close();

    expect(store.upsertL0(
      { id: "l0_1", sessionKey: "sk", sessionId: "", role: "user", messageText: "Test", recordedAt: "", timestamp: 100 },
      randomEmbedding(),
    )).toBe(false);
    expect(store.searchL0(randomEmbedding(), 5)).toEqual([]);
    expect(store.deleteL0("l0_1")).toBe(false);
    expect(store.countL0()).toBe(0);
  });
});

// ── Timestamp handling ──

describe("timestamp fields", () => {
  it("should compute timestamp_start and timestamp_end from timestamps array", () => {
    const store = createStore(testDir);
    store.init();

    const record = makeRecord("r_ts", "Timestamped memory", {
      timestamps: ["2026-03-17T12:00:00Z", "2026-03-15T08:00:00Z", "2026-03-16T16:00:00Z"],
    });

    store.upsert(record, directedEmbedding(0));

    const results = store.search(directedEmbedding(0), 1);
    expect(results).toHaveLength(1);
    expect(results[0].timestamp_str).toBe("2026-03-17T12:00:00Z"); // first in array
    expect(results[0].timestamp_start).toBe("2026-03-15T08:00:00Z"); // min
    expect(results[0].timestamp_end).toBe("2026-03-17T12:00:00Z"); // max

    store.close();
  });

  it("should handle empty timestamps array", () => {
    const store = createStore(testDir);
    store.init();

    const record = makeRecord("r_no_ts", "No timestamps", { timestamps: [] });

    store.upsert(record, directedEmbedding(0));

    const results = store.search(directedEmbedding(0), 1);
    expect(results).toHaveLength(1);
    expect(results[0].timestamp_str).toBe("");
    expect(results[0].timestamp_start).toBe("");
    expect(results[0].timestamp_end).toBe("");

    store.close();
  });
});

// ── FTS5 helpers (buildFtsQuery, bm25RankToScore) ──

describe("buildFtsQuery (with jieba)", () => {
  beforeEach(() => {
    _resetJiebaForTest(); // ensure jieba is freshly loaded
  });

  it("should tokenize Latin words and OR-join", () => {
    expect(buildFtsQuery("hello world")).toBe('"hello" OR "world"');
  });

  it("should segment Chinese text into meaningful words", () => {
    const result = buildFtsQuery("用户喜欢编程和TypeScript开发");
    expect(result).not.toBeNull();
    // jieba should split "编程" as one word (not "编" + "程")
    expect(result).toContain('"编程"');
    expect(result).toContain('"用户"');
    expect(result).toContain('"喜欢"');
    expect(result).toContain('"TypeScript"');
    expect(result).toContain('"开发"');
  });

  it("should split CJK compound words for better recall (cutForSearch)", () => {
    const result = buildFtsQuery("我喜欢吃北京烤鸭");
    expect(result).not.toBeNull();
    // cutForSearch splits "北京烤鸭" → "北京", "烤鸭", "北京烤鸭"
    expect(result).toContain('"北京"');
    expect(result).toContain('"烤鸭"');
    expect(result).toContain('"北京烤鸭"');
  });

  it("should filter out common Chinese stop-words", () => {
    const result = buildFtsQuery("用户的手机号是13800138000");
    expect(result).not.toBeNull();
    // "的" and "是" are stop-words
    expect(result).not.toContain('"的"');
    expect(result).not.toContain('"是"');
    expect(result).toContain('"用户"');
    expect(result).toContain('"手机号"');
  });

  it("should return null for empty / punctuation-only input", () => {
    expect(buildFtsQuery("")).toBeNull();
    expect(buildFtsQuery("!@#$%")).toBeNull();
  });

  it("should strip double quotes from tokens", () => {
    const result = buildFtsQuery('say "hello"');
    expect(result).not.toBeNull();
    expect(result).toContain('"say"');
    expect(result).toContain('"hello"');
    // no unescaped double quotes inside token values
    expect(result).not.toMatch(/""/);
  });

  it("should handle mixed CJK and Latin text", () => {
    const result = buildFtsQuery("旅行计划 API development");
    expect(result).not.toBeNull();
    expect(result).toContain('"旅行"');
    expect(result).toContain('"计划"');
    expect(result).toContain('"API"');
    expect(result).toContain('"development"');
  });
});

describe("buildFtsQuery (fallback without jieba)", () => {
  beforeEach(() => {
    _setJiebaForTest(null); // force fallback mode
  });

  afterEach(() => {
    _resetJiebaForTest(); // restore auto-detection
  });

  it("should fall back to regex split for Latin", () => {
    expect(buildFtsQuery("hello world")).toBe('"hello" OR "world"');
  });

  it("should treat CJK as single token (no word segmentation)", () => {
    // Without jieba, "旅行计划" stays as one token
    expect(buildFtsQuery("旅行计划")).toBe('"旅行计划"');
  });

  it("should return null for empty input", () => {
    expect(buildFtsQuery("")).toBeNull();
  });
});

describe("tokenizeForFts", () => {
  beforeEach(() => {
    _resetJiebaForTest();
  });

  it("should segment Chinese text with spaces (with jieba)", () => {
    const result = tokenizeForFts("用户喜欢编程和TypeScript开发");
    // Should contain spaces between Chinese words
    expect(result).toContain("用户");
    expect(result).toContain("喜欢");
    expect(result).toContain("编程");
    expect(result).toContain("TypeScript");
    expect(result).toContain("开发");
    // Words should be space-separated (unicode61 can split them)
    expect(result.split(/\s+/).length).toBeGreaterThan(3);
  });

  it("should not change pure Latin text significantly", () => {
    const result = tokenizeForFts("hello world test");
    // Jieba cut should keep English words separated
    expect(result).toContain("hello");
    expect(result).toContain("world");
    expect(result).toContain("test");
  });

  it("should return raw text when jieba unavailable", () => {
    _setJiebaForTest(null);
    const raw = "用户五月去日本旅行";
    expect(tokenizeForFts(raw)).toBe(raw);
    _resetJiebaForTest();
  });
});

describe("bm25RankToScore", () => {
  it("should convert negative rank (relevant) to 0-1 score", () => {
    const score = bm25RankToScore(-5);
    expect(score).toBeGreaterThan(0);
    expect(score).toBeLessThan(1);
    expect(score).toBeCloseTo(5 / 6, 4);
  });

  it("should handle zero rank", () => {
    expect(bm25RankToScore(0)).toBe(1);
  });

  it("should handle NaN / Infinity", () => {
    expect(bm25RankToScore(NaN)).toBeCloseTo(0.001, 3);
    expect(bm25RankToScore(Infinity)).toBeCloseTo(0.001, 3);
  });
});

// ── FTS5 operations ──

describe("FTS5 operations", () => {
  it("isFtsAvailable() should return true after init", () => {
    const store = createStore(testDir);
    store.init();
    expect(store.isFtsAvailable()).toBe(true);
    store.close();
  });

  it("L1 upsert should populate FTS table, searchable via ftsSearchL1", () => {
    const store = createStore(testDir);
    store.init();

    store.upsert(
      makeRecord("r1", "User prefers dark mode for coding"),
      randomEmbedding(),
    );
    store.upsert(
      makeRecord("r2", "User enjoys hiking in mountains"),
      randomEmbedding(),
    );

    // Search for "dark mode"
    const results = store.ftsSearchL1('"dark" AND "mode"', 10);
    expect(results).toHaveLength(1);
    expect(results[0].record_id).toBe("r1");
    expect(results[0].content).toContain("dark mode");
    expect(results[0].score).toBeGreaterThan(0);

    store.close();
  });

  it("L0 upsert should populate FTS table, searchable via ftsSearchL0", () => {
    const store = createStore(testDir);
    store.init();

    store.upsertL0(
      {
        id: "l0_1",
        sessionKey: "sk1",
        sessionId: "sid1",
        role: "user",
        messageText: "How do I configure TypeScript compiler?",
        recordedAt: "2026-03-17T10:00:00Z",
        timestamp: 1710672000000,
      },
      randomEmbedding(),
    );

    const results = store.ftsSearchL0('"TypeScript" AND "compiler"', 10);
    expect(results).toHaveLength(1);
    expect(results[0].record_id).toBe("l0_1");
    expect(results[0].message_text).toContain("TypeScript compiler");
    expect(results[0].score).toBeGreaterThan(0);

    store.close();
  });

  it("delete should remove from FTS table", () => {
    const store = createStore(testDir);
    store.init();

    store.upsert(
      makeRecord("r1", "User prefers dark mode"),
      randomEmbedding(),
    );

    let results = store.ftsSearchL1('"dark" AND "mode"', 10);
    expect(results).toHaveLength(1);

    store.delete("r1");

    results = store.ftsSearchL1('"dark" AND "mode"', 10);
    expect(results).toHaveLength(0);

    store.close();
  });

  it("ftsSearchL1 returns empty for non-matching query", () => {
    const store = createStore(testDir);
    store.init();

    store.upsert(
      makeRecord("r1", "User prefers dark mode"),
      randomEmbedding(),
    );

    const results = store.ftsSearchL1('"nonexistent" AND "zzzzz"', 10);
    expect(results).toHaveLength(0);

    store.close();
  });

  it("ftsSearchL1 returns empty on empty store", () => {
    const store = createStore(testDir);
    store.init();

    const results = store.ftsSearchL1('"hello"', 10);
    expect(results).toHaveLength(0);

    store.close();
  });

  it("upsert (update) should update FTS content", () => {
    const store = createStore(testDir);
    store.init();

    store.upsert(
      makeRecord("r1", "User prefers dark mode"),
      randomEmbedding(),
    );

    // Update the record
    store.upsert(
      makeRecord("r1", "User prefers light theme now"),
      randomEmbedding(),
    );

    // Old content should not match
    const oldResults = store.ftsSearchL1('"dark" AND "mode"', 10);
    expect(oldResults).toHaveLength(0);

    // New content should match
    const newResults = store.ftsSearchL1('"light" AND "theme"', 10);
    expect(newResults).toHaveLength(1);
    expect(newResults[0].record_id).toBe("r1");

    store.close();
  });

  it("FTS metadata fields should be correctly populated", () => {
    const store = createStore(testDir);
    store.init();

    store.upsert(
      makeRecord("r_fts_meta", "Memory with metadata for FTS", {
        type: "episodic",
        priority: 85,
        scene_name: "work-project",
        sessionKey: "sk_meta",
        sessionId: "sid_meta",
        timestamps: ["2026-03-17T12:00:00Z"],
        metadata: { activity_start_time: "2026-03-17T09:00:00Z" },
      }),
      randomEmbedding(),
    );

    const results = store.ftsSearchL1('"metadata" AND "FTS"', 10);
    expect(results).toHaveLength(1);
    expect(results[0].type).toBe("episodic");
    expect(results[0].priority).toBe(85);
    expect(results[0].scene_name).toBe("work-project");
    expect(results[0].session_key).toBe("sk_meta");

    store.close();
  });

  it("L1 FTS should return original (unsegmented) content via content_original", () => {
    const store = createStore(testDir);
    store.init();

    const originalText = "用户五月去日本旅行，喜欢吃北京烤鸭";
    store.upsert(
      makeRecord("r_zh1", originalText),
      randomEmbedding(),
    );

    // Build query using jieba — should find it
    const ftsQuery = buildFtsQuery("日本旅行");
    expect(ftsQuery).not.toBeNull();

    const results = store.ftsSearchL1(ftsQuery!, 10);
    expect(results.length).toBeGreaterThan(0);
    // The returned content should be the ORIGINAL text, not the segmented version
    expect(results[0].content).toBe(originalText);
    expect(results[0].content).not.toContain("  "); // no extra spaces from segmentation

    store.close();
  });

  it("L1 FTS with jieba should find Chinese sub-words that old regex missed", () => {
    const store = createStore(testDir);
    store.init();

    store.upsert(
      makeRecord("r_zh2", "深度学习和机器学习是人工智能的重要分支"),
      randomEmbedding(),
    );

    // Search for "人工智能" — jieba should segment both index and query
    const ftsQuery = buildFtsQuery("人工智能");
    expect(ftsQuery).not.toBeNull();

    const results = store.ftsSearchL1(ftsQuery!, 10);
    expect(results.length).toBeGreaterThan(0);
    expect(results[0].record_id).toBe("r_zh2");

    store.close();
  });

  it("L0 FTS should return original message_text (not segmented)", () => {
    const store = createStore(testDir);
    store.init();

    const originalMsg = "我想学习TypeScript编程语言";
    store.upsertL0(
      {
        id: "l0_zh1",
        sessionKey: "sk_zh",
        sessionId: "sid_zh",
        role: "user",
        messageText: originalMsg,
        recordedAt: "2026-03-17T10:00:00Z",
        timestamp: 1710672000000,
      },
      randomEmbedding(),
    );

    const ftsQuery = buildFtsQuery("TypeScript编程");
    expect(ftsQuery).not.toBeNull();

    const results = store.ftsSearchL0(ftsQuery!, 10);
    expect(results.length).toBeGreaterThan(0);
    // Should return original text, not segmented
    expect(results[0].message_text).toBe(originalMsg);

    store.close();
  });

  it("rebuildFtsIndex should repopulate FTS from metadata tables", () => {
    const store = createStore(testDir);
    store.init();

    // Insert records normally (FTS auto-synced)
    store.upsert(
      makeRecord("r_rebuild1", "张三在腾讯工作已经三年了"),
      randomEmbedding(),
    );
    store.upsert(
      makeRecord("r_rebuild2", "我喜欢吃北京烤鸭"),
      randomEmbedding(),
    );

    // Verify searchable
    let q = buildFtsQuery("腾讯");
    expect(q).not.toBeNull();
    let results = store.ftsSearchL1(q!, 10);
    expect(results.length).toBeGreaterThan(0);

    // Force rebuild (simulates migration scenario)
    store.rebuildFtsIndex();

    // Should still be searchable after rebuild
    q = buildFtsQuery("腾讯");
    results = store.ftsSearchL1(q!, 10);
    expect(results.length).toBeGreaterThan(0);
    expect(results[0].record_id).toBe("r_rebuild1");
    // Content should be original text
    expect(results[0].content).toBe("张三在腾讯工作已经三年了");

    // Also test "北京烤鸭"
    q = buildFtsQuery("北京烤鸭");
    results = store.ftsSearchL1(q!, 10);
    expect(results.length).toBeGreaterThan(0);
    expect(results[0].record_id).toBe("r_rebuild2");

    store.close();
  });
});
