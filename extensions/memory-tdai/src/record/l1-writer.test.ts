/**
 * Unit tests for L1 Memory Writer (A 同学自测).
 * LW-01: Memory correctly written to JSONL
 * LW-02: update/merge operations correctly update records
 * LW-03: Daily shard file naming
 *
 * Note: vectorStore dual-write with real embedding is covered by integration tests.
 *       Here we mock vectorStore + embeddingService for dual-write path validation.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { writeMemory, generateMemoryId } from "./l1-writer.js";
import type { ExtractedMemory, MemoryRecord, DedupDecision } from "./l1-writer.js";

let testDir: string;
const mkDir = async () => {
  const d = path.join(os.tmpdir(), `l1-writer-test-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`);
  await fs.mkdir(d, { recursive: true });
  return d;
};
const rmDir = async (d: string) => { try { await fs.rm(d, { recursive: true, force: true }); } catch {} };
const logger = () => ({ debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() });

const sampleMemory: ExtractedMemory = {
  content: "User prefers dark mode for coding",
  type: "persona",
  priority: 80,
  source_message_ids: ["msg_1", "msg_2"],
  metadata: {},
  scene_name: "coding-preferences",
};

function storeDecision(recordId: string): DedupDecision {
  return { record_id: recordId, action: "store", target_ids: [] };
}

function createMockVectorStore() {
  return {
    upsert: vi.fn().mockReturnValue(true),
    deleteBatch: vi.fn().mockReturnValue(true),
    search: vi.fn().mockReturnValue([]),
    count: vi.fn().mockReturnValue(0),
    queryL1Records: vi.fn().mockReturnValue([]),
    close: vi.fn(),
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

beforeEach(async () => { testDir = await mkDir(); });
afterEach(async () => { await rmDir(testDir); });

// ── generateMemoryId ──

describe("generateMemoryId", () => {
  it("should generate unique IDs starting with m_", () => {
    const id1 = generateMemoryId();
    const id2 = generateMemoryId();
    expect(id1).toMatch(/^m_\d+_[0-9a-f]+$/);
    expect(id1).not.toBe(id2);
  });
});

// ── writeMemory ──

describe("writeMemory", () => {
  // ── LW-01: store action ──

  describe("LW-01: store action — append to JSONL", () => {
    it("should write memory record to JSONL file", async () => {
      const recordId = generateMemoryId();
      const record = await writeMemory({
        memory: sampleMemory,
        decision: storeDecision(recordId),
        baseDir: testDir,
        sessionKey: "test-session",
        sessionId: "sid-1",
      });

      expect(record).not.toBeNull();
      expect(record!.id).toBe(recordId);
      expect(record!.content).toBe("User prefers dark mode for coding");
      expect(record!.type).toBe("persona");
      expect(record!.priority).toBe(80);
      expect(record!.sessionKey).toBe("test-session");
      expect(record!.sessionId).toBe("sid-1");
      expect(record!.scene_name).toBe("coding-preferences");
      expect(record!.source_message_ids).toEqual(["msg_1", "msg_2"]);

      // Verify file was written
      const recordsDir = path.join(testDir, "records");
      const files = await fs.readdir(recordsDir);
      expect(files).toHaveLength(1);
      expect(files[0]).toMatch(/^\d{4}-\d{2}-\d{2}\.jsonl$/);

      const raw = await fs.readFile(path.join(recordsDir, files[0]), "utf-8");
      const lines = raw.split("\n").filter(Boolean);
      expect(lines).toHaveLength(1);
      const parsed = JSON.parse(lines[0]) as MemoryRecord;
      expect(parsed.id).toBe(recordId);
      expect(parsed.content).toBe("User prefers dark mode for coding");
    });

    it("should append multiple records to same daily file", async () => {
      await writeMemory({
        memory: sampleMemory,
        decision: storeDecision(generateMemoryId()),
        baseDir: testDir, sessionKey: "s", sessionId: "sid",
      });
      await writeMemory({
        memory: { ...sampleMemory, content: "User likes TypeScript lang" },
        decision: storeDecision(generateMemoryId()),
        baseDir: testDir, sessionKey: "s", sessionId: "sid",
      });

      const files = await fs.readdir(path.join(testDir, "records"));
      expect(files).toHaveLength(1);
      const lines = (await fs.readFile(path.join(testDir, "records", files[0]), "utf-8")).split("\n").filter(Boolean);
      expect(lines).toHaveLength(2);
    });

    it("should auto-create records directory", async () => {
      const nested = path.join(testDir, "deep", "nested");
      await writeMemory({
        memory: sampleMemory,
        decision: storeDecision(generateMemoryId()),
        baseDir: nested, sessionKey: "s",
      });
      const stat = await fs.stat(path.join(nested, "records"));
      expect(stat.isDirectory()).toBe(true);
    });

    it("should set createdAt and updatedAt to ISO timestamps", async () => {
      const before = new Date().toISOString();
      const record = await writeMemory({
        memory: sampleMemory,
        decision: storeDecision(generateMemoryId()),
        baseDir: testDir, sessionKey: "s",
      });
      const after = new Date().toISOString();

      expect(record!.createdAt >= before).toBe(true);
      expect(record!.createdAt <= after).toBe(true);
      expect(record!.updatedAt).toBe(record!.createdAt);
    });
  });

  // ── LW-02: update/merge operations ──

  describe("LW-02: update/merge operations", () => {
    it("skip action should return null and not write file", async () => {
      const record = await writeMemory({
        memory: sampleMemory,
        decision: { record_id: "m_skip", action: "skip", target_ids: [] },
        baseDir: testDir, sessionKey: "s",
      });
      expect(record).toBeNull();

      // records dir should not exist
      await expect(fs.readdir(path.join(testDir, "records"))).rejects.toThrow();
    });

    it("update action should use merged_content and delete targets from vectorStore", async () => {
      const log = logger();
      const vs = createMockVectorStore();
      const es = createMockEmbeddingService();

      const record = await writeMemory({
        memory: sampleMemory,
        decision: {
          record_id: "m_update",
          action: "update",
          target_ids: ["old_1", "old_2"],
          merged_content: "Updated: user strongly prefers dark mode",
          merged_type: "persona",
          merged_priority: 90,
          merged_timestamps: ["2026-03-17T10:00:00Z", "2026-03-17T11:00:00Z"],
        },
        baseDir: testDir, sessionKey: "s", logger: log,
        vectorStore: vs as any, embeddingService: es as any,
      });

      expect(record).not.toBeNull();
      expect(record!.content).toBe("Updated: user strongly prefers dark mode");
      expect(record!.type).toBe("persona");
      expect(record!.priority).toBe(90);
      expect(record!.timestamps).toEqual(["2026-03-17T10:00:00Z", "2026-03-17T11:00:00Z"]);

      // Should have deleted old records from vectorStore
      expect(vs.deleteBatch).toHaveBeenCalledWith(["old_1", "old_2"]);
      // Should have upserted new record
      expect(vs.upsert).toHaveBeenCalled();
    });

    it("merge action should use merged fields and write JSONL", async () => {
      const record = await writeMemory({
        memory: sampleMemory,
        decision: {
          record_id: "m_merge",
          action: "merge",
          target_ids: ["target_1"],
          merged_content: "Merged: user prefers dark mode and uses VS Code",
          merged_type: "persona",
          merged_priority: 85,
        },
        baseDir: testDir, sessionKey: "s",
      });

      expect(record!.content).toBe("Merged: user prefers dark mode and uses VS Code");
      expect(record!.priority).toBe(85);

      // Verify JSONL
      const files = await fs.readdir(path.join(testDir, "records"));
      const lines = (await fs.readFile(path.join(testDir, "records", files[0]), "utf-8")).split("\n").filter(Boolean);
      expect(lines).toHaveLength(1);
      const parsed = JSON.parse(lines[0]) as MemoryRecord;
      expect(parsed.content).toBe("Merged: user prefers dark mode and uses VS Code");
    });

    it("update without merged fields should fall back to original memory fields", async () => {
      const record = await writeMemory({
        memory: sampleMemory,
        decision: {
          record_id: "m_update_nofields",
          action: "update",
          target_ids: ["old_1"],
          // no merged_content/type/priority
        },
        baseDir: testDir, sessionKey: "s",
      });

      expect(record!.content).toBe(sampleMemory.content);
      expect(record!.type).toBe(sampleMemory.type);
      expect(record!.priority).toBe(sampleMemory.priority);
    });
  });

  // ── LW-03: Daily shard file naming ──

  describe("LW-03: daily shard file naming", () => {
    it("should name file as YYYY-MM-DD.jsonl", async () => {
      await writeMemory({
        memory: sampleMemory,
        decision: storeDecision(generateMemoryId()),
        baseDir: testDir, sessionKey: "s",
      });

      const files = await fs.readdir(path.join(testDir, "records"));
      expect(files[0]).toMatch(/^\d{4}-\d{2}-\d{2}\.jsonl$/);

      const n = new Date();
      const expected = `${n.getFullYear()}-${String(n.getMonth()+1).padStart(2,"0")}-${String(n.getDate()).padStart(2,"0")}.jsonl`;
      expect(files[0]).toBe(expected);
    });
  });

  // ── VectorStore dual-write ──

  describe("vectorStore dual-write", () => {
    it("should call embed + upsert when vectorStore and embeddingService provided", async () => {
      const log = logger();
      const vs = createMockVectorStore();
      const es = createMockEmbeddingService();

      await writeMemory({
        memory: sampleMemory,
        decision: storeDecision("m_dual"),
        baseDir: testDir, sessionKey: "s", logger: log,
        vectorStore: vs as any, embeddingService: es as any,
      });

      expect(es.embed).toHaveBeenCalledWith("User prefers dark mode for coding");
      expect(vs.upsert).toHaveBeenCalled();
      const [record, embedding] = vs.upsert.mock.calls[0];
      expect(record.id).toBe("m_dual");
      expect(embedding).toBeInstanceOf(Float32Array);
    });

    it("should skip dual-write when vectorStore is not provided", async () => {
      const log = logger();
      await writeMemory({
        memory: sampleMemory,
        decision: storeDecision("m_no_vs"),
        baseDir: testDir, sessionKey: "s", logger: log,
      });
      // No error, JSONL still written
      const files = await fs.readdir(path.join(testDir, "records"));
      expect(files).toHaveLength(1);
    });

    it("should not block JSONL write when vectorStore upsert fails", async () => {
      const log = logger();
      const vs = createMockVectorStore();
      vs.upsert.mockImplementation(() => { throw new Error("SQLite error"); });
      const es = createMockEmbeddingService();

      const record = await writeMemory({
        memory: sampleMemory,
        decision: storeDecision("m_fail"),
        baseDir: testDir, sessionKey: "s", logger: log,
        vectorStore: vs as any, embeddingService: es as any,
      });

      // Record should still be returned (JSONL written)
      expect(record).not.toBeNull();
      expect(log.warn).toHaveBeenCalled();
    });

    it("should warn but not fail when vectorStore deleteBatch fails on update", async () => {
      const log = logger();
      const vs = createMockVectorStore();
      vs.deleteBatch.mockImplementation(() => { throw new Error("delete error"); });
      const es = createMockEmbeddingService();

      const record = await writeMemory({
        memory: sampleMemory,
        decision: { record_id: "m_del_fail", action: "update", target_ids: ["t1"], merged_content: "Updated content for testing" },
        baseDir: testDir, sessionKey: "s", logger: log,
        vectorStore: vs as any, embeddingService: es as any,
      });

      expect(record).not.toBeNull();
      expect(log.warn).toHaveBeenCalled();
    });
  });

  // ── sessionId default ──

  describe("sessionId handling", () => {
    it("should default sessionId to empty string when not provided", async () => {
      const record = await writeMemory({
        memory: sampleMemory,
        decision: storeDecision(generateMemoryId()),
        baseDir: testDir, sessionKey: "s",
        // no sessionId
      });
      expect(record!.sessionId).toBe("");
    });
  });

  // ── Edge cases ──

  describe("edge cases", () => {
    it("should auto-generate record_id when decision.record_id is empty string", async () => {
      const record = await writeMemory({
        memory: sampleMemory,
        decision: { record_id: "", action: "store", target_ids: [] },
        baseDir: testDir, sessionKey: "s",
      });
      expect(record).not.toBeNull();
      expect(record!.id).toMatch(/^m_\d+_[0-9a-f]+$/);
    });

    it("should write metadata-only vector record when only vectorStore is provided (no embeddingService)", async () => {
      const log = logger();
      const vs = createMockVectorStore();

      const record = await writeMemory({
        memory: sampleMemory,
        decision: storeDecision("m_no_es"),
        baseDir: testDir, sessionKey: "s", logger: log,
        vectorStore: vs as any,
        // no embeddingService
      });

      expect(record).not.toBeNull();
      expect(vs.upsert).toHaveBeenCalledTimes(1);
      const [, embedding] = vs.upsert.mock.calls[0];
      expect(embedding).toBeUndefined();
      // JSONL should still be written
      const files = await fs.readdir(path.join(testDir, "records"));
      expect(files).toHaveLength(1);
    });

    it("should skip dual-write when only embeddingService is provided (no vectorStore)", async () => {
      const es = createMockEmbeddingService();

      const record = await writeMemory({
        memory: sampleMemory,
        decision: storeDecision("m_no_vs2"),
        baseDir: testDir, sessionKey: "s",
        embeddingService: es as any,
        // no vectorStore
      });

      expect(record).not.toBeNull();
      expect(es.embed).not.toHaveBeenCalled();
    });

    it("should handle update with empty target_ids (no deleteBatch call)", async () => {
      const vs = createMockVectorStore();
      const es = createMockEmbeddingService();

      const record = await writeMemory({
        memory: sampleMemory,
        decision: {
          record_id: "m_update_empty",
          action: "update",
          target_ids: [],
          merged_content: "Updated content with empty targets list",
        },
        baseDir: testDir, sessionKey: "s",
        vectorStore: vs as any, embeddingService: es as any,
      });

      expect(record).not.toBeNull();
      expect(record!.content).toBe("Updated content with empty targets list");
      expect(vs.deleteBatch).not.toHaveBeenCalled();
    });

    it("should warn but not fail when embed fails during dual-write", async () => {
      const log = logger();
      const vs = createMockVectorStore();
      const es = createMockEmbeddingService();
      es.embed.mockRejectedValue(new Error("embedding model unavailable"));

      const record = await writeMemory({
        memory: sampleMemory,
        decision: storeDecision("m_embed_fail"),
        baseDir: testDir, sessionKey: "s", logger: log,
        vectorStore: vs as any, embeddingService: es as any,
      });

      expect(record).not.toBeNull();
      expect(log.warn).toHaveBeenCalledWith(expect.stringContaining("Embedding FAILED"));
      // After zero-vector fallback, upsert SHOULD still be called
      expect(vs.upsert).toHaveBeenCalled();
    });
  });

  // ── Embedding failure metadata-only fallback (LW-04) ──

  describe("LW-04: embedding failure metadata-only fallback", () => {
    it("should pass undefined embedding and still upsert l1_records when embed throws", async () => {
      const log = logger();
      const vs = createMockVectorStore();
      const es = createMockEmbeddingService();
      es.embed.mockRejectedValue(new Error("API timeout"));

      const record = await writeMemory({
        memory: sampleMemory,
        decision: storeDecision("m_zero_vec"),
        baseDir: testDir, sessionKey: "s", logger: log,
        vectorStore: vs as any, embeddingService: es as any,
      });

      // JSONL should still be written
      expect(record).not.toBeNull();
      expect(record!.id).toBe("m_zero_vec");

      // upsert MUST be called with record + undefined embedding (metadata-only)
      expect(vs.upsert).toHaveBeenCalledTimes(1);
      const [upsertedRecord, upsertedEmbedding] = vs.upsert.mock.calls[0];
      expect(upsertedRecord.id).toBe("m_zero_vec");
      expect(upsertedEmbedding).toBeUndefined();

      // Warn log should mention metadata-only fallback and error detail
      expect(log.warn).toHaveBeenCalledWith(expect.stringContaining("Embedding FAILED"));
      expect(log.warn).toHaveBeenCalledWith(expect.stringContaining("metadata only"));
      expect(log.warn).toHaveBeenCalledWith(expect.stringContaining("API timeout"));
    });

    it("should not call getDimensions when embed fails", async () => {
      const log = logger();
      const vs = createMockVectorStore();
      const es = createMockEmbeddingService();
      es.embed.mockRejectedValue(new Error("network error"));

      await writeMemory({
        memory: sampleMemory,
        decision: storeDecision("m_dims"),
        baseDir: testDir, sessionKey: "s", logger: log,
        vectorStore: vs as any, embeddingService: es as any,
      });

      const [, embedding] = vs.upsert.mock.calls[0];
      expect(embedding).toBeUndefined();
      expect(es.getDimensions).not.toHaveBeenCalled();
    });

    it("should log normal embedding info when embed succeeds", async () => {
      const log = logger();
      const vs = createMockVectorStore();
      const es = createMockEmbeddingService();
      // embed succeeds with normal vector
      es.embed.mockResolvedValue(new Float32Array([0.5, 0.3, 0.1]));

      await writeMemory({
        memory: sampleMemory,
        decision: storeDecision("m_normal"),
        baseDir: testDir, sessionKey: "s", logger: log,
        vectorStore: vs as any, embeddingService: es as any,
      });

      // upsert called with normal embedding
      const [, embedding] = vs.upsert.mock.calls[0];
      expect(embedding).toEqual(new Float32Array([0.5, 0.3, 0.1]));

      // Should log "Embedding OK", NOT "Embedding FAILED"
      const warnCalls = log.warn.mock.calls.map((c: unknown[]) => String(c[0]));
      expect(warnCalls.some((msg) => msg.includes("Embedding FAILED"))).toBe(false);
      const debugCalls = log.debug.mock.calls.map((c: unknown[]) => String(c[0]));
      expect(debugCalls.some((msg) => msg.includes("Embedding OK"))).toBe(true);
    });

    it("should still write JSONL even when both embed and upsert fail", async () => {
      const log = logger();
      const vs = createMockVectorStore();
      const es = createMockEmbeddingService();
      es.embed.mockRejectedValue(new Error("embed failure"));
      vs.upsert.mockImplementation(() => { throw new Error("upsert also fails"); });

      const record = await writeMemory({
        memory: sampleMemory,
        decision: storeDecision("m_both_fail"),
        baseDir: testDir, sessionKey: "s", logger: log,
        vectorStore: vs as any, embeddingService: es as any,
      });

      // JSONL must still be written (it happens before dual-write)
      expect(record).not.toBeNull();
      const files = await fs.readdir(path.join(testDir, "records"));
      expect(files).toHaveLength(1);
      const raw = await fs.readFile(path.join(testDir, "records", files[0]), "utf-8");
      const parsed = JSON.parse(raw.trim());
      expect(parsed.id).toBe("m_both_fail");
    });

    it("should work correctly with update action when embed fails", async () => {
      const log = logger();
      const vs = createMockVectorStore();
      const es = createMockEmbeddingService();
      es.embed.mockRejectedValue(new Error("rate limited"));

      const record = await writeMemory({
        memory: sampleMemory,
        decision: {
          record_id: "m_update_embed_fail",
          action: "update",
          target_ids: ["old_1"],
          merged_content: "Updated content after embed failure",
          merged_type: "persona",
          merged_priority: 95,
        },
        baseDir: testDir, sessionKey: "s", logger: log,
        vectorStore: vs as any, embeddingService: es as any,
      });

      expect(record).not.toBeNull();
      expect(record!.content).toBe("Updated content after embed failure");

      // deleteBatch should still be called for the old records
      expect(vs.deleteBatch).toHaveBeenCalledWith(["old_1"]);

      // upsert should be called with undefined embedding
      expect(vs.upsert).toHaveBeenCalledTimes(1);
      const [, embedding] = vs.upsert.mock.calls[0];
      expect(embedding).toBeUndefined();
    });
  });
});
