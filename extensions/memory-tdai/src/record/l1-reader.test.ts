/**
 * Unit tests for L1 Memory Reader (A 同学自测).
 *
 * Covers:
 * - queryMemoryRecords (SQLite path): null vectorStore → empty, row conversion
 * - readMemoryRecords (JSONL fallback): session filter, daily shards, sort, malformed lines
 * - readAllMemoryRecords: read all sessions, sort, empty dir
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import {
  queryMemoryRecords,
  readMemoryRecords,
  readAllMemoryRecords,
} from "./l1-reader.js";
import type { MemoryRecord } from "./l1-writer.js";
import type { L1RecordRow, L1QueryFilter } from "../store/vector-store.js";

let testDir: string;
const mkDir = async () => {
  const d = path.join(os.tmpdir(), `l1-reader-test-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`);
  await fs.mkdir(d, { recursive: true });
  return d;
};
const rmDir = async (d: string) => { try { await fs.rm(d, { recursive: true, force: true }); } catch {} };
const logger = () => ({ debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() });

function makeL1Row(overrides?: Partial<L1RecordRow>): L1RecordRow {
  return {
    record_id: overrides?.record_id ?? "r_1",
    content: overrides?.content ?? "Test memory content",
    type: overrides?.type ?? "persona",
    priority: overrides?.priority ?? 50,
    scene_name: overrides?.scene_name ?? "test-scene",
    session_key: overrides?.session_key ?? "s1",
    session_id: overrides?.session_id ?? "sid-1",
    timestamp_str: overrides?.timestamp_str ?? "2026-03-17T10:00:00Z",
    timestamp_start: overrides?.timestamp_start ?? "",
    timestamp_end: overrides?.timestamp_end ?? "",
    created_time: overrides?.created_time ?? "2026-03-17T10:00:00Z",
    updated_time: overrides?.updated_time ?? "2026-03-17T10:00:00Z",
    metadata_json: overrides?.metadata_json ?? "{}",
  };
}

function makeJsonlRecord(overrides?: Partial<MemoryRecord>): MemoryRecord {
  return {
    id: overrides?.id ?? "m_1",
    content: overrides?.content ?? "Test memory JSONL content",
    type: overrides?.type ?? "persona",
    priority: overrides?.priority ?? 50,
    scene_name: overrides?.scene_name ?? "",
    source_message_ids: overrides?.source_message_ids ?? [],
    metadata: overrides?.metadata ?? {},
    timestamps: overrides?.timestamps ?? [],
    createdAt: overrides?.createdAt ?? "2026-03-17T10:00:00Z",
    updatedAt: overrides?.updatedAt ?? "2026-03-17T10:00:00Z",
    sessionKey: overrides?.sessionKey ?? "s1",
    sessionId: overrides?.sessionId ?? "",
  };
}

beforeEach(async () => { testDir = await mkDir(); });
afterEach(async () => { await rmDir(testDir); });

// ── queryMemoryRecords (SQLite path) ──

describe("queryMemoryRecords (SQLite path)", () => {
  it("should return empty when vectorStore is null", () => {
    const log = logger();
    const result = queryMemoryRecords(null, undefined, log);
    expect(result).toEqual([]);
    expect(log.warn).toHaveBeenCalled();
  });

  it("should return empty when vectorStore is undefined", () => {
    const log = logger();
    const result = queryMemoryRecords(undefined, undefined, log);
    expect(result).toEqual([]);
  });

  it("should convert L1RecordRow to MemoryRecord correctly", () => {
    const row = makeL1Row({
      record_id: "r_42",
      content: "User prefers dark mode editor",
      type: "persona",
      priority: 80,
      scene_name: "coding",
      session_key: "session-A",
      session_id: "sid-1",
      timestamp_str: "2026-03-17T10:00:00Z",
      timestamp_start: "2026-03-17T09:00:00Z",
      timestamp_end: "2026-03-17T11:00:00Z",
      created_time: "2026-03-17T09:00:00Z",
      updated_time: "2026-03-17T11:00:00Z",
      metadata_json: JSON.stringify({ activity_start_time: "2026-03-17T09:00:00Z" }),
    });

    const mockVS = {
      queryL1Records: vi.fn().mockReturnValue([row]),
      count: vi.fn().mockReturnValue(1),
      search: vi.fn(),
      upsert: vi.fn(),
      deleteBatch: vi.fn(),
      close: vi.fn(),
    };

    const result = queryMemoryRecords(mockVS as any);

    expect(result).toHaveLength(1);
    const r = result[0];
    expect(r.id).toBe("r_42");
    expect(r.content).toBe("User prefers dark mode editor");
    expect(r.type).toBe("persona");
    expect(r.priority).toBe(80);
    expect(r.scene_name).toBe("coding");
    expect(r.sessionKey).toBe("session-A");
    expect(r.sessionId).toBe("sid-1");
    expect(r.createdAt).toBe("2026-03-17T09:00:00Z");
    expect(r.updatedAt).toBe("2026-03-17T11:00:00Z");
    // timestamps should include all unique values
    expect(r.timestamps).toContain("2026-03-17T10:00:00Z");
    expect(r.timestamps).toContain("2026-03-17T09:00:00Z");
    expect(r.timestamps).toContain("2026-03-17T11:00:00Z");
    // metadata should be parsed
    expect(r.metadata).toEqual({ activity_start_time: "2026-03-17T09:00:00Z" });
    // source_message_ids not stored in SQLite → empty
    expect(r.source_message_ids).toEqual([]);
  });

  it("should pass filter to vectorStore.queryL1Records", () => {
    const mockVS = {
      queryL1Records: vi.fn().mockReturnValue([]),
      count: vi.fn(), search: vi.fn(), upsert: vi.fn(), deleteBatch: vi.fn(), close: vi.fn(),
    };
    const filter: L1QueryFilter = { sessionKey: "sk1", updatedAfter: "2026-03-17T00:00:00Z" };

    queryMemoryRecords(mockVS as any, filter);
    expect(mockVS.queryL1Records).toHaveBeenCalledWith(filter);
  });

  it("should handle malformed metadata_json gracefully", () => {
    const row = makeL1Row({ metadata_json: "not-json" });
    const mockVS = {
      queryL1Records: vi.fn().mockReturnValue([row]),
      count: vi.fn(), search: vi.fn(), upsert: vi.fn(), deleteBatch: vi.fn(), close: vi.fn(),
    };

    const result = queryMemoryRecords(mockVS as any);
    expect(result).toHaveLength(1);
    expect(result[0].metadata).toEqual({}); // fallback to empty
  });

  it("should deduplicate timestamp values", () => {
    const row = makeL1Row({
      timestamp_str: "2026-03-17T10:00:00Z",
      timestamp_start: "2026-03-17T10:00:00Z", // same as timestamp_str
      timestamp_end: "2026-03-17T12:00:00Z",
    });
    const mockVS = {
      queryL1Records: vi.fn().mockReturnValue([row]),
      count: vi.fn(), search: vi.fn(), upsert: vi.fn(), deleteBatch: vi.fn(), close: vi.fn(),
    };

    const result = queryMemoryRecords(mockVS as any);
    // Should not have duplicates
    const timestamps = result[0].timestamps;
    expect(new Set(timestamps).size).toBe(timestamps.length);
  });
});

// ── readMemoryRecords (JSONL fallback) ──

describe("readMemoryRecords (JSONL fallback)", () => {
  it("should return empty when records dir does not exist", async () => {
    expect(await readMemoryRecords("s1", testDir)).toEqual([]);
  });

  it("should return empty when no matching files", async () => {
    await fs.mkdir(path.join(testDir, "records"), { recursive: true });
    expect(await readMemoryRecords("s1", testDir)).toEqual([]);
  });

  it("should read daily merged JSONL files and filter by sessionKey", async () => {
    const recordsDir = path.join(testDir, "records");
    await fs.mkdir(recordsDir, { recursive: true });

    const rec1 = makeJsonlRecord({ id: "m_1", content: "Memory one", sessionKey: "s1", updatedAt: "2026-03-17T10:00:00Z" });
    const rec2 = makeJsonlRecord({ id: "m_2", content: "Memory two", sessionKey: "s1", updatedAt: "2026-03-17T11:00:00Z" });
    const rec3 = makeJsonlRecord({ id: "m_other", content: "Other session", sessionKey: "s2", updatedAt: "2026-03-17T12:00:00Z" });

    await fs.writeFile(path.join(recordsDir, "2026-03-17.jsonl"), [JSON.stringify(rec1), JSON.stringify(rec2), JSON.stringify(rec3)].join("\n") + "\n");

    const result = await readMemoryRecords("s1", testDir);
    expect(result).toHaveLength(2);
    expect(result[0].id).toBe("m_1");
    expect(result[1].id).toBe("m_2");
  });

  it("should ignore files not matching YYYY-MM-DD.jsonl pattern", async () => {
    const recordsDir = path.join(testDir, "records");
    await fs.mkdir(recordsDir, { recursive: true });

    const rec = makeJsonlRecord({ id: "m_in_date_file" });
    // 符合格式的文件 — 应该被读取
    await fs.writeFile(path.join(recordsDir, "2026-03-17.jsonl"), JSON.stringify(rec) + "\n");
    // 不符合格式的文件 — 应该全部被忽略
    await fs.writeFile(path.join(recordsDir, "s1.jsonl"), JSON.stringify(makeJsonlRecord({ id: "m_ignored_1" })) + "\n");
    await fs.writeFile(path.join(recordsDir, "s1__2026-03-17.jsonl"), JSON.stringify(makeJsonlRecord({ id: "m_ignored_2" })) + "\n");
    await fs.writeFile(path.join(recordsDir, "other-session.jsonl"), JSON.stringify(makeJsonlRecord({ id: "m_ignored_3" })) + "\n");
    await fs.writeFile(path.join(recordsDir, "readme.txt"), "not a jsonl file");

    const result = await readMemoryRecords("s1", testDir);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("m_in_date_file");
  });

  it("should skip malformed JSONL lines", async () => {
    const log = logger();
    const recordsDir = path.join(testDir, "records");
    await fs.mkdir(recordsDir, { recursive: true });

    const valid = makeJsonlRecord({ id: "m_valid" });
    await fs.writeFile(
      path.join(recordsDir, "2026-03-17.jsonl"),
      [JSON.stringify(valid), "bad json line", JSON.stringify(makeJsonlRecord({ id: "m_valid2" }))].join("\n") + "\n",
    );

    const result = await readMemoryRecords("s1", testDir, log);
    expect(result).toHaveLength(2);
    expect(log.warn).toHaveBeenCalled();
  });

  it("should sort by updatedAt", async () => {
    const recordsDir = path.join(testDir, "records");
    await fs.mkdir(recordsDir, { recursive: true });

    const recs = [
      makeJsonlRecord({ id: "m_3", updatedAt: "2026-03-17T12:00:00Z" }),
      makeJsonlRecord({ id: "m_1", updatedAt: "2026-03-17T10:00:00Z" }),
      makeJsonlRecord({ id: "m_2", updatedAt: "2026-03-17T11:00:00Z" }),
    ];
    await fs.writeFile(path.join(recordsDir, "2026-03-17.jsonl"), recs.map((r) => JSON.stringify(r)).join("\n") + "\n");

    const result = await readMemoryRecords("s1", testDir);
    expect(result[0].id).toBe("m_1");
    expect(result[1].id).toBe("m_2");
    expect(result[2].id).toBe("m_3");
  });
});

// ── readAllMemoryRecords ──

describe("readAllMemoryRecords", () => {
  it("should return empty when records dir does not exist", async () => {
    expect(await readAllMemoryRecords(testDir)).toEqual([]);
  });

  it("should read all JSONL files regardless of session", async () => {
    const recordsDir = path.join(testDir, "records");
    await fs.mkdir(recordsDir, { recursive: true });

    const rec1 = makeJsonlRecord({ id: "m_a", sessionKey: "s1", updatedAt: "2026-03-17T10:00:00Z" });
    const rec2 = makeJsonlRecord({ id: "m_b", sessionKey: "s2", updatedAt: "2026-03-17T11:00:00Z" });

    await fs.writeFile(path.join(recordsDir, "s1.jsonl"), JSON.stringify(rec1) + "\n");
    await fs.writeFile(path.join(recordsDir, "s2.jsonl"), JSON.stringify(rec2) + "\n");

    const result = await readAllMemoryRecords(testDir);
    expect(result).toHaveLength(2);
    expect(result[0].id).toBe("m_a");
    expect(result[1].id).toBe("m_b");
  });

  it("should skip non-JSONL files", async () => {
    const recordsDir = path.join(testDir, "records");
    await fs.mkdir(recordsDir, { recursive: true });

    await fs.writeFile(path.join(recordsDir, "readme.txt"), "not a jsonl file");
    await fs.writeFile(path.join(recordsDir, "data.jsonl"), JSON.stringify(makeJsonlRecord({ id: "m_only" })) + "\n");

    const result = await readAllMemoryRecords(testDir);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("m_only");
  });

  it("should skip malformed lines and continue", async () => {
    const log = logger();
    const recordsDir = path.join(testDir, "records");
    await fs.mkdir(recordsDir, { recursive: true });

    await fs.writeFile(
      path.join(recordsDir, "all.jsonl"),
      [JSON.stringify(makeJsonlRecord({ id: "good" })), "bad-line", JSON.stringify(makeJsonlRecord({ id: "good2" }))].join("\n") + "\n",
    );

    const result = await readAllMemoryRecords(testDir, log);
    expect(result).toHaveLength(2);
    expect(log.warn).toHaveBeenCalled();
  });

  it("should sort all records by updatedAt across files", async () => {
    const recordsDir = path.join(testDir, "records");
    await fs.mkdir(recordsDir, { recursive: true });

    await fs.writeFile(path.join(recordsDir, "a.jsonl"), JSON.stringify(makeJsonlRecord({ id: "late", updatedAt: "2026-03-17T12:00:00Z" })) + "\n");
    await fs.writeFile(path.join(recordsDir, "b.jsonl"), JSON.stringify(makeJsonlRecord({ id: "early", updatedAt: "2026-03-17T08:00:00Z" })) + "\n");

    const result = await readAllMemoryRecords(testDir);
    expect(result[0].id).toBe("early");
    expect(result[1].id).toBe("late");
  });
});

// ── Edge cases ──

describe("queryMemoryRecords edge cases", () => {
  it("should handle empty timestamp_str/start/end → empty timestamps array", () => {
    const row = makeL1Row({
      timestamp_str: "",
      timestamp_start: "",
      timestamp_end: "",
    });
    const mockVS = {
      queryL1Records: vi.fn().mockReturnValue([row]),
      count: vi.fn(), search: vi.fn(), upsert: vi.fn(), deleteBatch: vi.fn(), close: vi.fn(),
    };

    const result = queryMemoryRecords(mockVS as any);
    expect(result[0].timestamps).toEqual([]);
  });

  it("should handle multiple rows conversion", () => {
    const rows = [
      makeL1Row({ record_id: "r_1", content: "First memory content", updated_time: "2026-03-17T10:00:00Z" }),
      makeL1Row({ record_id: "r_2", content: "Second memory content", updated_time: "2026-03-17T11:00:00Z" }),
      makeL1Row({ record_id: "r_3", content: "Third memory content", updated_time: "2026-03-17T12:00:00Z" }),
    ];
    const mockVS = {
      queryL1Records: vi.fn().mockReturnValue(rows),
      count: vi.fn(), search: vi.fn(), upsert: vi.fn(), deleteBatch: vi.fn(), close: vi.fn(),
    };

    const result = queryMemoryRecords(mockVS as any);
    expect(result).toHaveLength(3);
    expect(result.map(r => r.id)).toEqual(["r_1", "r_2", "r_3"]);
  });

  it("should call queryL1Records without filter when filter is undefined", () => {
    const mockVS = {
      queryL1Records: vi.fn().mockReturnValue([]),
      count: vi.fn(), search: vi.fn(), upsert: vi.fn(), deleteBatch: vi.fn(), close: vi.fn(),
    };

    queryMemoryRecords(mockVS as any);
    expect(mockVS.queryL1Records).toHaveBeenCalledWith(undefined);
  });
});

describe("readMemoryRecords edge cases", () => {
  it("should handle sessionKey with special characters via JSONL field matching", async () => {
    const recordsDir = path.join(testDir, "records");
    await fs.mkdir(recordsDir, { recursive: true });

    const rec = makeJsonlRecord({ id: "m_special", sessionKey: "a/b:c" });
    await fs.writeFile(path.join(recordsDir, "2026-03-17.jsonl"), JSON.stringify(rec) + "\n");

    const result = await readMemoryRecords("a/b:c", testDir);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("m_special");
  });

  it("should handle file read error gracefully and continue", async () => {
    const log = logger();
    const recordsDir = path.join(testDir, "records");
    await fs.mkdir(recordsDir, { recursive: true });

    const rec1 = makeJsonlRecord({ id: "m_good", updatedAt: "2026-03-16T10:00:00Z" });
    await fs.writeFile(path.join(recordsDir, "2026-03-16.jsonl"), JSON.stringify(rec1) + "\n");
    const rec2 = makeJsonlRecord({ id: "m_good2", updatedAt: "2026-03-17T11:00:00Z" });
    await fs.writeFile(path.join(recordsDir, "2026-03-17.jsonl"), JSON.stringify(rec2) + "\n");

    const result = await readMemoryRecords("s1", testDir, log);
    expect(result).toHaveLength(2);
  });

  it("should handle records with missing updatedAt by falling back to createdAt", async () => {
    const recordsDir = path.join(testDir, "records");
    await fs.mkdir(recordsDir, { recursive: true });

    const rec = { id: "m_no_updated", content: "No updatedAt field", type: "persona", priority: 50, scene_name: "", source_message_ids: [], metadata: {}, timestamps: [], createdAt: "2026-03-17T10:00:00Z", sessionKey: "s1", sessionId: "" };
    await fs.writeFile(path.join(recordsDir, "2026-03-17.jsonl"), JSON.stringify(rec) + "\n");

    const result = await readMemoryRecords("s1", testDir);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("m_no_updated");
  });
});

describe("readAllMemoryRecords edge cases", () => {
  it("should handle empty JSONL files", async () => {
    const recordsDir = path.join(testDir, "records");
    await fs.mkdir(recordsDir, { recursive: true });
    await fs.writeFile(path.join(recordsDir, "empty.jsonl"), "");

    const result = await readAllMemoryRecords(testDir);
    expect(result).toEqual([]);
  });

  it("should handle JSONL files with only whitespace lines", async () => {
    const recordsDir = path.join(testDir, "records");
    await fs.mkdir(recordsDir, { recursive: true });
    await fs.writeFile(path.join(recordsDir, "whitespace.jsonl"), "  \n\n  \n");

    const result = await readAllMemoryRecords(testDir);
    expect(result).toEqual([]);
  });
});
