import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { LocalMemoryCleaner } from "./memory-cleaner.js";
import type { VectorStore } from "../store/vector-store.js";

interface TestLogger {
  debug: ReturnType<typeof vi.fn>;
  info: ReturnType<typeof vi.fn>;
  warn: ReturnType<typeof vi.fn>;
  error: ReturnType<typeof vi.fn>;
}

type Logger = ConstructorParameters<typeof LocalMemoryCleaner>[0]["logger"];

let tmpDir: string;
let logger: TestLogger;

async function exists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

beforeEach(async () => {
  tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "memory-cleaner-test-"));
  logger = {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  };
});

afterEach(async () => {
  await fs.rm(tmpDir, { recursive: true, force: true });
});

describe("LocalMemoryCleaner.runOnce", () => {
  it("should delete only expired daily shard files in conversations/ and records/", async () => {
    const conversationsDir = path.join(tmpDir, "conversations");
    const recordsDir = path.join(tmpDir, "records");
    await fs.mkdir(conversationsDir, { recursive: true });
    await fs.mkdir(recordsDir, { recursive: true });

    const l0Expired = path.join(conversationsDir, "2026-03-16.jsonl");
    const l0Kept = path.join(conversationsDir, "2026-03-17.jsonl");
    const l0NonShard = path.join(conversationsDir, "sessionA__2026-03-16.jsonl");

    const l1Expired = path.join(recordsDir, "2026-03-16.jsonl");
    const l1Kept = path.join(recordsDir, "2026-03-18.jsonl");
    const l1ExpiredJson = path.join(recordsDir, "2026-03-15.json");
    const l1NonShard = path.join(recordsDir, "notes.jsonl");

    await fs.writeFile(l0Expired, "old\n", "utf-8");
    await fs.writeFile(l0Kept, "keep\n", "utf-8");
    await fs.writeFile(l0NonShard, "legacy\n", "utf-8");

    await fs.writeFile(l1Expired, "old\n", "utf-8");
    await fs.writeFile(l1Kept, "keep\n", "utf-8");
    await fs.writeFile(l1ExpiredJson, "old-json\n", "utf-8");
    await fs.writeFile(l1NonShard, "skip\n", "utf-8");

    const cleaner = new LocalMemoryCleaner({
      baseDir: tmpDir,
      retentionDays: 2,
      cleanTime: "03:00",
      logger: logger as Logger,
    });

    const nowMs = new Date(2026, 2, 18, 12, 0, 0, 0).getTime();
    await cleaner.runOnce(nowMs);

    expect(await exists(l0Expired)).toBe(false);
    expect(await exists(l1Expired)).toBe(false);
    expect(await exists(l1ExpiredJson)).toBe(false);

    expect(await exists(l0Kept)).toBe(true);
    expect(await exists(l1Kept)).toBe(true);

    expect(await exists(l0NonShard)).toBe(true);
    expect(await exists(l1NonShard)).toBe(true);
  });

  it("should call VectorStore cleanup with computed cutoff ISO", async () => {
    const conversationsDir = path.join(tmpDir, "conversations");
    const recordsDir = path.join(tmpDir, "records");
    await fs.mkdir(conversationsDir, { recursive: true });
    await fs.mkdir(recordsDir, { recursive: true });

    const deleteL0ExpiredByRecordedAt = vi.fn(() => 1);
    const deleteL1ExpiredByUpdatedTime = vi.fn(() => 2);

    const vectorStore = {
      deleteL0ExpiredByRecordedAt,
      deleteL1ExpiredByUpdatedTime,
    } as unknown as VectorStore;

    const cleaner = new LocalMemoryCleaner({
      baseDir: tmpDir,
      retentionDays: 2,
      cleanTime: "03:00",
      logger: logger as Logger,
      vectorStore,
    });

    const nowMs = new Date(2026, 2, 18, 12, 0, 0, 0).getTime();
    await cleaner.runOnce(nowMs);

    const expectedCutoffIso = new Date(2026, 2, 17, 0, 0, 0, 0).toISOString();

    expect(deleteL0ExpiredByRecordedAt).toHaveBeenCalledTimes(1);
    expect(deleteL1ExpiredByUpdatedTime).toHaveBeenCalledTimes(1);
    expect(deleteL0ExpiredByRecordedAt).toHaveBeenCalledWith(expectedCutoffIso);
    expect(deleteL1ExpiredByUpdatedTime).toHaveBeenCalledWith(expectedCutoffIso);
  });

  it("should continue L1 cleanup when L0 cleanup throws", async () => {
    const conversationsDir = path.join(tmpDir, "conversations");
    const recordsDir = path.join(tmpDir, "records");
    await fs.mkdir(conversationsDir, { recursive: true });
    await fs.mkdir(recordsDir, { recursive: true });

    const expectedError = new Error("mock l0 cleanup failed");
    const deleteL0ExpiredByRecordedAt = vi.fn(() => {
      throw expectedError;
    });
    const deleteL1ExpiredByUpdatedTime = vi.fn(() => 2);

    const vectorStore = {
      deleteL0ExpiredByRecordedAt,
      deleteL1ExpiredByUpdatedTime,
    } as unknown as VectorStore;

    const cleaner = new LocalMemoryCleaner({
      baseDir: tmpDir,
      retentionDays: 2,
      cleanTime: "03:00",
      logger: logger as Logger,
      vectorStore,
    });

    const nowMs = new Date(2026, 2, 18, 12, 0, 0, 0).getTime();
    await cleaner.runOnce(nowMs);

    expect(deleteL0ExpiredByRecordedAt).toHaveBeenCalledTimes(1);
    expect(deleteL1ExpiredByUpdatedTime).toHaveBeenCalledTimes(1);
    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("SQLite cleanup L0 failed: mock l0 cleanup failed"),
    );
    expect(logger.info).toHaveBeenCalledWith(
      expect.stringContaining("failedL0DbCleanup=1"),
    );
    expect(logger.info).toHaveBeenCalledWith(
      expect.stringContaining("removedL1Records=2"),
    );
  });

  it("should skip cleanup when retentionDays is invalid", async () => {
    const conversationsDir = path.join(tmpDir, "conversations");
    await fs.mkdir(conversationsDir, { recursive: true });

    const oldFile = path.join(conversationsDir, "2026-03-01.jsonl");
    await fs.writeFile(oldFile, "old\n", "utf-8");

    const deleteL0ExpiredByRecordedAt = vi.fn(() => 1);
    const deleteL1ExpiredByUpdatedTime = vi.fn(() => 1);

    const vectorStore = {
      deleteL0ExpiredByRecordedAt,
      deleteL1ExpiredByUpdatedTime,
    } as unknown as VectorStore;

    const cleaner = new LocalMemoryCleaner({
      baseDir: tmpDir,
      retentionDays: 0,
      cleanTime: "03:00",
      logger: logger as Logger,
      vectorStore,
    });

    await cleaner.runOnce(new Date(2026, 2, 18, 12, 0, 0, 0).getTime());

    expect(await exists(oldFile)).toBe(true);
    expect(deleteL0ExpiredByRecordedAt).not.toHaveBeenCalled();
    expect(deleteL1ExpiredByUpdatedTime).not.toHaveBeenCalled();
  });
});
