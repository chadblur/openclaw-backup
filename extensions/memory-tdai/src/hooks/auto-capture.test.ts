import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import type { ConversationMessage } from "../conversation/l0-recorder.js";
import type { VectorStore, L0VectorRecord } from "../store/vector-store.js";
import type { EmbeddingService } from "../store/embedding.js";
import type { MemoryTdaiConfig } from "../config.js";

// ============================
// Module-level mocks
// ============================

// Mock recordConversation — we test the orchestration, not the recorder itself
const mockRecordConversation = vi.fn<(opts: Record<string, unknown>) => Promise<ConversationMessage[]>>().mockResolvedValue([]);
vi.mock("../conversation/l0-recorder.js", () => ({
  recordConversation: (...args: unknown[]) => mockRecordConversation(...(args as [Record<string, unknown>])),
}));

// Mock CheckpointManager — track calls without touching the real filesystem
const mockCaptureAtomically = vi.fn<(sessionKey: string, pluginStartTimestamp: number | undefined, fn: (afterTimestamp: number) => Promise<{ maxTimestamp: number; messageCount: number } | null>) => Promise<void>>();

vi.mock("../utils/checkpoint.js", () => {
  const MockCheckpointManager = class {
    captureAtomically = mockCaptureAtomically;
  };
  return { CheckpointManager: MockCheckpointManager };
});

// Import AFTER mocks are set up
const { performAutoCapture } = await import("./auto-capture.js");

// ============================
// Helpers
// ============================

function makeMsg(role: "user" | "assistant", content: string, timestamp: number): ConversationMessage {
  return { id: `msg_${timestamp}`, role, content, timestamp };
}

const minimalCfg = {} as MemoryTdaiConfig;

function makeLogger() {
  return {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  };
}

function makeVectorStore(overrides?: Partial<VectorStore>): VectorStore {
  return {
    upsertL0: vi.fn().mockReturnValue(true),
    ...overrides,
  } as unknown as VectorStore;
}

function makeEmbeddingService(overrides?: Partial<EmbeddingService>): EmbeddingService {
  return {
    embed: vi.fn().mockResolvedValue(new Float32Array([0.1, 0.2, 0.3])),
    getDimensions: vi.fn().mockReturnValue(3),
    ...overrides,
  } as unknown as EmbeddingService;
}

function makeScheduler() {
  return {
    notifyConversation: vi.fn().mockResolvedValue(undefined),
  };
}

// ============================
// Setup / Teardown
// ============================

let tmpDir: string;

beforeEach(async () => {
  tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "auto-capture-test-"));
  vi.clearAllMocks();

  // Default captureAtomically mock: invokes the callback with cursor=0
  // and resolves without error.
  mockCaptureAtomically.mockImplementation(async (_sessionKey, _pluginStartTs, fn) => {
    await fn(0);
  });
});

afterEach(async () => {
  await fs.rm(tmpDir, { recursive: true, force: true });
});

// ============================
// Tests
// ============================

describe("performAutoCapture", () => {
  // ──────────────────────────────────
  // Step 1: L0 local recording
  // ──────────────────────────────────
  describe("Step 1: L0 recording", () => {
    it("records new messages via captureAtomically and returns them", async () => {
      const msgs = [
        makeMsg("user", "hello", 1000),
        makeMsg("assistant", "hi there", 1001),
      ];
      mockRecordConversation.mockResolvedValue(msgs);

      const result = await performAutoCapture({
        messages: [{ role: "user", content: "hello" }],
        sessionKey: "s1",
        cfg: minimalCfg,
        pluginDataDir: tmpDir,
      });

      expect(mockCaptureAtomically).toHaveBeenCalledOnce();
      expect(mockRecordConversation).toHaveBeenCalledOnce();
      expect(result.l0RecordedCount).toBe(2);
      expect(result.filteredMessages).toEqual(msgs);
    });

    it("returns zero recorded count when recordConversation returns empty array", async () => {
      mockRecordConversation.mockResolvedValue([]);

      const result = await performAutoCapture({
        messages: [],
        sessionKey: "s1",
        cfg: minimalCfg,
        pluginDataDir: tmpDir,
      });

      expect(result.l0RecordedCount).toBe(0);
    });

    it("catches captureAtomically error and continues pipeline", async () => {
      mockCaptureAtomically.mockRejectedValue(new Error("disk full"));
      const logger = makeLogger();
      const scheduler = makeScheduler();

      const result = await performAutoCapture({
        messages: [{ role: "user", content: "test" }],
        sessionKey: "s1",
        cfg: minimalCfg,
        pluginDataDir: tmpDir,
        logger,
        scheduler: scheduler as any,
      });

      // L0 failed but pipeline continues → scheduler is still notified
      expect(logger.error).toHaveBeenCalledWith(expect.stringContaining("L0 recording failed"));
      expect(scheduler.notifyConversation).toHaveBeenCalled();
      expect(result.l0RecordedCount).toBe(0);
      expect(result.schedulerNotified).toBe(true);
    });

    it("passes pluginStartTimestamp to captureAtomically", async () => {
      const pluginStartTs = 5000;
      mockRecordConversation.mockResolvedValue([]);

      await performAutoCapture({
        messages: [{ role: "user", content: "hi" }],
        sessionKey: "s1",
        cfg: minimalCfg,
        pluginDataDir: tmpDir,
        pluginStartTimestamp: pluginStartTs,
      });

      // captureAtomically should be called with pluginStartTimestamp
      expect(mockCaptureAtomically).toHaveBeenCalledWith(
        "s1",
        pluginStartTs,
        expect.any(Function),
      );
    });

    it("uses afterTimestamp from captureAtomically callback in recordConversation", async () => {
      // Simulate captureAtomically passing cursor=9000 to the callback
      mockCaptureAtomically.mockImplementation(async (_sessionKey, _pluginStartTs, fn) => {
        await fn(9000);
      });
      mockRecordConversation.mockResolvedValue([]);

      await performAutoCapture({
        messages: [{ role: "user", content: "hi" }],
        sessionKey: "s1",
        cfg: minimalCfg,
        pluginDataDir: tmpDir,
        pluginStartTimestamp: 5000,
      });

      const callArgs = mockRecordConversation.mock.calls[0][0] as Record<string, unknown>;
      expect(callArgs.afterTimestamp).toBe(9000);
    });

    it("passes originalUserText and originalUserMessageCount to recordConversation", async () => {
      mockRecordConversation.mockResolvedValue([]);

      await performAutoCapture({
        messages: [{ role: "user", content: "test" }],
        sessionKey: "s1",
        sessionId: "sid-1",
        cfg: minimalCfg,
        pluginDataDir: tmpDir,
        originalUserText: "clean prompt",
        originalUserMessageCount: 3,
      });

      const callArgs = mockRecordConversation.mock.calls[0][0] as Record<string, unknown>;
      expect(callArgs.originalUserText).toBe("clean prompt");
      expect(callArgs.originalUserMessageCount).toBe(3);
      expect(callArgs.sessionId).toBe("sid-1");
    });

    it("callback returns maxTimestamp + messageCount when messages exist", async () => {
      const msgs = [
        makeMsg("user", "aaa", 1000),
        makeMsg("assistant", "bbb", 3000),
        makeMsg("user", "ccc", 2000),
      ];
      mockRecordConversation.mockResolvedValue(msgs);

      // Capture what the callback returns
      let callbackResult: { maxTimestamp: number; messageCount: number } | null = null;
      mockCaptureAtomically.mockImplementation(async (_sessionKey, _pluginStartTs, fn) => {
        callbackResult = await fn(0);
      });

      await performAutoCapture({
        messages: [],
        sessionKey: "s1",
        cfg: minimalCfg,
        pluginDataDir: tmpDir,
      });

      expect(callbackResult).toEqual({ maxTimestamp: 3000, messageCount: 3 });
    });

    it("callback returns null when no messages recorded", async () => {
      mockRecordConversation.mockResolvedValue([]);

      let callbackResult: unknown = "sentinel";
      mockCaptureAtomically.mockImplementation(async (_sessionKey, _pluginStartTs, fn) => {
        callbackResult = await fn(0);
      });

      await performAutoCapture({
        messages: [],
        sessionKey: "s1",
        cfg: minimalCfg,
        pluginDataDir: tmpDir,
      });

      expect(callbackResult).toBeNull();
    });
  });

  // ──────────────────────────────────
  // Step 1.5: L0 vector indexing
  // ──────────────────────────────────
  describe("Step 1.5: L0 vector indexing", () => {
    it("embeds and upserts each message when vectorStore + embeddingService available", async () => {
      const msgs = [
        makeMsg("user", "what is AI", 1000),
        makeMsg("assistant", "AI is artificial intelligence", 1001),
      ];
      mockRecordConversation.mockResolvedValue(msgs);

      const vectorStore = makeVectorStore();
      const embeddingService = makeEmbeddingService();

      const result = await performAutoCapture({
        messages: [],
        sessionKey: "s1",
        sessionId: "sid-1",
        cfg: minimalCfg,
        pluginDataDir: tmpDir,
        vectorStore,
        embeddingService,
      });

      expect(embeddingService.embed).toHaveBeenCalledTimes(2);
      expect(embeddingService.embed).toHaveBeenCalledWith("what is AI");
      expect(embeddingService.embed).toHaveBeenCalledWith("AI is artificial intelligence");
      expect(vectorStore.upsertL0).toHaveBeenCalledTimes(2);
      expect(result.l0VectorsWritten).toBe(2);
    });

    it("skips vector indexing when vectorStore is undefined", async () => {
      mockRecordConversation.mockResolvedValue([makeMsg("user", "test", 1000)]);
      const embeddingService = makeEmbeddingService();
      const logger = makeLogger();

      const result = await performAutoCapture({
        messages: [],
        sessionKey: "s1",
        cfg: minimalCfg,
        pluginDataDir: tmpDir,
        embeddingService,
        logger,
      });

      expect(embeddingService.embed).not.toHaveBeenCalled();
      expect(result.l0VectorsWritten).toBe(0);
      expect(logger.warn).toHaveBeenCalledWith(expect.stringContaining("SKIPPED"));
    });

    it("writes metadata-only vectors when embeddingService is undefined", async () => {
      mockRecordConversation.mockResolvedValue([makeMsg("user", "test", 1000)]);
      const vectorStore = makeVectorStore();
      const logger = makeLogger();

      const result = await performAutoCapture({
        messages: [],
        sessionKey: "s1",
        cfg: minimalCfg,
        pluginDataDir: tmpDir,
        vectorStore,
        logger,
      });

      expect(vectorStore.upsertL0).toHaveBeenCalledOnce();
      const [, embedding] = (vectorStore.upsertL0 as ReturnType<typeof vi.fn>).mock.calls[0];
      expect(embedding).toBeUndefined();
      expect(result.l0VectorsWritten).toBe(1);
      expect(logger.warn).not.toHaveBeenCalledWith(expect.stringContaining("SKIPPED"));
    });

    it("writes metadata-only when embed() throws", async () => {
      const msgs = [makeMsg("user", "test", 1000)];
      mockRecordConversation.mockResolvedValue(msgs);

      const vectorStore = makeVectorStore();
      const embeddingService = makeEmbeddingService({
        embed: vi.fn().mockRejectedValue(new Error("model not loaded")),
      });
      const logger = makeLogger();

      const result = await performAutoCapture({
        messages: [],
        sessionKey: "s1",
        cfg: minimalCfg,
        pluginDataDir: tmpDir,
        vectorStore,
        embeddingService,
        logger,
      });

      // Should still call upsertL0, but with undefined embedding (metadata-only)
      expect(vectorStore.upsertL0).toHaveBeenCalledOnce();
      const [record, embedding] = (vectorStore.upsertL0 as ReturnType<typeof vi.fn>).mock.calls[0];
      expect(embedding).toBeUndefined();
      expect(record.role).toBe("user");
      expect(result.l0VectorsWritten).toBe(1);
      expect(logger.warn).toHaveBeenCalledWith(expect.stringContaining("Embedding FAILED"));
      expect(logger.warn).toHaveBeenCalledWith(expect.stringContaining("metadata only"));
    });

    it("does not count when upsertL0 returns false", async () => {
      mockRecordConversation.mockResolvedValue([makeMsg("user", "test", 1000)]);

      const vectorStore = makeVectorStore({
        upsertL0: vi.fn().mockReturnValue(false),
      } as any);
      const embeddingService = makeEmbeddingService();
      const logger = makeLogger();

      const result = await performAutoCapture({
        messages: [],
        sessionKey: "s1",
        cfg: minimalCfg,
        pluginDataDir: tmpDir,
        vectorStore,
        embeddingService,
        logger,
      });

      expect(result.l0VectorsWritten).toBe(0);
      expect(logger.warn).toHaveBeenCalledWith(expect.stringContaining("upsertL0 returned false"));
    });

    it("continues to next message when upsertL0 throws for one message", async () => {
      const msgs = [
        makeMsg("user", "first", 1000),
        makeMsg("assistant", "second", 1001),
      ];
      mockRecordConversation.mockResolvedValue(msgs);

      let callCount = 0;
      const vectorStore = makeVectorStore({
        upsertL0: vi.fn().mockImplementation(() => {
          callCount++;
          if (callCount === 1) throw new Error("DB locked");
          return true;
        }),
      } as any);
      const embeddingService = makeEmbeddingService();
      const logger = makeLogger();

      const result = await performAutoCapture({
        messages: [],
        sessionKey: "s1",
        cfg: minimalCfg,
        pluginDataDir: tmpDir,
        vectorStore,
        embeddingService,
        logger,
      });

      // First message failed, second succeeded
      expect(vectorStore.upsertL0).toHaveBeenCalledTimes(2);
      expect(result.l0VectorsWritten).toBe(1);
    });

    it("verifies L0VectorRecord fields passed to upsertL0", async () => {
      const msg = makeMsg("assistant", "hello world", 2000);
      mockRecordConversation.mockResolvedValue([msg]);

      const vectorStore = makeVectorStore();
      const embeddingService = makeEmbeddingService();

      await performAutoCapture({
        messages: [],
        sessionKey: "session-abc",
        sessionId: "sid-xyz",
        cfg: minimalCfg,
        pluginDataDir: tmpDir,
        vectorStore,
        embeddingService,
      });

      const [record] = (vectorStore.upsertL0 as ReturnType<typeof vi.fn>).mock.calls[0] as [L0VectorRecord, Float32Array];
      expect(record.id).toMatch(/^l0_session-abc_/);
      expect(record.sessionKey).toBe("session-abc");
      expect(record.sessionId).toBe("sid-xyz");
      expect(record.role).toBe("assistant");
      expect(record.messageText).toBe("hello world");
      expect(record.timestamp).toBe(2000);
      expect(typeof record.recordedAt).toBe("string");
    });
  });

  // ──────────────────────────────────
  // Step 2: Checkpoint update
  // ──────────────────────────────────
  describe("Step 2: checkpoint update (merged into captureAtomically)", () => {
    it("captureAtomically is called with correct sessionKey", async () => {
      const msgs = [
        makeMsg("user", "aaa", 1000),
        makeMsg("assistant", "bbb", 3000),
        makeMsg("user", "ccc", 2000),
      ];
      mockRecordConversation.mockResolvedValue(msgs);

      await performAutoCapture({
        messages: [],
        sessionKey: "s1",
        cfg: minimalCfg,
        pluginDataDir: tmpDir,
      });

      expect(mockCaptureAtomically).toHaveBeenCalledWith("s1", undefined, expect.any(Function));
    });

    it("does not call captureAtomically callback advance when no messages recorded", async () => {
      mockRecordConversation.mockResolvedValue([]);

      let callbackResult: unknown = "sentinel";
      mockCaptureAtomically.mockImplementation(async (_sessionKey, _pluginStartTs, fn) => {
        callbackResult = await fn(0);
      });

      await performAutoCapture({
        messages: [],
        sessionKey: "s1",
        cfg: minimalCfg,
        pluginDataDir: tmpDir,
      });

      // callback returns null — captureAtomically will not advance cursor
      expect(callbackResult).toBeNull();
    });
  });

  // ──────────────────────────────────
  // Step 3: Scheduler notification
  // ──────────────────────────────────
  describe("Step 3: scheduler notification", () => {
    it("notifies scheduler and returns schedulerNotified=true", async () => {
      mockRecordConversation.mockResolvedValue([makeMsg("user", "test", 1000)]);
      const scheduler = makeScheduler();

      const result = await performAutoCapture({
        messages: [],
        sessionKey: "s1",
        cfg: minimalCfg,
        pluginDataDir: tmpDir,
        scheduler: scheduler as any,
      });

      expect(scheduler.notifyConversation).toHaveBeenCalledWith("s1", []);
      expect(result.schedulerNotified).toBe(true);
    });

    it("returns schedulerNotified=false when no scheduler provided", async () => {
      mockRecordConversation.mockResolvedValue([makeMsg("user", "test", 1000)]);

      const result = await performAutoCapture({
        messages: [],
        sessionKey: "s1",
        cfg: minimalCfg,
        pluginDataDir: tmpDir,
      });

      expect(result.schedulerNotified).toBe(false);
    });
  });

  // ──────────────────────────────────
  // End-to-end return value
  // ──────────────────────────────────
  describe("full pipeline return value", () => {
    it("returns correct composite result for a complete pipeline run", async () => {
      const msgs = [
        makeMsg("user", "Q", 1000),
        makeMsg("assistant", "A", 1001),
      ];
      mockRecordConversation.mockResolvedValue(msgs);

      const vectorStore = makeVectorStore();
      const embeddingService = makeEmbeddingService();
      const scheduler = makeScheduler();

      const result = await performAutoCapture({
        messages: [{ role: "user", content: "Q" }],
        sessionKey: "s1",
        cfg: minimalCfg,
        pluginDataDir: tmpDir,
        vectorStore,
        embeddingService,
        scheduler: scheduler as any,
      });

      expect(result).toEqual({
        schedulerNotified: true,
        l0RecordedCount: 2,
        l0VectorsWritten: 2,
        filteredMessages: msgs,
      });
    });

    it("returns zeros for empty pipeline (no messages, no vectors, no scheduler)", async () => {
      mockRecordConversation.mockResolvedValue([]);

      const result = await performAutoCapture({
        messages: [],
        sessionKey: "s1",
        cfg: minimalCfg,
        pluginDataDir: tmpDir,
      });

      expect(result).toEqual({
        schedulerNotified: false,
        l0RecordedCount: 0,
        l0VectorsWritten: 0,
        filteredMessages: [],
      });
    });
  });
});
