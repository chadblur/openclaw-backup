/**
 * Unit tests for CheckpointManager.
 *
 * Covers:
 * - Read/Write checkpoint (basic I/O)
 * - Atomic write (tmp + rename)
 * - File lock serialization (concurrent mutations)
 * - advanceCapturedTimestamp / advanceSessionCapturedTimestamp
 * - Per-session runner state (L0/L1)
 * - Per-session pipeline state (PipelineManager owned)
 * - mergePipelineStates (split-state design)
 * - markL1ExtractionComplete
 * - Persona methods (markPersonaGenerated, clearPersonaRequest)
 * - incrementL0ConversationCount / incrementScenesProcessed
 * - Default checkpoint when file doesn't exist
 * - Migration from old session_states format
 * - File not found → default checkpoint
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { CheckpointManager } from "./checkpoint.js";
import type { Checkpoint, PipelineSessionState, RunnerSessionState } from "./checkpoint.js";

// ============================
// Helpers
// ============================

let testDir: string;

async function createTestDir(): Promise<string> {
  const dir = path.join(os.tmpdir(), `checkpoint-test-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`);
  await fs.mkdir(dir, { recursive: true });
  return dir;
}

async function cleanupTestDir(dir: string): Promise<void> {
  try {
    await fs.rm(dir, { recursive: true, force: true });
  } catch {
    // ignore
  }
}

function createTestLogger() {
  return {
    info: vi.fn(),
    warn: vi.fn(),
  };
}

// ============================
// Tests
// ============================

describe("CheckpointManager", () => {
  beforeEach(async () => {
    testDir = await createTestDir();
  });

  afterEach(async () => {
    await cleanupTestDir(testDir);
  });

  // ─────────────────────────────────────
  // Basic read/write
  // ─────────────────────────────────────

  describe("basic read/write", () => {
    it("should return default checkpoint when file does not exist", async () => {
      const cm = new CheckpointManager(testDir);
      const cp = await cm.read();

      expect(cp.last_captured_timestamp).toBe(0);
      expect(cp.total_processed).toBe(0);
      expect(cp.runner_states).toEqual({});
      expect(cp.pipeline_states).toEqual({});
      expect(cp.l0_conversations_count).toBe(0);
      expect(cp.total_memories_extracted).toBe(0);
      expect(cp.memories_since_last_persona).toBe(0);
    });

    it("should write and read back a checkpoint correctly", async () => {
      const cm = new CheckpointManager(testDir, createTestLogger());
      const cp = await cm.read();

      cp.last_captured_timestamp = 12345;
      cp.total_processed = 100;
      cp.runner_states = {
        "session-a": {
          last_captured_timestamp: 12345,
          last_l1_cursor: 10000,
          last_scene_name: "work",
        },
      };

      await cm.write(cp);

      const cp2 = await cm.read();
      expect(cp2.last_captured_timestamp).toBe(12345);
      expect(cp2.total_processed).toBe(100);
      expect(cp2.runner_states["session-a"].last_l1_cursor).toBe(10000);
      expect(cp2.runner_states["session-a"].last_scene_name).toBe("work");
    });

    it("should create directory automatically when writing", async () => {
      const nestedDir = path.join(testDir, "deep", "nested");
      const cm = new CheckpointManager(nestedDir);
      const cp = await cm.read();
      cp.total_processed = 42;
      await cm.write(cp);

      const cm2 = new CheckpointManager(nestedDir);
      const cp2 = await cm2.read();
      expect(cp2.total_processed).toBe(42);
    });
  });

  // ─────────────────────────────────────
  // Atomic write
  // ─────────────────────────────────────

  describe("atomic write", () => {
    it("checkpoint file should exist after write (tmp+rename)", async () => {
      const cm = new CheckpointManager(testDir);
      const cp = await cm.read();
      cp.total_processed = 1;
      await cm.write(cp);

      const filePath = path.join(testDir, ".metadata", "recall_checkpoint.json");
      const stat = await fs.stat(filePath);
      expect(stat.isFile()).toBe(true);
    });

    it("checkpoint content should be valid JSON", async () => {
      const cm = new CheckpointManager(testDir);
      const cp = await cm.read();
      cp.total_processed = 99;
      await cm.write(cp);

      const filePath = path.join(testDir, ".metadata", "recall_checkpoint.json");
      const raw = await fs.readFile(filePath, "utf-8");
      const parsed = JSON.parse(raw);
      expect(parsed.total_processed).toBe(99);
    });
  });

  // ─────────────────────────────────────
  // advanceCapturedTimestamp
  // ─────────────────────────────────────

  describe("advanceCapturedTimestamp", () => {
    it("should advance global cursor and counters", async () => {
      const cm = new CheckpointManager(testDir, createTestLogger());

      await cm.advanceCapturedTimestamp(5000, 10);

      const cp = await cm.read();
      expect(cp.last_captured_timestamp).toBe(5000);
      expect(cp.total_processed).toBe(10);
      expect(cp.memories_since_last_persona).toBe(10);
    });

    it("should accumulate on multiple calls", async () => {
      const cm = new CheckpointManager(testDir, createTestLogger());

      await cm.advanceCapturedTimestamp(1000, 5);
      await cm.advanceCapturedTimestamp(2000, 3);

      const cp = await cm.read();
      expect(cp.last_captured_timestamp).toBe(2000);
      expect(cp.total_processed).toBe(8);
      expect(cp.memories_since_last_persona).toBe(8);
    });
  });

  // ─────────────────────────────────────
  // advanceSessionCapturedTimestamp
  // ─────────────────────────────────────

  describe("advanceSessionCapturedTimestamp", () => {
    it("should advance per-session cursor and global stats", async () => {
      const cm = new CheckpointManager(testDir, createTestLogger());

      await cm.advanceSessionCapturedTimestamp("session-a", 3000, 7);

      const cp = await cm.read();
      expect(cp.runner_states["session-a"].last_captured_timestamp).toBe(3000);
      expect(cp.last_captured_timestamp).toBe(3000);
      expect(cp.total_processed).toBe(7);
    });

    it("should track multiple sessions independently", async () => {
      const cm = new CheckpointManager(testDir, createTestLogger());

      await cm.advanceSessionCapturedTimestamp("s1", 1000, 5);
      await cm.advanceSessionCapturedTimestamp("s2", 2000, 3);

      const cp = await cm.read();
      expect(cp.runner_states["s1"].last_captured_timestamp).toBe(1000);
      expect(cp.runner_states["s2"].last_captured_timestamp).toBe(2000);
      expect(cp.last_captured_timestamp).toBe(2000); // max
      expect(cp.total_processed).toBe(8);
    });
  });

  // ─────────────────────────────────────
  // incrementL0ConversationCount
  // ─────────────────────────────────────

  describe("incrementL0ConversationCount", () => {
    it("should increment count by 1", async () => {
      const cm = new CheckpointManager(testDir);

      await cm.incrementL0ConversationCount();
      await cm.incrementL0ConversationCount();
      await cm.incrementL0ConversationCount();

      const cp = await cm.read();
      expect(cp.l0_conversations_count).toBe(3);
    });
  });

  // ─────────────────────────────────────
  // Persona methods
  // ─────────────────────────────────────

  describe("persona methods", () => {
    it("markPersonaGenerated should reset persona counters", async () => {
      const cm = new CheckpointManager(testDir);

      // Set up some state first
      await cm.advanceCapturedTimestamp(1000, 20);

      await cm.markPersonaGenerated(20);

      const cp = await cm.read();
      expect(cp.last_persona_at).toBe(20);
      expect(cp.last_persona_time).not.toBe("");
      expect(cp.memories_since_last_persona).toBe(0);
      expect(cp.request_persona_update).toBe(false);
    });

    it("clearPersonaRequest should clear request flag", async () => {
      const cm = new CheckpointManager(testDir);
      const cp = await cm.read();
      cp.request_persona_update = true;
      cp.persona_update_reason = "test reason";
      await cm.write(cp);

      await cm.clearPersonaRequest();

      const cp2 = await cm.read();
      expect(cp2.request_persona_update).toBe(false);
      expect(cp2.persona_update_reason).toBe("");
    });
  });

  // ─────────────────────────────────────
  // incrementScenesProcessed
  // ─────────────────────────────────────

  describe("incrementScenesProcessed", () => {
    it("should increment scenes_processed counter", async () => {
      const cm = new CheckpointManager(testDir, createTestLogger());

      await cm.incrementScenesProcessed();
      await cm.incrementScenesProcessed();

      const cp = await cm.read();
      expect(cp.scenes_processed).toBe(2);
    });
  });

  // ─────────────────────────────────────
  // Per-session helpers
  // ─────────────────────────────────────

  describe("per-session helpers", () => {
    it("getRunnerState should create default state for new session", async () => {
      const cm = new CheckpointManager(testDir);
      const cp = await cm.read();

      const state = cm.getRunnerState(cp, "new-session");
      expect(state.last_captured_timestamp).toBe(0);
      expect(state.last_l1_cursor).toBe(0);
      expect(state.last_scene_name).toBe("");
    });

    it("getRunnerState should return existing state for known session", async () => {
      const cm = new CheckpointManager(testDir, createTestLogger());

      await cm.advanceSessionCapturedTimestamp("s1", 5000, 1);

      const cp = await cm.read();
      const state = cm.getRunnerState(cp, "s1");
      expect(state.last_captured_timestamp).toBe(5000);
    });

    it("getPipelineState should create default state for new session", async () => {
      const cm = new CheckpointManager(testDir);
      const cp = await cm.read();

      const state = cm.getPipelineState(cp, "new-session");
      expect(state.conversation_count).toBe(0);
      expect(state.l2_pending_l1_count).toBe(0);
      expect(state.warmup_threshold).toBe(0);
    });

    it("getAllPipelineStates should return all pipeline states", async () => {
      const cm = new CheckpointManager(testDir);

      // Merge some pipeline states
      await cm.mergePipelineStates({
        "s1": {
          conversation_count: 3,
          last_extraction_time: "",
          last_extraction_updated_time: "",
          last_active_time: Date.now(),
          l2_pending_l1_count: 2,
          warmup_threshold: 0,
          l2_last_extraction_time: "",
        },
      });

      const cp = await cm.read();
      const all = cm.getAllPipelineStates(cp);
      expect(Object.keys(all)).toContain("s1");
      expect(all["s1"].conversation_count).toBe(3);
    });
  });

  // ─────────────────────────────────────
  // mergePipelineStates (split-state design)
  // ─────────────────────────────────────

  describe("mergePipelineStates", () => {
    it("should merge pipeline states without overwriting runner states", async () => {
      const cm = new CheckpointManager(testDir, createTestLogger());

      // First, set some runner state
      await cm.advanceSessionCapturedTimestamp("s1", 9000, 5);

      // Now merge pipeline state
      await cm.mergePipelineStates({
        "s1": {
          conversation_count: 10,
          last_extraction_time: "2026-03-17T10:00:00Z",
          last_extraction_updated_time: "",
          last_active_time: Date.now(),
          l2_pending_l1_count: 3,
          warmup_threshold: 2,
          l2_last_extraction_time: "",
        },
      });

      // Read back — both runner and pipeline states should coexist
      const cp = await cm.read();
      expect(cp.runner_states["s1"].last_captured_timestamp).toBe(9000); // preserved
      expect(cp.pipeline_states["s1"].conversation_count).toBe(10); // merged
      expect(cp.pipeline_states["s1"].l2_pending_l1_count).toBe(3);
    });

    it("should merge incrementally (not overwrite existing pipeline state fields)", async () => {
      const cm = new CheckpointManager(testDir);

      await cm.mergePipelineStates({
        "s1": {
          conversation_count: 5,
          last_extraction_time: "2026-03-17T00:00:00Z",
          last_extraction_updated_time: "cursor-1",
          last_active_time: 1000,
          l2_pending_l1_count: 2,
          warmup_threshold: 4,
          l2_last_extraction_time: "",
        },
      });

      // Merge again with updated fields
      await cm.mergePipelineStates({
        "s1": {
          conversation_count: 0,
          last_extraction_time: "2026-03-17T01:00:00Z",
          last_extraction_updated_time: "cursor-2",
          last_active_time: 2000,
          l2_pending_l1_count: 0,
          warmup_threshold: 0,
          l2_last_extraction_time: "2026-03-17T01:00:00Z",
        },
      });

      const cp = await cm.read();
      expect(cp.pipeline_states["s1"].conversation_count).toBe(0);
      expect(cp.pipeline_states["s1"].last_extraction_updated_time).toBe("cursor-2");
      expect(cp.pipeline_states["s1"].warmup_threshold).toBe(0);
    });

    it("should handle multiple sessions", async () => {
      const cm = new CheckpointManager(testDir);

      await cm.mergePipelineStates({
        "s1": {
          conversation_count: 1,
          last_extraction_time: "",
          last_extraction_updated_time: "",
          last_active_time: Date.now(),
          l2_pending_l1_count: 0,
          warmup_threshold: 1,
          l2_last_extraction_time: "",
        },
        "s2": {
          conversation_count: 2,
          last_extraction_time: "",
          last_extraction_updated_time: "",
          last_active_time: Date.now(),
          l2_pending_l1_count: 1,
          warmup_threshold: 0,
          l2_last_extraction_time: "",
        },
      });

      const cp = await cm.read();
      expect(cp.pipeline_states["s1"].conversation_count).toBe(1);
      expect(cp.pipeline_states["s2"].conversation_count).toBe(2);
    });
  });

  // ─────────────────────────────────────
  // markL1ExtractionComplete
  // ─────────────────────────────────────

  describe("markL1ExtractionComplete", () => {
    it("should update runner state and global counters", async () => {
      const cm = new CheckpointManager(testDir, createTestLogger());

      await cm.markL1ExtractionComplete("s1", 5, 8000, "travel");

      const cp = await cm.read();
      expect(cp.runner_states["s1"].last_l1_cursor).toBe(8000);
      expect(cp.runner_states["s1"].last_scene_name).toBe("travel");
      expect(cp.total_memories_extracted).toBe(5);
      expect(cp.memories_since_last_persona).toBe(5);
    });

    it("should not overwrite cursor if not provided", async () => {
      const cm = new CheckpointManager(testDir, createTestLogger());

      // Set initial cursor
      await cm.markL1ExtractionComplete("s1", 3, 5000, "work");

      // Call again without cursor
      await cm.markL1ExtractionComplete("s1", 2, undefined, "work2");

      const cp = await cm.read();
      expect(cp.runner_states["s1"].last_l1_cursor).toBe(5000); // unchanged
      expect(cp.runner_states["s1"].last_scene_name).toBe("work2"); // updated
      expect(cp.total_memories_extracted).toBe(5);
    });

    it("should accumulate total_memories_extracted across calls", async () => {
      const cm = new CheckpointManager(testDir, createTestLogger());

      await cm.markL1ExtractionComplete("s1", 3, 1000, "");
      await cm.markL1ExtractionComplete("s2", 7, 2000, "");

      const cp = await cm.read();
      expect(cp.total_memories_extracted).toBe(10);
    });
  });

  // ─────────────────────────────────────
  // Concurrent mutations (file lock)
  // ─────────────────────────────────────

  describe("concurrent mutations (file lock serialization)", () => {
    it("should serialize concurrent increments without losing updates", async () => {
      const cm = new CheckpointManager(testDir);

      // Run 10 concurrent incrementL0ConversationCount calls
      const promises = Array.from({ length: 10 }, () =>
        cm.incrementL0ConversationCount(),
      );
      await Promise.all(promises);

      const cp = await cm.read();
      expect(cp.l0_conversations_count).toBe(10);
    });

    it("should serialize concurrent session timestamp advances", async () => {
      const cm = new CheckpointManager(testDir, createTestLogger());

      const promises = Array.from({ length: 5 }, (_, i) =>
        cm.advanceSessionCapturedTimestamp(`session-${i}`, (i + 1) * 1000, 1),
      );
      await Promise.all(promises);

      const cp = await cm.read();
      expect(cp.total_processed).toBe(5);
      expect(Object.keys(cp.runner_states)).toHaveLength(5);
    });
  });

  // ─────────────────────────────────────
  // Migration from old session_states format
  // ─────────────────────────────────────

  describe("migration from old session_states format", () => {
    it("should migrate old session_states to runner_states + pipeline_states", async () => {
      // Write an old-format checkpoint directly
      const metadataDir = path.join(testDir, ".metadata");
      await fs.mkdir(metadataDir, { recursive: true });
      const filePath = path.join(metadataDir, "recall_checkpoint.json");
      const oldCheckpoint = {
        last_captured_timestamp: 5000,
        total_processed: 10,
        last_persona_at: 0,
        last_persona_time: "",
        request_persona_update: false,
        persona_update_reason: "",
        memories_since_last_persona: 10,
        scenes_processed: 0,
        l0_conversations_count: 0,
        total_memories_extracted: 0,
        // Old format: session_states (not split)
        session_states: {
          "old-session": {
            last_captured_timestamp: 5000,
            last_l1_cursor: 3000,
            last_scene_name: "legacy",
            conversation_count: 2,
            last_extraction_time: "2026-01-01",
            last_extraction_updated_time: "cursor-old",
            last_active_time: 1000,
            l2_pending_l1_count: 1,
            l2_last_extraction_time: "",
          },
        },
      };
      await fs.writeFile(filePath, JSON.stringify(oldCheckpoint), "utf-8");

      // Read with CheckpointManager — should auto-migrate
      const cm = new CheckpointManager(testDir);
      const cp = await cm.read();

      // Should have migrated runner_states
      expect(cp.runner_states["old-session"]).toBeDefined();
      expect(cp.runner_states["old-session"].last_captured_timestamp).toBe(5000);
      expect(cp.runner_states["old-session"].last_l1_cursor).toBe(3000);
      expect(cp.runner_states["old-session"].last_scene_name).toBe("legacy");

      // Should have migrated pipeline_states
      expect(cp.pipeline_states["old-session"]).toBeDefined();
      expect(cp.pipeline_states["old-session"].conversation_count).toBe(2);
      expect(cp.pipeline_states["old-session"].last_extraction_updated_time).toBe("cursor-old");
      expect(cp.pipeline_states["old-session"].l2_pending_l1_count).toBe(1);
    });
  });

  // ─────────────────────────────────────
  // Default filling
  // ─────────────────────────────────────

  describe("default filling for missing fields", () => {
    it("should fill missing fields with defaults when reading old checkpoints", async () => {
      const metadataDir = path.join(testDir, ".metadata");
      await fs.mkdir(metadataDir, { recursive: true });
      const filePath = path.join(metadataDir, "recall_checkpoint.json");

      // Write a minimal checkpoint (missing many fields)
      await fs.writeFile(filePath, JSON.stringify({
        last_captured_timestamp: 100,
        total_processed: 5,
        runner_states: {
          "s1": {
            last_captured_timestamp: 100,
            // missing last_l1_cursor, last_scene_name
          },
        },
        pipeline_states: {
          "s1": {
            conversation_count: 2,
            // missing many fields
          },
        },
      }), "utf-8");

      const cm = new CheckpointManager(testDir);
      const cp = await cm.read();

      // Global defaults
      expect(cp.l0_conversations_count).toBe(0);
      expect(cp.total_memories_extracted).toBe(0);

      // Runner state defaults
      expect(cp.runner_states["s1"].last_l1_cursor).toBe(0);
      expect(cp.runner_states["s1"].last_scene_name).toBe("");

      // Pipeline state defaults
      expect(cp.pipeline_states["s1"].l2_pending_l1_count).toBe(0);
      expect(cp.pipeline_states["s1"].warmup_threshold).toBe(0);
      expect(cp.pipeline_states["s1"].l2_last_extraction_time).toBe("");
    });
  });

  // ─────────────────────────────────────
  // Multiple CheckpointManager instances (shared lock)
  // ─────────────────────────────────────

  describe("shared file lock across instances", () => {
    it("two managers pointing to same file should not corrupt data", async () => {
      const cm1 = new CheckpointManager(testDir);
      const cm2 = new CheckpointManager(testDir);

      // Interleaved mutations from both instances
      await Promise.all([
        cm1.incrementL0ConversationCount(),
        cm2.incrementL0ConversationCount(),
        cm1.incrementL0ConversationCount(),
        cm2.incrementL0ConversationCount(),
      ]);

      const cp = await cm1.read();
      expect(cp.l0_conversations_count).toBe(4);
    });
  });

  // ─────────────────────────────────────
  // captureAtomically (race-condition fix)
  // ─────────────────────────────────────

  describe("captureAtomically", () => {
    it("should atomically read cursor, execute callback, and advance cursor", async () => {
      const cm = new CheckpointManager(testDir, createTestLogger());

      await cm.captureAtomically("s1", undefined, async (afterTimestamp) => {
        expect(afterTimestamp).toBe(0); // fresh session
        return { maxTimestamp: 5000, messageCount: 3 };
      });

      const cp = await cm.read();
      expect(cp.runner_states["s1"].last_captured_timestamp).toBe(5000);
      expect(cp.last_captured_timestamp).toBe(5000);
      expect(cp.total_processed).toBe(3);
      expect(cp.memories_since_last_persona).toBe(3);
      expect(cp.l0_conversations_count).toBe(1);
    });

    it("should not advance cursor when callback returns null", async () => {
      const cm = new CheckpointManager(testDir, createTestLogger());

      await cm.captureAtomically("s1", undefined, async (_afterTimestamp) => {
        return null; // nothing captured
      });

      const cp = await cm.read();
      expect(cp.runner_states["s1"].last_captured_timestamp).toBe(0);
      expect(cp.total_processed).toBe(0);
      expect(cp.l0_conversations_count).toBe(0);
    });

    it("should use pluginStartTimestamp as floor when cursor is 0", async () => {
      const cm = new CheckpointManager(testDir);
      let receivedTimestamp = -1;

      await cm.captureAtomically("s1", 8000, async (afterTimestamp) => {
        receivedTimestamp = afterTimestamp;
        return null;
      });

      expect(receivedTimestamp).toBe(8000);
    });

    it("should use existing cursor when it is > 0 (ignore pluginStartTimestamp)", async () => {
      const cm = new CheckpointManager(testDir, createTestLogger());

      // First capture — sets cursor to 3000
      await cm.captureAtomically("s1", undefined, async () => {
        return { maxTimestamp: 3000, messageCount: 1 };
      });

      // Second capture — should use 3000, not pluginStartTimestamp=500
      let receivedTimestamp = -1;
      await cm.captureAtomically("s1", 500, async (afterTimestamp) => {
        receivedTimestamp = afterTimestamp;
        return { maxTimestamp: 6000, messageCount: 2 };
      });

      expect(receivedTimestamp).toBe(3000);
      const cp = await cm.read();
      expect(cp.runner_states["s1"].last_captured_timestamp).toBe(6000);
      expect(cp.total_processed).toBe(3); // 1 + 2
      expect(cp.l0_conversations_count).toBe(2);
    });

    it("should serialize concurrent captureAtomically calls without duplicate captures", async () => {
      const cm = new CheckpointManager(testDir, createTestLogger());

      // Simulate two concurrent agent_end events for the SAME session.
      // Without atomicity, both would read cursor=0 and capture the same messages.
      // With captureAtomically, they are serialized: the second call sees
      // the cursor advanced by the first call.
      const cursorsReceived: number[] = [];

      const p1 = cm.captureAtomically("s1", undefined, async (afterTimestamp) => {
        cursorsReceived.push(afterTimestamp);
        // Simulate slow recording (gives p2 time to try to race)
        await new Promise((r) => setTimeout(r, 50));
        return { maxTimestamp: 5000, messageCount: 3 };
      });

      const p2 = cm.captureAtomically("s1", undefined, async (afterTimestamp) => {
        cursorsReceived.push(afterTimestamp);
        // Second call sees the cursor advanced by p1
        return { maxTimestamp: 8000, messageCount: 2 };
      });

      await Promise.all([p1, p2]);

      // First call should see cursor=0, second should see cursor=5000
      expect(cursorsReceived).toEqual([0, 5000]);

      const cp = await cm.read();
      expect(cp.runner_states["s1"].last_captured_timestamp).toBe(8000);
      expect(cp.total_processed).toBe(5); // 3 + 2
      expect(cp.l0_conversations_count).toBe(2);
    });

    it("should track multiple sessions independently under concurrent calls", async () => {
      const cm = new CheckpointManager(testDir, createTestLogger());

      await Promise.all([
        cm.captureAtomically("s1", undefined, async () => {
          return { maxTimestamp: 1000, messageCount: 1 };
        }),
        cm.captureAtomically("s2", undefined, async () => {
          return { maxTimestamp: 2000, messageCount: 2 };
        }),
      ]);

      const cp = await cm.read();
      expect(cp.runner_states["s1"].last_captured_timestamp).toBe(1000);
      expect(cp.runner_states["s2"].last_captured_timestamp).toBe(2000);
      expect(cp.total_processed).toBe(3);
      expect(cp.l0_conversations_count).toBe(2);
    });
  });
});
