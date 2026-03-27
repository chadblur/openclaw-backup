import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { PersonaTrigger } from "./persona-trigger.js";
import { CheckpointManager, type Checkpoint } from "../utils/checkpoint.js";

let tmpDir: string;

beforeEach(async () => {
  tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "persona-trigger-test-"));
});

afterEach(async () => {
  await fs.rm(tmpDir, { recursive: true, force: true });
});

/** Helper: write a checkpoint with custom overrides */
async function writeCheckpoint(overrides: Partial<Checkpoint>): Promise<void> {
  const metaDir = path.join(tmpDir, ".metadata");
  await fs.mkdir(metaDir, { recursive: true });
  const defaults: Checkpoint = {
    last_captured_timestamp: 0,
    total_processed: 0,
    last_persona_at: 0,
    last_persona_time: "",
    request_persona_update: false,
    persona_update_reason: "",
    memories_since_last_persona: 0,
    scenes_processed: 0,
    runner_states: {},
    pipeline_states: {},
    l0_conversations_count: 0,
    total_memories_extracted: 0,
  };
  await fs.writeFile(
    path.join(metaDir, "recall_checkpoint.json"),
    JSON.stringify({ ...defaults, ...overrides }, null, 2),
    "utf-8",
  );
}

/** Helper: create scene block .md files in scene_blocks/ */
async function createSceneBlocks(count: number): Promise<void> {
  const blocksDir = path.join(tmpDir, "scene_blocks");
  await fs.mkdir(blocksDir, { recursive: true });
  for (let i = 0; i < count; i++) {
    await fs.writeFile(
      path.join(blocksDir, `scene_${String(i + 1).padStart(3, "0")}.md`),
      `-----META-START-----\nsummary: Scene ${i + 1}\nheat: ${(i + 1) * 10}\n-----META-END-----\n\nContent ${i + 1}`,
      "utf-8",
    );
  }
}

/** Helper: write persona.md with given content */
async function writePersona(content: string): Promise<void> {
  await fs.writeFile(path.join(tmpDir, "persona.md"), content, "utf-8");
}

function makeTrigger(interval = 10): PersonaTrigger {
  return new PersonaTrigger({ dataDir: tmpDir, interval });
}

// ── P1: Explicit request ──

describe("P1: explicit persona update request", () => {
  it("should trigger when request_persona_update is true", async () => {
    await writeCheckpoint({
      request_persona_update: true,
      persona_update_reason: "用户要求更新",
    });
    const result = await makeTrigger().shouldGenerate();
    expect(result.should).toBe(true);
    expect(result.reason).toContain("主动请求");
    expect(result.reason).toContain("用户要求更新");
  });

  it("should include default reason when persona_update_reason is empty", async () => {
    await writeCheckpoint({
      request_persona_update: true,
      persona_update_reason: "",
    });
    const result = await makeTrigger().shouldGenerate();
    expect(result.should).toBe(true);
    expect(result.reason).toContain("Agent 请求更新");
  });

  it("should take priority over all other conditions (P1 > P2/P3/P4)", async () => {
    // All conditions met simultaneously — P1 should win
    await writeCheckpoint({
      request_persona_update: true,
      persona_update_reason: "urgent",
      scenes_processed: 1,
      last_persona_at: 0,
      memories_since_last_persona: 999,
    });
    await createSceneBlocks(3);
    const result = await makeTrigger(5).shouldGenerate();
    expect(result.should).toBe(true);
    expect(result.reason).toContain("主动请求");
  });
});

// ── P2: Cold start ──

describe("P2: cold start", () => {
  it("should trigger on first extraction with scene files and no persona", async () => {
    await writeCheckpoint({
      scenes_processed: 2,
      last_persona_at: 0,
    });
    await createSceneBlocks(2);
    const result = await makeTrigger().shouldGenerate();
    expect(result.should).toBe(true);
    expect(result.reason).toContain("冷启动");
  });

  it("should NOT trigger when scenes_processed is 0", async () => {
    await writeCheckpoint({
      scenes_processed: 0,
      last_persona_at: 0,
    });
    await createSceneBlocks(1);
    const result = await makeTrigger().shouldGenerate();
    expect(result.should).toBe(false);
  });

  it("should NOT trigger when scene_blocks directory does not exist", async () => {
    await writeCheckpoint({
      scenes_processed: 5,
      last_persona_at: 0,
    });
    // No scene_blocks dir created
    const result = await makeTrigger().shouldGenerate();
    expect(result.should).toBe(false);
  });

  it("should NOT trigger when scene_blocks has no .md files", async () => {
    await writeCheckpoint({
      scenes_processed: 3,
      last_persona_at: 0,
    });
    const blocksDir = path.join(tmpDir, "scene_blocks");
    await fs.mkdir(blocksDir, { recursive: true });
    await fs.writeFile(path.join(blocksDir, "readme.txt"), "not a scene", "utf-8");
    const result = await makeTrigger().shouldGenerate();
    expect(result.should).toBe(false);
  });
});

// ── P2.5: Recovery ──

describe("P2.5: persona recovery", () => {
  it("should trigger when persona was generated before but persona.md is missing", async () => {
    await writeCheckpoint({
      last_persona_at: 50,
      scenes_processed: 5,
    });
    await createSceneBlocks(2);
    // No persona.md file
    const result = await makeTrigger().shouldGenerate();
    expect(result.should).toBe(true);
    expect(result.reason).toContain("恢复");
  });

  it("should trigger when persona.md body is empty (only navigation)", async () => {
    await writeCheckpoint({
      last_persona_at: 50,
      scenes_processed: 5,
    });
    await createSceneBlocks(2);
    // Persona with only navigation, no body content
    await writePersona("---\n## 🗺️ Scene Navigation (Scene Index)\nSome nav content...");
    const result = await makeTrigger().shouldGenerate();
    expect(result.should).toBe(true);
    expect(result.reason).toContain("恢复");
  });

  it("should trigger when persona.md is empty string", async () => {
    await writeCheckpoint({
      last_persona_at: 50,
      scenes_processed: 5,
    });
    await createSceneBlocks(1);
    await writePersona("");
    const result = await makeTrigger().shouldGenerate();
    expect(result.should).toBe(true);
    expect(result.reason).toContain("恢复");
  });

  it("should NOT trigger recovery when persona has real body content", async () => {
    await writeCheckpoint({
      last_persona_at: 50,
      scenes_processed: 5,
      memories_since_last_persona: 0,
    });
    await createSceneBlocks(2);
    await writePersona("# User Profile\n\nReal persona content here.");
    const result = await makeTrigger().shouldGenerate();
    expect(result.should).toBe(false);
  });

  it("should NOT trigger recovery when last_persona_at is 0 (never generated)", async () => {
    // This is P2 territory, not P2.5
    await writeCheckpoint({
      last_persona_at: 0,
      scenes_processed: 5,
    });
    await createSceneBlocks(1);
    // Even without persona.md, if last_persona_at=0, P2.5 won't fire
    // (but P2 would fire instead since scenes_processed > 0 + no persona)
    const result = await makeTrigger().shouldGenerate();
    expect(result.should).toBe(true);
    expect(result.reason).toContain("冷启动"); // P2, not P2.5
  });
});

// ── P3: First scene block extraction ──

describe("P3: first scene block extraction", () => {
  it("should trigger when scenes_processed=1 and memories_since > 0", async () => {
    await writeCheckpoint({
      scenes_processed: 1,
      last_persona_at: 10,
      memories_since_last_persona: 3,
    });
    await createSceneBlocks(1);
    await writePersona("# Existing persona content");
    const result = await makeTrigger().shouldGenerate();
    expect(result.should).toBe(true);
    expect(result.reason).toContain("首次 Scene Block");
  });

  it("should NOT trigger when scenes_processed=1 but memories_since=0", async () => {
    await writeCheckpoint({
      scenes_processed: 1,
      last_persona_at: 10,
      memories_since_last_persona: 0,
    });
    await createSceneBlocks(1);
    await writePersona("# Existing persona content");
    const result = await makeTrigger().shouldGenerate();
    expect(result.should).toBe(false);
  });

  it("should NOT trigger when scenes_processed > 1 (not first)", async () => {
    await writeCheckpoint({
      scenes_processed: 2,
      last_persona_at: 10,
      memories_since_last_persona: 5,
    });
    await createSceneBlocks(2);
    await writePersona("# Existing persona content");
    // P3 requires scenes_processed === 1 exactly
    // This would fall through to P4 if threshold met
    const result = await makeTrigger(100).shouldGenerate();
    expect(result.should).toBe(false);
  });
});

// ── P4: Threshold reached ──

describe("P4: threshold reached", () => {
  it("should trigger when memories_since >= interval", async () => {
    await writeCheckpoint({
      scenes_processed: 5,
      last_persona_at: 20,
      memories_since_last_persona: 10,
    });
    await createSceneBlocks(5);
    await writePersona("# Some content");
    const result = await makeTrigger(10).shouldGenerate();
    expect(result.should).toBe(true);
    expect(result.reason).toContain("达到阈值");
    expect(result.reason).toContain("10 >= 10");
  });

  it("should trigger when memories_since > interval", async () => {
    await writeCheckpoint({
      scenes_processed: 3,
      last_persona_at: 10,
      memories_since_last_persona: 25,
    });
    await createSceneBlocks(3);
    await writePersona("# Content");
    const result = await makeTrigger(10).shouldGenerate();
    expect(result.should).toBe(true);
    expect(result.reason).toContain("25 >= 10");
  });

  it("should NOT trigger when memories_since < interval (boundary: interval-1)", async () => {
    await writeCheckpoint({
      scenes_processed: 3,
      last_persona_at: 10,
      memories_since_last_persona: 9,
    });
    await createSceneBlocks(3);
    await writePersona("# Content");
    const result = await makeTrigger(10).shouldGenerate();
    expect(result.should).toBe(false);
  });
});

// ── No trigger ──

describe("no trigger conditions met", () => {
  it("should return should=false with empty reason", async () => {
    await writeCheckpoint({
      scenes_processed: 3,
      last_persona_at: 10,
      memories_since_last_persona: 2,
    });
    await createSceneBlocks(3);
    await writePersona("# Some persona content");
    const result = await makeTrigger(50).shouldGenerate();
    expect(result.should).toBe(false);
    expect(result.reason).toBe("");
  });

  it("should handle missing checkpoint file gracefully (all defaults)", async () => {
    // No checkpoint, no scene_blocks, no persona — all defaults → no trigger
    const result = await makeTrigger().shouldGenerate();
    expect(result.should).toBe(false);
  });
});
