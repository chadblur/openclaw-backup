import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { readSceneIndex, writeSceneIndex, syncSceneIndex } from "./scene-index.js";
import type { SceneIndexEntry } from "./scene-index.js";

let tmpDir: string;

beforeEach(async () => {
  tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "scene-index-test-"));
});

afterEach(async () => {
  await fs.rm(tmpDir, { recursive: true, force: true });
});

// ── readSceneIndex ──

describe("readSceneIndex", () => {
  it("should return empty array when index file does not exist", async () => {
    const entries = await readSceneIndex(tmpDir);
    expect(entries).toEqual([]);
  });

  it("should read a valid scene index", async () => {
    const metaDir = path.join(tmpDir, ".metadata");
    await fs.mkdir(metaDir, { recursive: true });

    const data: SceneIndexEntry[] = [
      { filename: "scene_001.md", summary: "First", heat: 10, created: "2026-01-01", updated: "2026-01-02" },
      { filename: "scene_002.md", summary: "Second", heat: 20, created: "2026-02-01", updated: "2026-02-02" },
    ];
    await fs.writeFile(path.join(metaDir, "scene_index.json"), JSON.stringify(data), "utf-8");

    const entries = await readSceneIndex(tmpDir);
    expect(entries).toEqual(data);
  });

  it("should return empty array on malformed JSON", async () => {
    const metaDir = path.join(tmpDir, ".metadata");
    await fs.mkdir(metaDir, { recursive: true });
    await fs.writeFile(path.join(metaDir, "scene_index.json"), "not valid json{{{", "utf-8");

    const entries = await readSceneIndex(tmpDir);
    expect(entries).toEqual([]);
  });

  it("should skip entries with no filename", async () => {
    const metaDir = path.join(tmpDir, ".metadata");
    await fs.mkdir(metaDir, { recursive: true });

    const data = [
      { filename: "valid.md", summary: "ok", heat: 1, created: "", updated: "" },
      { summary: "no filename or path", heat: 2, created: "", updated: "" },
      { filename: "", summary: "empty filename", heat: 3, created: "", updated: "" },
    ];
    await fs.writeFile(path.join(metaDir, "scene_index.json"), JSON.stringify(data), "utf-8");

    const entries = await readSceneIndex(tmpDir);
    expect(entries).toHaveLength(1);
    expect(entries[0]!.filename).toBe("valid.md");
  });

  it("should return empty array if JSON is not an array", async () => {
    const metaDir = path.join(tmpDir, ".metadata");
    await fs.mkdir(metaDir, { recursive: true });
    await fs.writeFile(path.join(metaDir, "scene_index.json"), '{"not": "array"}', "utf-8");

    const entries = await readSceneIndex(tmpDir);
    expect(entries).toEqual([]);
  });

});

// ── writeSceneIndex ──

describe("writeSceneIndex", () => {
  it("should write index and create .metadata dir if needed", async () => {
    const data: SceneIndexEntry[] = [
      { filename: "s1.md", summary: "hello", heat: 5, created: "2026-01-01", updated: "" },
    ];
    await writeSceneIndex(tmpDir, data);

    const raw = await fs.readFile(path.join(tmpDir, ".metadata", "scene_index.json"), "utf-8");
    expect(JSON.parse(raw)).toEqual(data);
  });

  it("should overwrite existing index", async () => {
    const first: SceneIndexEntry[] = [
      { filename: "old.md", summary: "old", heat: 1, created: "", updated: "" },
    ];
    const second: SceneIndexEntry[] = [
      { filename: "new.md", summary: "new", heat: 99, created: "", updated: "" },
    ];
    await writeSceneIndex(tmpDir, first);
    await writeSceneIndex(tmpDir, second);

    const entries = await readSceneIndex(tmpDir);
    expect(entries).toEqual(second);
  });
});

// ── syncSceneIndex ──

describe("syncSceneIndex", () => {
  it("should return empty array when scene_blocks dir does not exist", async () => {
    const entries = await syncSceneIndex(tmpDir);
    expect(entries).toEqual([]);
  });

  it("should return empty array for empty scene_blocks dir", async () => {
    await fs.mkdir(path.join(tmpDir, "scene_blocks"), { recursive: true });
    const entries = await syncSceneIndex(tmpDir);
    expect(entries).toEqual([]);
  });

  it("should rebuild index from .md files in scene_blocks", async () => {
    const blocksDir = path.join(tmpDir, "scene_blocks");
    await fs.mkdir(blocksDir, { recursive: true });

    // Write two scene block files with META
    const block1 = [
      "-----META-START-----",
      "created: 2026-01-10",
      "updated: 2026-01-15",
      "summary: Project architecture decisions",
      "heat: 30",
      "-----META-END-----",
      "",
      "## Content",
      "Details about architecture.",
    ].join("\n");
    await fs.writeFile(path.join(blocksDir, "scene_001.md"), block1, "utf-8");

    const block2 = [
      "-----META-START-----",
      "created: 2026-02-01T00:00:00.000Z",
      "updated: 2026-02-10T00:00:00.000Z",
      "summary: User preferences",
      "heat: 55",
      "-----META-END-----",
      "",
      "User likes dark mode.",
    ].join("\n");
    await fs.writeFile(path.join(blocksDir, "scene_002.md"), block2, "utf-8");

    // Also write a non-md file that should be ignored
    await fs.writeFile(path.join(blocksDir, "readme.txt"), "ignore me", "utf-8");

    const entries = await syncSceneIndex(tmpDir);
    expect(entries).toHaveLength(2);

    const s1 = entries.find((e) => e.filename === "scene_001.md");
    expect(s1).toBeDefined();
    expect(s1!.summary).toBe("Project architecture decisions");
    expect(s1!.heat).toBe(30);

    const s2 = entries.find((e) => e.filename === "scene_002.md");
    expect(s2).toBeDefined();
    expect(s2!.summary).toBe("User preferences");
    expect(s2!.heat).toBe(55);

    // Verify index was also persisted to disk
    const persisted = await readSceneIndex(tmpDir);
    expect(persisted).toEqual(entries);
  });

  it("should handle scene blocks without META gracefully", async () => {
    const blocksDir = path.join(tmpDir, "scene_blocks");
    await fs.mkdir(blocksDir, { recursive: true });

    await fs.writeFile(path.join(blocksDir, "plain.md"), "No meta, just content.", "utf-8");

    const entries = await syncSceneIndex(tmpDir);
    expect(entries).toHaveLength(1);
    expect(entries[0]!.filename).toBe("plain.md");
    expect(entries[0]!.summary).toBe("");
    expect(entries[0]!.heat).toBe(0);
  });
});
