import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { BackupManager } from "./backup.js";

let tmpDir: string;
let backupRoot: string;
let mgr: BackupManager;

beforeEach(async () => {
  tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "backup-test-"));
  backupRoot = path.join(tmpDir, ".backup");
  mgr = new BackupManager(backupRoot);
});

afterEach(async () => {
  await fs.rm(tmpDir, { recursive: true, force: true });
});

// ────────────────────────────────────────
// backupFile
// ────────────────────────────────────────
describe("backupFile", () => {
  it("should copy a file into <backupRoot>/<category>/ with timestamped name", async () => {
    const src = path.join(tmpDir, "data.md");
    await fs.writeFile(src, "hello world");

    await mgr.backupFile(src, "persona", "offset42", 0);

    const entries = await fs.readdir(path.join(backupRoot, "persona"));
    expect(entries).toHaveLength(1);
    expect(entries[0]).toMatch(/^persona_\d{8}_\d{6}_offset42\.md$/);

    const content = await fs.readFile(path.join(backupRoot, "persona", entries[0]!), "utf-8");
    expect(content).toBe("hello world");
  });

  it("should silently skip when source file does not exist", async () => {
    await mgr.backupFile(path.join(tmpDir, "nonexistent.txt"), "cat", "t", 0);

    // backupRoot should not even be created
    await expect(fs.access(backupRoot)).rejects.toThrow();
  });

  it("should prune old backups when maxKeep > 0", async () => {
    const src = path.join(tmpDir, "data.json");
    await fs.writeFile(src, "{}");

    // Manually create 3 existing backup files with older timestamps
    const catDir = path.join(backupRoot, "cfg");
    await fs.mkdir(catDir, { recursive: true });
    await fs.writeFile(path.join(catDir, "cfg_20250101_000001_old1.json"), "1");
    await fs.writeFile(path.join(catDir, "cfg_20250101_000002_old2.json"), "2");
    await fs.writeFile(path.join(catDir, "cfg_20250101_000003_old3.json"), "3");

    // Now backup with maxKeep=2 — should keep 2 (the new one + newest old)
    await mgr.backupFile(src, "cfg", "new", 2);

    const entries = await fs.readdir(catDir);
    expect(entries).toHaveLength(2);
    // The newest entries should survive (sorted ascending, keep last 2)
    expect(entries.some((e) => e.includes("_new"))).toBe(true);
  });

  it("should not prune when maxKeep is 0", async () => {
    const src = path.join(tmpDir, "x.txt");
    await fs.writeFile(src, "x");

    const catDir = path.join(backupRoot, "unlimited");
    await fs.mkdir(catDir, { recursive: true });
    await fs.writeFile(path.join(catDir, "unlimited_20250101_000001_a.txt"), "a");
    await fs.writeFile(path.join(catDir, "unlimited_20250101_000002_b.txt"), "b");

    await mgr.backupFile(src, "unlimited", "c", 0);

    const entries = await fs.readdir(catDir);
    expect(entries).toHaveLength(3); // all preserved
  });
});

// ────────────────────────────────────────
// backupDirectory
// ────────────────────────────────────────
describe("backupDirectory", () => {
  it("should copy all files from source directory into a timestamped subdirectory", async () => {
    const srcDir = path.join(tmpDir, "scenes");
    await fs.mkdir(srcDir);
    await fs.writeFile(path.join(srcDir, "a.md"), "scene A");
    await fs.writeFile(path.join(srcDir, "b.md"), "scene B");

    await mgr.backupDirectory(srcDir, "scene_blocks", "offset10", 0);

    const parentDir = path.join(backupRoot, "scene_blocks");
    const entries = await fs.readdir(parentDir);
    expect(entries).toHaveLength(1);
    expect(entries[0]).toMatch(/^scene_blocks_\d{8}_\d{6}_offset10$/);

    const copied = await fs.readdir(path.join(parentDir, entries[0]!));
    expect(copied.sort()).toEqual(["a.md", "b.md"]);
  });

  it("should silently skip when source directory does not exist", async () => {
    await mgr.backupDirectory(path.join(tmpDir, "nope"), "cat", "t", 0);
    await expect(fs.access(backupRoot)).rejects.toThrow();
  });

  it("should silently skip when source directory is empty", async () => {
    const srcDir = path.join(tmpDir, "empty");
    await fs.mkdir(srcDir);

    await mgr.backupDirectory(srcDir, "cat", "t", 0);
    await expect(fs.access(backupRoot)).rejects.toThrow();
  });

  it("should prune old backup directories when maxKeep > 0", async () => {
    const srcDir = path.join(tmpDir, "scenes");
    await fs.mkdir(srcDir);
    await fs.writeFile(path.join(srcDir, "a.md"), "content");

    const parentDir = path.join(backupRoot, "scenes");
    // Create 2 existing backup dirs
    const old1 = path.join(parentDir, "scenes_20250101_000001_old1");
    const old2 = path.join(parentDir, "scenes_20250101_000002_old2");
    await fs.mkdir(old1, { recursive: true });
    await fs.writeFile(path.join(old1, "f.txt"), "old1");
    await fs.mkdir(old2, { recursive: true });
    await fs.writeFile(path.join(old2, "f.txt"), "old2");

    await mgr.backupDirectory(srcDir, "scenes", "new", 2);

    const entries = await fs.readdir(parentDir);
    expect(entries).toHaveLength(2);
    expect(entries.some((e) => e.includes("_new"))).toBe(true);
  });
});
