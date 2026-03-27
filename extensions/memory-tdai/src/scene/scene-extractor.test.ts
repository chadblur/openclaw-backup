import { describe, it, expect, vi, beforeEach } from "vitest";
import path from "node:path";

// ============================
// Module-level mocks (must be before import)
// ============================

// -- CleanContextRunner: the LLM gateway — this is the key mock that replaces the model
const mockRunnerRun = vi.fn<(opts: Record<string, unknown>) => Promise<string>>().mockResolvedValue("LLM output");
vi.mock("../utils/clean-context-runner.js", () => {
  const MockCleanContextRunner = class {
    constructor(_opts: unknown) {
      // no-op
    }
    run = mockRunnerRun;
  };
  return { CleanContextRunner: MockCleanContextRunner };
});

// -- CheckpointManager
const mockCheckpointRead = vi.fn().mockResolvedValue({ total_processed: 42 });
const mockSetPersonaUpdateRequest = vi.fn().mockResolvedValue(undefined);
vi.mock("../utils/checkpoint.js", () => {
  const MockCheckpointManager = class {
    read = mockCheckpointRead;
    setPersonaUpdateRequest = mockSetPersonaUpdateRequest;
  };
  return { CheckpointManager: MockCheckpointManager };
});

// -- BackupManager
const mockBackupDirectory = vi.fn().mockResolvedValue(undefined);
vi.mock("../utils/backup.js", () => {
  const MockBackupManager = class {
    backupDirectory = mockBackupDirectory;
  };
  return { BackupManager: MockBackupManager };
});

// -- scene-index
const mockReadSceneIndex = vi.fn().mockResolvedValue([]);
const mockSyncSceneIndex = vi.fn().mockResolvedValue([]);
vi.mock("../scene/scene-index.js", () => ({
  readSceneIndex: (...args: unknown[]) => mockReadSceneIndex(...args),
  syncSceneIndex: (...args: unknown[]) => mockSyncSceneIndex(...args),
}));

// -- scene-navigation
const mockGenerateSceneNavigation = vi.fn().mockReturnValue("");
const mockStripSceneNavigation = vi.fn((content: string) => content);
vi.mock("../scene/scene-navigation.js", () => ({
  generateSceneNavigation: (...args: unknown[]) => mockGenerateSceneNavigation(...args),
  stripSceneNavigation: (...args: unknown[]) => mockStripSceneNavigation(...(args as [string])),
}));

// -- buildSceneExtractionPrompt
const mockBuildPrompt = vi.fn().mockReturnValue("MOCK_PROMPT");
vi.mock("../prompts/scene-extraction.js", () => ({
  buildSceneExtractionPrompt: (...args: unknown[]) => mockBuildPrompt(...args),
}));

// -- node:fs/promises
const mockMkdir = vi.fn().mockResolvedValue(undefined);
const mockReaddir = vi.fn().mockResolvedValue([]);
const mockReadFile = vi.fn().mockResolvedValue("");
const mockWriteFile = vi.fn().mockResolvedValue(undefined);
const mockUnlink = vi.fn().mockResolvedValue(undefined);
vi.mock("node:fs/promises", () => ({
  default: {
    mkdir: (...args: unknown[]) => mockMkdir(...args),
    readdir: (...args: unknown[]) => mockReaddir(...args),
    readFile: (...args: unknown[]) => mockReadFile(...args),
    writeFile: (...args: unknown[]) => mockWriteFile(...args),
    unlink: (...args: unknown[]) => mockUnlink(...args),
  },
}));

// Import AFTER mocks are set up
const { SceneExtractor, parsePersonaUpdateSignal } = await import("./scene-extractor.js");
import type { SceneIndexEntry } from "../scene/scene-index.js";

// ============================
// Helpers
// ============================

const DATA_DIR = "/tmp/test-scene-extractor";
const SCENE_BLOCKS_DIR = path.join(DATA_DIR, "scene_blocks");

function makeLogger() {
  return {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  };
}

function makeMemories(count: number) {
  return Array.from({ length: count }, (_, i) => ({
    content: `Memory ${i + 1}`,
    created_at: `2026-03-18T10:${String(i).padStart(2, "0")}:00Z`,
    id: `mem_${i + 1}`,
  }));
}

function makeSceneIndexEntries(count: number): SceneIndexEntry[] {
  return Array.from({ length: count }, (_, i) => ({
    filename: `scene_${i + 1}.md`,
    summary: `Scene ${i + 1} summary`,
    heat: (i + 1) * 10,
    created: "2026-03-01T10:00:00.000Z",
    updated: "2026-03-18T10:00:00.000Z",
  }));
}

function createExtractor(overrides?: Record<string, unknown>) {
  return new SceneExtractor({
    dataDir: DATA_DIR,
    config: { agents: { defaults: {} } },
    logger: makeLogger(),
    ...overrides,
  });
}

// ============================
// Tests
// ============================

describe("SceneExtractor", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset default return values
    mockCheckpointRead.mockResolvedValue({ total_processed: 42 });
    mockSetPersonaUpdateRequest.mockResolvedValue(undefined);
    mockReadSceneIndex.mockResolvedValue([]);
    mockSyncSceneIndex.mockResolvedValue([]);
    mockGenerateSceneNavigation.mockReturnValue("");
    mockStripSceneNavigation.mockImplementation((content: string) => content);
    mockBuildPrompt.mockReturnValue("MOCK_PROMPT");
    mockRunnerRun.mockResolvedValue("LLM output");
    mockReaddir.mockResolvedValue([]);
    mockReadFile.mockResolvedValue("");
  });

  // ─── Constructor ──────────────────────────────────────────────

  describe("constructor", () => {
    it("should apply default values for optional params", () => {
      const extractor = new SceneExtractor({
        dataDir: DATA_DIR,
        config: {},
      });
      expect(extractor).toBeDefined();
    });

    it("should accept custom maxScenes, sceneBackupCount, and timeoutMs", () => {
      const logger = makeLogger();
      const extractor = new SceneExtractor({
        dataDir: DATA_DIR,
        config: {},
        model: "azure/gpt-5.2",
        maxScenes: 5,
        sceneBackupCount: 3,
        timeoutMs: 60_000,
        logger,
      });
      expect(extractor).toBeDefined();
      expect(logger.debug).toHaveBeenCalledWith(
        expect.stringContaining("maxScenes=5"),
      );
    });
  });

  // ─── extract() — empty input ─────────────────────────────────

  describe("extract() — empty input", () => {
    it("should return success with 0 processed when memories is empty", async () => {
      const extractor = createExtractor();
      const result = await extractor.extract([]);
      expect(result).toEqual({ memoriesProcessed: 0, success: true });
      expect(mockMkdir).not.toHaveBeenCalled();
      expect(mockRunnerRun).not.toHaveBeenCalled();
    });
  });

  // ─── extract() — Phase 1-3: Backup + Index + Prompt ──────────

  describe("extract() — backup, index, prompt phases", () => {
    it("should create directories, read checkpoint, backup, load index, and build prompt", async () => {
      const memories = makeMemories(2);
      const extractor = createExtractor();

      await extractor.extract(memories);

      // mkdir for scene_blocks and .metadata
      expect(mockMkdir).toHaveBeenCalledWith(SCENE_BLOCKS_DIR, { recursive: true });
      expect(mockMkdir).toHaveBeenCalledWith(
        path.join(DATA_DIR, ".metadata"),
        { recursive: true },
      );

      // CheckpointManager.read
      expect(mockCheckpointRead).toHaveBeenCalledOnce();

      // BackupManager.backupDirectory
      expect(mockBackupDirectory).toHaveBeenCalledWith(
        SCENE_BLOCKS_DIR,
        "scene_blocks",
        "offset42",
        10,
      );

      // readSceneIndex
      expect(mockReadSceneIndex).toHaveBeenCalledWith(DATA_DIR);

      // buildSceneExtractionPrompt called with correct structure (no sceneIndexPath/checkpointPath)
      expect(mockBuildPrompt).toHaveBeenCalledOnce();
      const promptArgs = mockBuildPrompt.mock.calls[0]![0] as Record<string, unknown>;
      expect(promptArgs.sceneSummaries).toBe("(无已有场景)");
      expect(promptArgs.sceneCountWarning).toBeUndefined();
      expect(promptArgs.existingSceneFiles).toEqual([]);
      // These should NOT be present
      expect(promptArgs).not.toHaveProperty("sceneIndexPath");
      expect(promptArgs).not.toHaveProperty("checkpointPath");
      expect(promptArgs).not.toHaveProperty("sceneBlocksDir");
    });

    it("should pass scene summaries with relative filenames when index is non-empty", async () => {
      const entries = makeSceneIndexEntries(2);
      mockReadSceneIndex.mockResolvedValue(entries);

      const extractor = createExtractor();
      await extractor.extract(makeMemories(1));

      const promptArgs = mockBuildPrompt.mock.calls[0]![0] as Record<string, unknown>;
      // summaries should contain scene info with relative names
      expect(promptArgs.sceneSummaries).toContain("scene_1.md");
      expect(promptArgs.sceneSummaries).toContain("Scene 1 summary");
      // filenames should be relative (not absolute paths)
      expect(promptArgs.existingSceneFiles).toEqual([
        "scene_1.md",
        "scene_2.md",
      ]);
    });

    it("should add tier-1 (FORCE MERGE) sceneCountWarning when index.length >= maxScenes", async () => {
      const entries = makeSceneIndexEntries(5);
      mockReadSceneIndex.mockResolvedValue(entries);

      const logger = makeLogger();
      const extractor = new SceneExtractor({
        dataDir: DATA_DIR,
        config: {},
        maxScenes: 5,
        logger,
      });

      await extractor.extract(makeMemories(1));

      const promptArgs = mockBuildPrompt.mock.calls[0]![0] as Record<string, unknown>;
      expect(promptArgs.sceneCountWarning).toContain("5");
      expect(promptArgs.sceneCountWarning).toContain("上限");
      expect(promptArgs.sceneCountWarning).toContain("必须先执行 MERGE 操作");
      expect(logger.warn).toHaveBeenCalledWith(
        expect.stringContaining("scene count at limit"),
      );
    });

    it("should add tier-2 (CREATE blocked) warning when index.length == maxScenes - 1", async () => {
      const entries = makeSceneIndexEntries(14);
      mockReadSceneIndex.mockResolvedValue(entries);

      const logger = makeLogger();
      const extractor = new SceneExtractor({
        dataDir: DATA_DIR,
        config: {},
        maxScenes: 15,
        logger,
      });

      await extractor.extract(makeMemories(1));

      const promptArgs = mockBuildPrompt.mock.calls[0]![0] as Record<string, unknown>;
      expect(promptArgs.sceneCountWarning).toContain("14");
      expect(promptArgs.sceneCountWarning).toContain("不能 CREATE 新场景");
      expect(logger.warn).toHaveBeenCalledWith(
        expect.stringContaining("CREATE blocked"),
      );
    });

    it("should add tier-3 (prefer UPDATE) warning when index.length >= maxScenes - 3", async () => {
      const entries = makeSceneIndexEntries(12);
      mockReadSceneIndex.mockResolvedValue(entries);

      const logger = makeLogger();
      const extractor = new SceneExtractor({
        dataDir: DATA_DIR,
        config: {},
        maxScenes: 15,
        logger,
      });

      await extractor.extract(makeMemories(1));

      const promptArgs = mockBuildPrompt.mock.calls[0]![0] as Record<string, unknown>;
      expect(promptArgs.sceneCountWarning).toContain("12");
      expect(promptArgs.sceneCountWarning).toContain("优先考虑 UPDATE");
      expect(logger.debug).toHaveBeenCalledWith(
        expect.stringContaining("approaching limit"),
      );
    });

    it("should NOT add sceneCountWarning when index.length is well below threshold", async () => {
      const entries = makeSceneIndexEntries(10);
      mockReadSceneIndex.mockResolvedValue(entries);

      const extractor = new SceneExtractor({
        dataDir: DATA_DIR,
        config: {},
        maxScenes: 15,
      });

      await extractor.extract(makeMemories(1));

      const promptArgs = mockBuildPrompt.mock.calls[0]![0] as Record<string, unknown>;
      expect(promptArgs.sceneCountWarning).toBeUndefined();
    });

    it("should use custom sceneBackupCount", async () => {
      const extractor = createExtractor({ sceneBackupCount: 3 });
      await extractor.extract(makeMemories(1));

      expect(mockBackupDirectory).toHaveBeenCalledWith(
        expect.any(String),
        "scene_blocks",
        "offset42",
        3,
      );
    });
  });

  // ─── extract() — Phase 4: LLM runner ────────────────────────

  describe("extract() — LLM runner", () => {
    it("should call runner.run with workspaceDir = scene_blocks (sandboxed)", async () => {
      const extractor = createExtractor({ timeoutMs: 60_000 });
      await extractor.extract(makeMemories(2));

      expect(mockRunnerRun).toHaveBeenCalledOnce();
      const runArgs = mockRunnerRun.mock.calls[0]![0] as Record<string, unknown>;
      expect(runArgs.prompt).toBe("MOCK_PROMPT");
      expect(runArgs.timeoutMs).toBe(60_000);
      expect(runArgs.maxTokens).toBeUndefined();
      // KEY: workspaceDir should be scene_blocks, not dataDir
      expect(runArgs.workspaceDir).toBe(SCENE_BLOCKS_DIR);
      expect((runArgs.taskId as string).startsWith("scene-extract-")).toBe(true);
    });

    it("should return {success: false, error} when runner.run throws", async () => {
      mockRunnerRun.mockRejectedValue(new Error("LLM timeout"));

      const logger = makeLogger();
      const extractor = createExtractor({ logger });
      const result = await extractor.extract(makeMemories(1));

      expect(result).toEqual({
        memoriesProcessed: 0,
        success: false,
        error: "LLM timeout",
      });
      expect(logger.error).toHaveBeenCalledWith(
        expect.stringContaining("LLM runner failed"),
      );
      expect(mockSyncSceneIndex).not.toHaveBeenCalled();
    });

    it("should handle non-Error thrown values", async () => {
      mockRunnerRun.mockRejectedValue("string error");

      const extractor = createExtractor();
      const result = await extractor.extract(makeMemories(1));

      expect(result).toEqual({
        memoriesProcessed: 0,
        success: false,
        error: "string error",
      });
    });
  });

  // ─── extract() — Phase 5: Soft-delete cleanup ───────────────

  describe("extract() — soft-delete cleanup", () => {
    it("should unlink empty .md files", async () => {
      mockReaddir.mockResolvedValue(["scene_a.md", "scene_b.md", "notes.txt"]);
      mockReadFile.mockImplementation(async (filePath: string) => {
        if (typeof filePath === "string" && filePath.includes("scene_a.md")) return "# Content";
        if (typeof filePath === "string" && filePath.includes("scene_b.md")) return "   \n  ";
        return "";
      });

      const logger = makeLogger();
      const extractor = createExtractor({ logger });
      await extractor.extract(makeMemories(1));

      expect(mockUnlink).toHaveBeenCalledTimes(1);
      expect(mockUnlink).toHaveBeenCalledWith(
        path.join(SCENE_BLOCKS_DIR, "scene_b.md"),
      );
    });

    it("should not unlink non-empty files", async () => {
      mockReaddir.mockResolvedValue(["scene_a.md"]);
      mockReadFile.mockResolvedValue("# Has content");

      const extractor = createExtractor();
      await extractor.extract(makeMemories(1));

      expect(mockUnlink).not.toHaveBeenCalled();
    });

    it("should continue to sync even if cleanup throws", async () => {
      mockReaddir.mockRejectedValue(new Error("readdir failed"));

      const logger = makeLogger();
      const extractor = createExtractor({ logger });
      const result = await extractor.extract(makeMemories(1));

      expect(logger.warn).toHaveBeenCalledWith(
        expect.stringContaining("soft-delete cleanup error"),
      );
      expect(mockSyncSceneIndex).toHaveBeenCalledWith(DATA_DIR);
      expect(result.success).toBe(true);
    });
  });

  // ─── extract() — Phase 6: syncSceneIndex ─────────────────────

  describe("extract() — syncSceneIndex", () => {
    it("should call syncSceneIndex with dataDir after cleanup", async () => {
      const extractor = createExtractor();
      await extractor.extract(makeMemories(1));

      expect(mockSyncSceneIndex).toHaveBeenCalledWith(DATA_DIR);
    });
  });

  // ─── extract() — Phase 7: updateSceneNavigation ──────────────

  describe("extract() — updateSceneNavigation", () => {
    it("should write updated persona.md when persona exists with body content", async () => {
      mockReadSceneIndex
        .mockResolvedValueOnce([]) // Phase 2
        .mockResolvedValueOnce(makeSceneIndexEntries(1)); // Phase 7

      mockGenerateSceneNavigation.mockReturnValue("## Nav Section");
      mockStripSceneNavigation.mockReturnValue("# Persona Body");

      mockReadFile.mockImplementation(async (filePath: string) => {
        if (typeof filePath === "string" && filePath.includes("persona.md")) {
          return "# Persona Body\n\n## Old Nav";
        }
        return "";
      });

      const extractor = createExtractor();
      await extractor.extract(makeMemories(1));

      expect(mockWriteFile).toHaveBeenCalledWith(
        path.join(DATA_DIR, "persona.md"),
        "# Persona Body\n\n## Nav Section\n",
        "utf-8",
      );
    });

    it("should skip writing when persona file does not exist (readFile throws)", async () => {
      mockReadFile.mockRejectedValue(new Error("ENOENT"));

      const logger = makeLogger();
      const extractor = createExtractor({ logger });
      await extractor.extract(makeMemories(1));

      expect(mockWriteFile).not.toHaveBeenCalled();
    });

    it("should skip writing when persona body is empty after stripping navigation", async () => {
      mockReadSceneIndex
        .mockResolvedValueOnce([])
        .mockResolvedValueOnce(makeSceneIndexEntries(1));

      mockGenerateSceneNavigation.mockReturnValue("## Nav");
      mockStripSceneNavigation.mockReturnValue("");

      mockReadFile.mockImplementation(async (filePath: string) => {
        if (typeof filePath === "string" && filePath.includes("persona.md")) {
          return "## Nav Only Content";
        }
        return "";
      });

      const logger = makeLogger();
      const extractor = createExtractor({ logger });
      await extractor.extract(makeMemories(1));

      expect(mockWriteFile).not.toHaveBeenCalled();
      expect(logger.debug).toHaveBeenCalledWith(
        expect.stringContaining("persona body is empty"),
      );
    });

    it("should write persona without nav when generateSceneNavigation returns empty", async () => {
      mockReadSceneIndex
        .mockResolvedValueOnce([])
        .mockResolvedValueOnce([]);

      mockGenerateSceneNavigation.mockReturnValue("");
      mockStripSceneNavigation.mockReturnValue("# Persona Content");

      mockReadFile.mockImplementation(async (filePath: string) => {
        if (typeof filePath === "string" && filePath.includes("persona.md")) {
          return "# Persona Content";
        }
        return "";
      });

      const extractor = createExtractor();
      await extractor.extract(makeMemories(1));

      expect(mockWriteFile).toHaveBeenCalledWith(
        path.join(DATA_DIR, "persona.md"),
        "# Persona Content\n",
        "utf-8",
      );
    });

    it("should not fail extract() when updateSceneNavigation throws", async () => {
      mockReadSceneIndex
        .mockResolvedValueOnce([])
        .mockRejectedValueOnce(new Error("index read failed"));

      const logger = makeLogger();
      const extractor = createExtractor({ logger });
      const result = await extractor.extract(makeMemories(1));

      expect(result).toEqual({ memoriesProcessed: 1, success: true });
      expect(logger.warn).toHaveBeenCalledWith(
        expect.stringContaining("failed to update persona navigation"),
      );
    });

    it("should skip when both existing.trim() and nav are empty", async () => {
      mockReadSceneIndex
        .mockResolvedValueOnce([])
        .mockResolvedValueOnce([]);

      mockGenerateSceneNavigation.mockReturnValue("");

      mockReadFile.mockImplementation(async (filePath: string) => {
        if (typeof filePath === "string" && filePath.includes("persona.md")) {
          return "   \n  ";
        }
        return "";
      });

      const extractor = createExtractor();
      await extractor.extract(makeMemories(1));

      expect(mockWriteFile).not.toHaveBeenCalled();
    });
  });

  // ─── extract() — Phase 8: Persona update signal parsing ──────

  describe("extract() — persona update signal (Phase 8)", () => {
    it("should call setPersonaUpdateRequest when LLM output contains signal", async () => {
      mockRunnerRun.mockResolvedValue(
        "Processing complete.\n[PERSONA_UPDATE_REQUEST]\nreason: 用户价值观发生重大转变\n[/PERSONA_UPDATE_REQUEST]\nDone.",
      );

      const logger = makeLogger();
      const extractor = createExtractor({ logger });
      await extractor.extract(makeMemories(1));

      expect(mockSetPersonaUpdateRequest).toHaveBeenCalledWith("用户价值观发生重大转变");
      expect(logger.debug).toHaveBeenCalledWith(
        expect.stringContaining("persona update requested by LLM"),
      );
    });

    it("should NOT call setPersonaUpdateRequest when LLM output has no signal", async () => {
      mockRunnerRun.mockResolvedValue("Processing complete. No persona changes needed.");

      const extractor = createExtractor();
      await extractor.extract(makeMemories(1));

      expect(mockSetPersonaUpdateRequest).not.toHaveBeenCalled();
    });

    it("should handle empty/undefined LLM output gracefully", async () => {
      mockRunnerRun.mockResolvedValue("");

      const extractor = createExtractor();
      const result = await extractor.extract(makeMemories(1));

      expect(result.success).toBe(true);
      expect(mockSetPersonaUpdateRequest).not.toHaveBeenCalled();
    });
  });

  // ─── extract() — return value ────────────────────────────────

  describe("extract() — return value", () => {
    it("should return correct memoriesProcessed count on success", async () => {
      const extractor = createExtractor();
      const result = await extractor.extract(makeMemories(5));

      expect(result).toEqual({ memoriesProcessed: 5, success: true });
    });

    it("should serialize memories with id defaulting to empty string", async () => {
      const memories = [
        { content: "No ID memory", created_at: "2026-03-18T10:00:00Z" },
      ];
      const extractor = createExtractor();
      await extractor.extract(memories);

      const promptArgs = mockBuildPrompt.mock.calls[0]![0] as Record<string, unknown>;
      const parsed = JSON.parse(promptArgs.memoriesJson as string) as Array<Record<string, string>>;
      expect(parsed[0]!.id).toBe("");
    });
  });

  // ─── buildSceneSummaries (private, tested through extract) ───

  describe("buildSceneSummaries (indirect)", () => {
    it("should pass empty summaries and filenames for empty index", async () => {
      mockReadSceneIndex.mockResolvedValue([]);

      const extractor = createExtractor();
      await extractor.extract(makeMemories(1));

      const promptArgs = mockBuildPrompt.mock.calls[0]![0] as Record<string, unknown>;
      expect(promptArgs.sceneSummaries).toBe("(无已有场景)");
      expect(promptArgs.existingSceneFiles).toEqual([]);
    });

    it("should build summaries with capacity counter and relative filenames", async () => {
      const entries: SceneIndexEntry[] = [
        { filename: "work.md", summary: "Work stuff", heat: 50, created: "2026-01-01T00:00:00.000Z", updated: "2026-03-18T00:00:00.000Z" },
        { filename: "hobby.md", summary: "Hobbies", heat: 30, created: "2026-02-01T00:00:00.000Z", updated: "2026-03-17T00:00:00.000Z" },
      ];
      mockReadSceneIndex.mockResolvedValue(entries);

      const extractor = createExtractor();
      await extractor.extract(makeMemories(1));

      const promptArgs = mockBuildPrompt.mock.calls[0]![0] as Record<string, unknown>;
      const summaries = promptArgs.sceneSummaries as string;

      // Should start with capacity counter
      expect(summaries).toContain("**当前场景总数：2 / 20**");

      // Should contain relative filenames (not absolute paths)
      expect(summaries).toContain("### work.md");
      expect(summaries).toContain("**热度**: 50");
      expect(summaries).toContain("**summary**: Work stuff");
      expect(summaries).toContain("### hobby.md");
      expect(summaries).toContain("**summary**: Hobbies");

      // Filenames should be relative
      expect(promptArgs.existingSceneFiles).toEqual(["work.md", "hobby.md"]);
    });

    it("should reflect custom maxScenes in capacity counter", async () => {
      const entries = makeSceneIndexEntries(3);
      mockReadSceneIndex.mockResolvedValue(entries);

      const extractor = createExtractor({ maxScenes: 10 });
      await extractor.extract(makeMemories(1));

      const promptArgs = mockBuildPrompt.mock.calls[0]![0] as Record<string, unknown>;
      const summaries = promptArgs.sceneSummaries as string;
      expect(summaries).toContain("**当前场景总数：3 / 10**");
    });
  });
});

// ============================
// parsePersonaUpdateSignal (exported utility)
// ============================

describe("parsePersonaUpdateSignal", () => {
  it("should parse block format with reason prefix", () => {
    const text = "Some output\n[PERSONA_UPDATE_REQUEST]\nreason: 价值观转变\n[/PERSONA_UPDATE_REQUEST]\nMore output";
    const result = parsePersonaUpdateSignal(text);
    expect(result).toEqual({ reason: "价值观转变" });
  });

  it("should parse block format without reason prefix", () => {
    const text = "[PERSONA_UPDATE_REQUEST]用户从技术转向管理[/PERSONA_UPDATE_REQUEST]";
    const result = parsePersonaUpdateSignal(text);
    expect(result).toEqual({ reason: "用户从技术转向管理" });
  });

  it("should parse inline format", () => {
    const text = "PERSONA_UPDATE_REQUEST: 发现重大价值观变化\n其他内容";
    const result = parsePersonaUpdateSignal(text);
    expect(result).toEqual({ reason: "发现重大价值观变化" });
  });

  it("should return null when no signal is present", () => {
    const text = "Normal LLM output with no special signals.";
    expect(parsePersonaUpdateSignal(text)).toBeNull();
  });

  it("should return null for empty string", () => {
    expect(parsePersonaUpdateSignal("")).toBeNull();
  });

  it("should trim whitespace from reason", () => {
    const text = "[PERSONA_UPDATE_REQUEST]  reason:   some reason   [/PERSONA_UPDATE_REQUEST]";
    const result = parsePersonaUpdateSignal(text);
    expect(result).toEqual({ reason: "some reason" });
  });

  it("should handle multiline reason in block format", () => {
    const text = "[PERSONA_UPDATE_REQUEST]\nreason: 用户从反对加班转向接受弹性工作\n[/PERSONA_UPDATE_REQUEST]";
    const result = parsePersonaUpdateSignal(text);
    expect(result).toEqual({ reason: "用户从反对加班转向接受弹性工作" });
  });
});
