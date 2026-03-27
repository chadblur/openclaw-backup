import { describe, it, expect } from "vitest";
import { buildSceneExtractionPrompt } from "./scene-extraction.js";
import type { SceneExtractionPromptParams } from "./scene-extraction.js";

// ============================
// Helpers
// ============================

function makeParams(overrides?: Partial<SceneExtractionPromptParams>): SceneExtractionPromptParams {
  return {
    memoriesJson: '[{"content":"test memory","created_at":"2026-03-18T10:00:00Z","id":"m1"}]',
    sceneSummaries: "(无已有场景)",
    currentTimestamp: "2026-03-18T10:00:00.000Z",
    ...overrides,
  };
}

// ============================
// Tests
// ============================

describe("buildSceneExtractionPrompt", () => {
  it("should embed all required params into the output", () => {
    const params = makeParams();
    const result = buildSceneExtractionPrompt(params);

    // Fixed structural elements
    expect(result).toContain("# System Prompt: Memory Consolidation Architect");
    expect(result).toContain("## 角色定义 (Role Definition)");

    // Interpolated values
    expect(result).toContain(params.memoriesJson);
    expect(result).toContain(params.sceneSummaries);
    expect(result).toContain(params.currentTimestamp);
  });

  it("should NOT contain sceneIndexPath or checkpointPath references", () => {
    const result = buildSceneExtractionPrompt(makeParams());

    expect(result).not.toContain("scene_index.json");
    expect(result).not.toContain("recall_checkpoint.json");
    expect(result).not.toContain("checkpointPath");
    expect(result).not.toContain("sceneIndexPath");
  });

  it("should use relative filenames, not absolute paths", () => {
    const result = buildSceneExtractionPrompt(makeParams({
      existingSceneFiles: ["work.md", "hobby.md"],
    }));

    expect(result).toContain("- `work.md`");
    expect(result).toContain("- `hobby.md`");
    // Should NOT contain absolute path patterns
    expect(result).not.toContain("/data/");
    expect(result).not.toContain("scene_blocks/");
  });

  it("should NOT include warning section when sceneCountWarning is undefined", () => {
    const result = buildSceneExtractionPrompt(makeParams({ sceneCountWarning: undefined }));

    expect(result).not.toContain("⚠️ **场景数量警告**");
  });

  it("should include warning section when sceneCountWarning is provided", () => {
    const warning = "⛔ 当前已有 15 个场景（上限 15），**已达上限**！必须先合并（MERGE）2-4个相似场景后才能处理新记忆，**严禁 CREATE**！";
    const result = buildSceneExtractionPrompt(makeParams({ sceneCountWarning: warning }));

    expect(result).toContain("⚠️ **场景数量警告**");
    expect(result).toContain(warning);
  });

  it("should list existing scene files when existingSceneFiles is non-empty", () => {
    const files = ["work.md", "hobby.md"];
    const result = buildSceneExtractionPrompt(makeParams({ existingSceneFiles: files }));

    expect(result).toContain("已有场景文件清单（仅以下文件可 read_file）");
    expect(result).toContain("- `work.md`");
    expect(result).toContain("- `hobby.md`");
    expect(result).not.toContain("当前无已有场景文件");
  });

  it("should show empty notice when existingSceneFiles is undefined or empty", () => {
    const result1 = buildSceneExtractionPrompt(makeParams({ existingSceneFiles: undefined }));
    const result2 = buildSceneExtractionPrompt(makeParams({ existingSceneFiles: [] }));

    for (const result of [result1, result2]) {
      expect(result).toContain("（当前无已有场景文件）");
      expect(result).not.toContain("仅以下文件可 read_file");
    }
  });

  it("should return a string (not throw) for minimal valid input", () => {
    const result = buildSceneExtractionPrompt(makeParams());
    expect(typeof result).toBe("string");
    expect(result.length).toBeGreaterThan(0);
  });

  // ─── Strategy / workflow tests ───────────────────────────────

  it("should state UPDATE as the default/preferred strategy", () => {
    const result = buildSceneExtractionPrompt(makeParams());

    expect(result).toContain("默认策略是 UPDATE，不是 CREATE");
    expect(result).toContain("【首选策略】");
    expect(result).toContain("当犹豫于 UPDATE 和 CREATE 之间时，选择 UPDATE");
  });

  it("should require read_file verification before CREATE", () => {
    const result = buildSceneExtractionPrompt(makeParams());

    expect(result).toContain("CREATE 前的强制验证");
    expect(result).toContain("read_file");
    expect(result).toContain("跳过验证直接 CREATE 是被禁止的");
  });

  it("should reference tiered warning system in workflow phase 0", () => {
    const result = buildSceneExtractionPrompt(makeParams());

    expect(result).toContain("遵守分级预警");
    expect(result).toContain("红色预警");
    expect(result).toContain("橙色预警");
    expect(result).toContain("黄色预警");
  });

  it("should set scene limit to 15 in architecture description", () => {
    const result = buildSceneExtractionPrompt(makeParams());

    expect(result).toContain("强制限制在15个以内");
    expect(result).not.toContain("强制限制在20个以内");
  });

  it("should order strategies as UPDATE > MERGE > CREATE", () => {
    const result = buildSceneExtractionPrompt(makeParams());

    const updatePos = result.indexOf("UPDATE（更新）");
    const mergePos = result.indexOf("MERGE（合并）");
    const createPos = result.indexOf("CREATE（新建）");

    expect(updatePos).toBeLessThan(mergePos);
    expect(mergePos).toBeLessThan(createPos);
  });

  it("should label CREATE as last resort", () => {
    const result = buildSceneExtractionPrompt(makeParams());

    expect(result).toContain("【最后手段】");
  });

  // ─── Persona update signal (out-of-band) ─────────────────────

  it("should include persona update signal instructions using text output", () => {
    const result = buildSceneExtractionPrompt(makeParams());

    expect(result).toContain("[PERSONA_UPDATE_REQUEST]");
    expect(result).toContain("[/PERSONA_UPDATE_REQUEST]");
    expect(result).toContain("text output");
    expect(result).toContain("触发方式");
    // Should NOT reference checkpoint file operations for persona update
    expect(result).not.toContain("request_persona_update");
    expect(result).not.toContain("checkpoint");
  });

  it("should instruct relative file operations only", () => {
    const result = buildSceneExtractionPrompt(makeParams());

    expect(result).toContain("相对文件名");
    expect(result).not.toContain("绝对路径");
    expect(result).not.toContain("必须使用绝对路径");
  });
});
