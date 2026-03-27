import { describe, it, expect } from "vitest";
import {
  generateSceneNavigation,
  stripSceneNavigation,
} from "./scene-navigation.js";
import type { SceneIndexEntry } from "./scene-index.js";

// ── generateSceneNavigation ──

describe("generateSceneNavigation", () => {
  it("should return empty string for empty entries", () => {
    expect(generateSceneNavigation([])).toBe("");
  });

  it("should generate navigation for a single entry", () => {
    const entries: SceneIndexEntry[] = [
      { filename: "scene_001.md", summary: "First scene", heat: 10, created: "2026-01-01T00:00:00.000Z", updated: "2026-01-02T00:00:00.000Z" },
    ];
    const nav = generateSceneNavigation(entries);
    expect(nav).toContain("## 🗺️ Scene Navigation");
    expect(nav).toContain("scene_blocks/scene_001.md");
    expect(nav).toContain("First scene");
    expect(nav).toContain("**热度**: 10");
    expect(nav).toContain("**更新**: 2026-01-02T00:00:00.000Z");
  });

  it("should sort entries by heat descending", () => {
    const entries: SceneIndexEntry[] = [
      { filename: "low.md", summary: "Low heat", heat: 5, created: "", updated: "" },
      { filename: "high.md", summary: "High heat", heat: 500, created: "", updated: "" },
      { filename: "mid.md", summary: "Mid heat", heat: 100, created: "", updated: "" },
    ];
    const nav = generateSceneNavigation(entries);
    const highIdx = nav.indexOf("high.md");
    const midIdx = nav.indexOf("mid.md");
    const lowIdx = nav.indexOf("low.md");
    expect(highIdx).toBeLessThan(midIdx);
    expect(midIdx).toBeLessThan(lowIdx);
  });

  it("should show correct fire emoji for heat thresholds", () => {
    const makeEntry = (heat: number): SceneIndexEntry => ({
      filename: `h${heat}.md`, summary: "", heat, created: "", updated: "",
    });

    // heat < 50: no emoji
    const nav0 = generateSceneNavigation([makeEntry(10)]);
    expect(nav0).not.toContain("🔥");

    // heat >= 50: 🔥
    const nav50 = generateSceneNavigation([makeEntry(50)]);
    expect(nav50).toContain("🔥");
    expect(nav50).not.toContain("🔥🔥");

    // heat >= 100: 🔥🔥
    const nav100 = generateSceneNavigation([makeEntry(100)]);
    expect(nav100).toContain("🔥🔥");
    expect(nav100).not.toContain("🔥🔥🔥");

    // heat >= 200: 🔥🔥🔥
    const nav200 = generateSceneNavigation([makeEntry(200)]);
    expect(nav200).toContain("🔥🔥🔥");
    expect(nav200).not.toContain("🔥🔥🔥🔥");

    // heat >= 500: 🔥🔥🔥🔥
    const nav500 = generateSceneNavigation([makeEntry(500)]);
    expect(nav500).toContain("🔥🔥🔥🔥");
    expect(nav500).not.toContain("🔥🔥🔥🔥🔥");

    // heat >= 1000: 🔥🔥🔥🔥🔥
    const nav1000 = generateSceneNavigation([makeEntry(1000)]);
    expect(nav1000).toContain("🔥🔥🔥🔥🔥");
  });

  it("should include usage instructions footer", () => {
    const entries: SceneIndexEntry[] = [
      { filename: "x.md", summary: "s", heat: 1, created: "", updated: "" },
    ];
    const nav = generateSceneNavigation(entries);
    expect(nav).toContain("📌 使用说明");
    expect(nav).toContain("read_file");
  });

  it("should omit updated section when updated is empty", () => {
    const entries: SceneIndexEntry[] = [
      { filename: "x.md", summary: "s", heat: 1, created: "2026-01-01T00:00:00.000Z", updated: "" },
    ];
    const nav = generateSceneNavigation(entries);
    expect(nav).not.toContain("**更新**");
  });

  it("should not mutate the original array", () => {
    const entries: SceneIndexEntry[] = [
      { filename: "a.md", summary: "a", heat: 1, created: "", updated: "" },
      { filename: "b.md", summary: "b", heat: 100, created: "", updated: "" },
    ];
    const copy = [...entries];
    generateSceneNavigation(entries);
    expect(entries).toEqual(copy);
  });
});

// ── stripSceneNavigation ──

describe("stripSceneNavigation", () => {
  it("should strip navigation section from persona content", () => {
    const persona = "# Persona\n\nSome body text.\n\n---\n## 🗺️ Scene Navigation (Scene Index)\nSome nav content...";
    const stripped = stripSceneNavigation(persona);
    expect(stripped).toBe("# Persona\n\nSome body text.");
    expect(stripped).not.toContain("Scene Navigation");
  });

  it("should return content unchanged when no navigation present", () => {
    const persona = "# Persona\n\nJust body text.";
    expect(stripSceneNavigation(persona)).toBe(persona);
  });

  it("should handle content that is only navigation", () => {
    const persona = "---\n## 🗺️ Scene Navigation (Scene Index)\nAll nav.";
    const stripped = stripSceneNavigation(persona);
    expect(stripped).toBe("");
  });

  it("should trim trailing whitespace before navigation", () => {
    const persona = "# Persona\n\n\n   \n---\n## 🗺️ Scene Navigation (Scene Index)\nnav";
    const stripped = stripSceneNavigation(persona);
    expect(stripped).toBe("# Persona");
  });
});
