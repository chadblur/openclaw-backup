import { describe, it, expect } from "vitest";
import {
  parseSceneBlock,
  formatSceneBlock,
  formatMeta,
  type SceneBlockMeta,
} from "./scene-format.js";

// ── parseSceneBlock ──

describe("parseSceneBlock", () => {
  const validRaw = [
    "-----META-START-----",
    "created: 2026-01-15 10:30:00",
    "updated: 2026-03-10 14:20:00",
    "summary: User loves TypeScript and Vim",
    "heat: 42",
    "-----META-END-----",
    "",
    "## Scene Content",
    "Some detailed memory content here.",
  ].join("\n");

  it("should parse a valid scene block", () => {
    const block = parseSceneBlock(validRaw, "scene_001.md");
    expect(block.filename).toBe("scene_001.md");
    expect(block.meta.created).toBe("2026-01-15 10:30:00");
    expect(block.meta.updated).toBe("2026-03-10 14:20:00");
    expect(block.meta.summary).toBe("User loves TypeScript and Vim");
    expect(block.meta.heat).toBe(42);
    expect(block.content).toContain("## Scene Content");
    expect(block.content).toContain("Some detailed memory content here.");
  });

  it("should parse a scene block with ISO timestamps", () => {
    const isoRaw = [
      "-----META-START-----",
      "created: 2026-01-15T10:30:00.000Z",
      "updated: 2026-03-10T14:20:00.000Z",
      "summary: User loves TypeScript and Vim",
      "heat: 42",
      "-----META-END-----",
      "",
      "## Scene Content",
      "Some detailed memory content here.",
    ].join("\n");
    const block = parseSceneBlock(isoRaw, "scene_iso.md");
    expect(block.filename).toBe("scene_iso.md");
    expect(block.meta.created).toBe("2026-01-15T10:30:00.000Z");
    expect(block.meta.updated).toBe("2026-03-10T14:20:00.000Z");
    expect(block.meta.summary).toBe("User loves TypeScript and Vim");
    expect(block.meta.heat).toBe(42);
    expect(block.content).toContain("## Scene Content");
  });

  it("should treat entire file as content when META markers are missing", () => {
    const raw = "Just some plain text\nwith no meta section.";
    const block = parseSceneBlock(raw, "no-meta.md");
    expect(block.meta).toEqual({ created: "", updated: "", summary: "", heat: 0 });
    expect(block.content).toBe("Just some plain text\nwith no meta section.");
  });

  it("should handle missing META-END (only META-START present)", () => {
    const raw = "-----META-START-----\ncreated: 2026-01-01\nSome content after";
    const block = parseSceneBlock(raw, "broken.md");
    // Missing META-END → fallback
    expect(block.meta).toEqual({ created: "", updated: "", summary: "", heat: 0 });
    expect(block.content).toBe(raw.trim());
  });

  it("should default heat to 0 when non-numeric", () => {
    const raw = [
      "-----META-START-----",
      "created: 2026-01-01",
      "updated: 2026-01-02",
      "summary: test",
      "heat: not-a-number",
      "-----META-END-----",
      "content",
    ].join("\n");
    const block = parseSceneBlock(raw, "bad-heat.md");
    expect(block.meta.heat).toBe(0);
  });

  it("should default missing fields to empty string / 0", () => {
    const raw = [
      "-----META-START-----",
      "-----META-END-----",
      "only content",
    ].join("\n");
    const block = parseSceneBlock(raw, "empty-meta.md");
    expect(block.meta.created).toBe("");
    expect(block.meta.updated).toBe("");
    expect(block.meta.summary).toBe("");
    expect(block.meta.heat).toBe(0);
    expect(block.content).toBe("only content");
  });

  it("should handle leading whitespace: trim() strips metaBlock head, inner lines keep indent", () => {
    // metaBlock = raw between META_START / META_END, then .trim()
    // → first field's leading spaces are stripped (it becomes the start of trimmed block)
    // → subsequent fields still have leading spaces → regex ^field: won't match them
    const raw = [
      "-----META-START-----",
      "  created:   2026-03-01 08:00:00  ",
      "  summary:   trimmed value  ",
      "  heat:  7  ",
      "-----META-END-----",
      "",
      "  content with leading space  ",
    ].join("\n");
    const block = parseSceneBlock(raw, "ws.md");
    // First field matched (trim removed leading whitespace of whole metaBlock)
    expect(block.meta.created).toBe("2026-03-01 08:00:00");
    // Inner lines still have leading spaces → no match → defaults
    expect(block.meta.summary).toBe("");
    expect(block.meta.heat).toBe(0);
  });

  it("should trim values when fields start at column 0", () => {
    const raw = [
      "-----META-START-----",
      "created:   2026-03-01 08:00:00  ",
      "summary:   trimmed value  ",
      "heat:  7  ",
      "-----META-END-----",
      "content",
    ].join("\n");
    const block = parseSceneBlock(raw, "trim.md");
    expect(block.meta.created).toBe("2026-03-01 08:00:00");
    expect(block.meta.summary).toBe("trimmed value");
    expect(block.meta.heat).toBe(7);
  });
});

// ── formatMeta ──

describe("formatMeta", () => {
  it("should format meta into the expected block", () => {
    const meta: SceneBlockMeta = {
      created: "2026-01-15 10:30:00",
      updated: "2026-03-10 14:20:00",
      summary: "Test summary",
      heat: 5,
    };
    const result = formatMeta(meta);
    expect(result).toBe(
      [
        "-----META-START-----",
        "created: 2026-01-15 10:30:00",
        "updated: 2026-03-10 14:20:00",
        "summary: Test summary",
        "heat: 5",
        "-----META-END-----",
      ].join("\n"),
    );
  });
  it("should format meta with ISO timestamps", () => {
    const meta: SceneBlockMeta = {
      created: "2026-01-15T10:30:00.000Z",
      updated: "2026-03-10T14:20:00.000Z",
      summary: "Test summary",
      heat: 5,
    };
    const result = formatMeta(meta);
    expect(result).toBe(
      [
        "-----META-START-----",
        "created: 2026-01-15T10:30:00.000Z",
        "updated: 2026-03-10T14:20:00.000Z",
        "summary: Test summary",
        "heat: 5",
        "-----META-END-----",
      ].join("\n"),
    );
  });
});

// ── formatSceneBlock ──

describe("formatSceneBlock", () => {
  it("should combine meta and content", () => {
    const meta: SceneBlockMeta = {
      created: "2026-01-01",
      updated: "2026-01-02",
      summary: "hello",
      heat: 10,
    };
    const result = formatSceneBlock(meta, "## My Content\nDetails.");
    expect(result).toContain("-----META-START-----");
    expect(result).toContain("-----META-END-----");
    expect(result).toContain("## My Content\nDetails.");
  });

  it("should round-trip: parse(format(meta, content)) ≈ original", () => {
    const meta: SceneBlockMeta = {
      created: "2026-02-20 12:00:00",
      updated: "2026-03-15 09:30:00",
      summary: "Round-trip test",
      heat: 99,
    };
    const content = "## Heading\n\nParagraph with details.\n\n- bullet 1\n- bullet 2";
    const formatted = formatSceneBlock(meta, content);
    const parsed = parseSceneBlock(formatted, "roundtrip.md");
    expect(parsed.meta).toEqual(meta);
    expect(parsed.content).toBe(content);
  });

  it("should round-trip with ISO timestamps", () => {
    const meta: SceneBlockMeta = {
      created: "2026-02-20T12:00:00.000Z",
      updated: "2026-03-15T09:30:00.000Z",
      summary: "ISO round-trip test",
      heat: 42,
    };
    const content = "## ISO Content\n\nDetails with ISO timestamps.";
    const formatted = formatSceneBlock(meta, content);
    const parsed = parseSceneBlock(formatted, "iso-roundtrip.md");
    expect(parsed.meta).toEqual(meta);
    expect(parsed.content).toBe(content);
  });
});
