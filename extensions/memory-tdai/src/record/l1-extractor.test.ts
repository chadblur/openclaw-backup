/**
 * Unit tests for L1 Memory Extractor (A 同学自测).
 *
 * Strategy:
 * - parseExtractionResult: pure JSON parsing → full coverage (via extractL1Memories with mocked LLM)
 * - normalizeType: pure mapping function → full coverage (via extractL1Memories)
 * - extractL1Memories pipeline: mock CleanContextRunner to avoid real LLM calls
 *
 * NOTE: callLlmExtraction is private and uses CleanContextRunner internally.
 * We mock CleanContextRunner.prototype.run to control LLM responses.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";

// Shared mock run function — each test sets its behavior via mockRunFn.mockXxx()
const mockRunFn = vi.fn();

vi.mock("../utils/clean-context-runner.js", () => ({
  CleanContextRunner: function CleanContextRunner() {
    return { run: mockRunFn };
  },
}));

import { extractL1Memories } from "./l1-extractor.js";
import type { L1ExtractionResult } from "./l1-extractor.js";
import type { ConversationMessage } from "../conversation/l0-recorder.js";

let testDir: string;
const mkDir = async () => {
  const d = path.join(os.tmpdir(), `l1-ext-test-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`);
  await fs.mkdir(d, { recursive: true });
  return d;
};
const rmDir = async (d: string) => {
  try { await fs.rm(d, { recursive: true, force: true }); } catch { /* ignore */ }
};
const logger = () => ({ debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() });

function makeMessage(role: "user" | "assistant", content: string, ts?: number): ConversationMessage {
  return {
    id: `msg_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
    role,
    content,
    timestamp: ts ?? Date.now(),
  };
}

/** Helper: build a valid LLM extraction response JSON string */
function buildLlmResponse(scenes: Array<{
  scene_name: string;
  message_ids?: string[];
  memories: Array<{
    content: string;
    type?: string;
    priority?: number;
    source_message_ids?: string[];
    metadata?: Record<string, unknown>;
  }>;
}>): string {
  return JSON.stringify(scenes.map((s) => ({
    scene_name: s.scene_name,
    message_ids: s.message_ids ?? [],
    memories: s.memories.map((m) => ({
      content: m.content,
      type: m.type ?? "persona",
      priority: m.priority ?? 50,
      source_message_ids: m.source_message_ids ?? [],
      metadata: m.metadata ?? {},
    })),
  })));
}

beforeEach(async () => {
  testDir = await mkDir();
  mockRunFn.mockReset();
});
afterEach(async () => { await rmDir(testDir); });

// ============================
// parseExtractionResult (tested indirectly through extractL1Memories)
// ============================

describe("parseExtractionResult (via extractL1Memories)", () => {
  it("should parse valid JSON array response", async () => {
    const response = buildLlmResponse([{
      scene_name: "编程讨论",
      memories: [
        { content: "用户喜欢使用 TypeScript 编程语言开发项目", type: "persona", priority: 80 },
        { content: "用户正在开发一个记忆系统的插件项目", type: "episodic", priority: 60 },
      ],
    }]);
    mockRunFn.mockResolvedValue(response);

    const result = await extractL1Memories({
      messages: [makeMessage("user", "I love TypeScript"), makeMessage("assistant", "Great choice!")],
      sessionKey: "test", baseDir: testDir, config: {},
      options: { enableDedup: false },
    });

    expect(result.success).toBe(true);
    expect(result.extractedCount).toBe(2);
    expect(result.storedCount).toBe(2);
    expect(result.sceneNames).toEqual(["编程讨论"]);
    expect(result.lastSceneName).toBe("编程讨论");
  });

  it("should handle markdown-wrapped JSON (```json ... ```)", async () => {
    const json = buildLlmResponse([{
      scene_name: "test-scene",
      memories: [{ content: "用户偏好暗色主题进行代码编辑和开发", type: "persona", priority: 70 }],
    }]);
    mockRunFn.mockResolvedValue("```json\n" + json + "\n```");

    const result = await extractL1Memories({
      messages: [makeMessage("user", "I prefer dark mode")],
      sessionKey: "test", baseDir: testDir, config: {},
      options: { enableDedup: false },
    });

    expect(result.success).toBe(true);
    expect(result.extractedCount).toBe(1);
  });

  it("should handle markdown-wrapped JSON with just ``` (no json tag)", async () => {
    const json = buildLlmResponse([{
      scene_name: "scene1",
      memories: [{ content: "用户每天早上六点起床进行晨跑锻炼", type: "episodic" }],
    }]);
    mockRunFn.mockResolvedValue("```\n" + json + "\n```");

    const result = await extractL1Memories({
      messages: [makeMessage("user", "I run every morning")],
      sessionKey: "test", baseDir: testDir, config: {},
      options: { enableDedup: false },
    });

    expect(result.success).toBe(true);
    expect(result.extractedCount).toBe(1);
  });

  it("should extract JSON array from text with surrounding noise", async () => {
    const json = buildLlmResponse([{
      scene_name: "scene1",
      memories: [{ content: "用户对前端框架 React 有深入的学习经验", type: "persona" }],
    }]);
    mockRunFn.mockResolvedValue("Here is the result:\n" + json + "\nDone.");

    const result = await extractL1Memories({
      messages: [makeMessage("user", "I know React well")],
      sessionKey: "test", baseDir: testDir, config: {},
      options: { enableDedup: false },
    });

    expect(result.success).toBe(true);
    expect(result.extractedCount).toBe(1);
  });

  it("should return empty scenes when no JSON array found", async () => {
    const log = logger();
    mockRunFn.mockResolvedValue("This is not valid JSON at all");

    const result = await extractL1Memories({
      messages: [makeMessage("user", "Hello world test message")],
      sessionKey: "test", baseDir: testDir, config: {},
      options: { enableDedup: false }, logger: log,
    });

    expect(result.success).toBe(true);
    expect(result.extractedCount).toBe(0);
    expect(result.sceneNames).toEqual([]);
  });

  it("should return empty scenes for invalid JSON", async () => {
    const log = logger();
    mockRunFn.mockResolvedValue("[{invalid json content here}]");

    const result = await extractL1Memories({
      messages: [makeMessage("user", "Test message content")],
      sessionKey: "test", baseDir: testDir, config: {},
      options: { enableDedup: false }, logger: log,
    });

    expect(result.success).toBe(true);
    expect(result.extractedCount).toBe(0);
  });

  it("should handle non-array JSON response", async () => {
    mockRunFn.mockResolvedValue('{"not": "an array"}');

    const result = await extractL1Memories({
      messages: [makeMessage("user", "Test message for non-array")],
      sessionKey: "test", baseDir: testDir, config: {},
      options: { enableDedup: false },
    });

    expect(result.success).toBe(true);
    expect(result.extractedCount).toBe(0);
  });

  it("should skip non-object items in array", async () => {
    mockRunFn.mockResolvedValue(
      '[null, "string", 42, {"scene_name": "valid", "memories": [{"content": "用户使用 VS Code 编辑器进行日常开发工作", "type": "persona"}]}]',
    );

    const result = await extractL1Memories({
      messages: [makeMessage("user", "I use VS Code editor")],
      sessionKey: "test", baseDir: testDir, config: {},
      options: { enableDedup: false },
    });

    expect(result.success).toBe(true);
    expect(result.extractedCount).toBe(1);
  });

  it("should default scene_name to '未知情境' when missing", async () => {
    mockRunFn.mockResolvedValue(
      '[{"memories": [{"content": "用户对 Python 编程语言有一定的使用经验", "type": "persona"}]}]',
    );

    const result = await extractL1Memories({
      messages: [makeMessage("user", "I know Python well")],
      sessionKey: "test", baseDir: testDir, config: {},
      options: { enableDedup: false },
    });

    expect(result.sceneNames).toEqual(["未知情境"]);
  });

  it("should filter out memories with empty content but keep short content (length > 0)", async () => {
    const response = buildLlmResponse([{
      scene_name: "scene1",
      memories: [
        { content: "", type: "persona" },   // empty, filtered
        { content: "ab", type: "persona" },  // short but kept (length > 0)
        { content: "这是一条足够长度的有效记忆内容信息", type: "persona" }, // valid
      ],
    }]);
    mockRunFn.mockResolvedValue(response);

    const result = await extractL1Memories({
      messages: [makeMessage("user", "Test message here")],
      sessionKey: "test", baseDir: testDir, config: {},
      options: { enableDedup: false },
    });

    expect(result.extractedCount).toBe(2);
  });

  it("should default type to 'episodic' when missing", async () => {
    mockRunFn.mockResolvedValue(
      '[{"scene_name":"s1","memories":[{"content":"用户今天下午参加了一个重要的团队会议"}]}]',
    );

    const result = await extractL1Memories({
      messages: [makeMessage("user", "Had a meeting today")],
      sessionKey: "test", baseDir: testDir, config: {},
      options: { enableDedup: false },
    });

    expect(result.extractedCount).toBe(1);
    expect(result.records[0].type).toBe("episodic");
  });

  it("should default priority to 50 when missing or not a number", async () => {
    mockRunFn.mockResolvedValue(
      '[{"scene_name":"s1","memories":[{"content":"用户喜欢在安静的环境中进行深度工作", "type":"persona", "priority":"high"}]}]',
    );

    const result = await extractL1Memories({
      messages: [makeMessage("user", "I like quiet environments")],
      sessionKey: "test", baseDir: testDir, config: {},
      options: { enableDedup: false },
    });

    expect(result.records[0].priority).toBe(50);
  });

  it("should handle empty memories array in scene", async () => {
    mockRunFn.mockResolvedValue('[{"scene_name":"empty-scene","memories":[]}]');

    const result = await extractL1Memories({
      messages: [makeMessage("user", "Just casual chat here")],
      sessionKey: "test", baseDir: testDir, config: {},
      options: { enableDedup: false },
    });

    expect(result.success).toBe(true);
    expect(result.extractedCount).toBe(0);
    expect(result.sceneNames).toEqual(["empty-scene"]);
    expect(result.lastSceneName).toBe("empty-scene");
  });

  it("should handle missing memories field in scene (non-array)", async () => {
    mockRunFn.mockResolvedValue('[{"scene_name":"no-memories"}]');

    const result = await extractL1Memories({
      messages: [makeMessage("user", "Another chat message")],
      sessionKey: "test", baseDir: testDir, config: {},
      options: { enableDedup: false },
    });

    expect(result.extractedCount).toBe(0);
    expect(result.sceneNames).toEqual(["no-memories"]);
  });
});

// ============================
// normalizeType (tested indirectly through extractL1Memories)
// ============================

describe("normalizeType (via extractL1Memories)", () => {
  const testNormalizeType = async (inputType: string, expectedType: string | null) => {
    const content = "这是一条用于测试类型标准化的有效记忆内容";
    const response = buildLlmResponse([{
      scene_name: "test",
      memories: [{ content, type: inputType }],
    }]);
    mockRunFn.mockResolvedValue(response);
    const log = logger();

    const result = await extractL1Memories({
      messages: [makeMessage("user", "Test message")],
      sessionKey: "test", baseDir: testDir, config: {},
      options: { enableDedup: false }, logger: log,
    });

    if (expectedType) {
      expect(result.extractedCount).toBe(1);
      expect(result.records[0].type).toBe(expectedType);
    } else {
      expect(result.extractedCount).toBe(0);
      expect(log.warn).toHaveBeenCalled();
    }
  };

  it("should accept 'persona' as-is", async () => { await testNormalizeType("persona", "persona"); });
  it("should accept 'episodic' as-is", async () => { await testNormalizeType("episodic", "episodic"); });
  it("should accept 'instruction' as-is", async () => { await testNormalizeType("instruction", "instruction"); });
  it("should accept 'Persona' (case-insensitive)", async () => { await testNormalizeType("Persona", "persona"); });
  it("should accept 'EPISODIC' (case-insensitive)", async () => { await testNormalizeType("EPISODIC", "episodic"); });
  it("should map legacy 'episode' → 'episodic'", async () => { await testNormalizeType("episode", "episodic"); });
  it("should map legacy 'instruct' → 'instruction'", async () => { await testNormalizeType("instruct", "instruction"); });
  it("should map legacy 'preference' → 'persona'", async () => { await testNormalizeType("preference", "persona"); });
  it("should reject unknown type 'random'", async () => { await testNormalizeType("random", null); });
  it("should reject empty type string", async () => { await testNormalizeType("", null); });
});

// ============================
// extractL1Memories pipeline logic
// ============================

describe("extractL1Memories: empty messages fast return", () => {
  it("should return immediately for empty messages", async () => {
    const result = await extractL1Memories({
      messages: [],
      sessionKey: "test", baseDir: testDir, config: {},
    });

    expect(result).toEqual({
      success: true,
      extractedCount: 0,
      storedCount: 0,
      records: [],
      sceneNames: [],
    });
    // CleanContextRunner should not be invoked
    expect(mockRunFn).not.toHaveBeenCalled();
  });
});

describe("extractL1Memories: message splitting (background + new)", () => {
  it("should split messages correctly with default maxMessagesPerExtraction=10", async () => {
    const messages: ConversationMessage[] = [];
    for (let i = 0; i < 15; i++) {
      messages.push(makeMessage(i % 2 === 0 ? "user" : "assistant", `Message number ${i} content here`, 1000 + i));
    }

    mockRunFn.mockResolvedValue(buildLlmResponse([{
      scene_name: "chat", memories: [{ content: "这是从对话中提取出来的一条有效记忆信息", type: "persona" }],
    }]));

    await extractL1Memories({
      messages, sessionKey: "test", baseDir: testDir, config: {},
      options: { enableDedup: false },
    });

    expect(mockRunFn).toHaveBeenCalledTimes(1);
  });

  it("should respect maxMessagesPerExtraction option", async () => {
    const messages = Array.from({ length: 5 }, (_, i) =>
      makeMessage(i % 2 === 0 ? "user" : "assistant", `Message content number ${i}`, 1000 + i),
    );

    mockRunFn.mockResolvedValue(buildLlmResponse([{
      scene_name: "chat", memories: [{ content: "从少量消息中提取出的一条有效记忆内容", type: "persona" }],
    }]));

    await extractL1Memories({
      messages, sessionKey: "test", baseDir: testDir, config: {},
      options: { maxMessagesPerExtraction: 3, enableDedup: false },
    });

    expect(mockRunFn).toHaveBeenCalledTimes(1);
  });
});

describe("extractL1Memories: maxMemoriesPerSession truncation", () => {
  it("should truncate when extracted exceeds maxMemoriesPerSession", async () => {
    const memories = Array.from({ length: 15 }, (_, i) => ({
      content: `这是第${i + 1}条从对话中提取出来的有效记忆信息`,
      type: "persona" as const,
      priority: 50 + i,
    }));

    mockRunFn.mockResolvedValue(buildLlmResponse([{ scene_name: "big-scene", memories }]));
    const log = logger();

    const result = await extractL1Memories({
      messages: [makeMessage("user", "Lots of information here")],
      sessionKey: "test", baseDir: testDir, config: {},
      options: { maxMemoriesPerSession: 5, enableDedup: false },
      logger: log,
    });

    expect(result.extractedCount).toBe(5); // truncated from 15
    expect(result.storedCount).toBe(5);
    expect(log.debug).toHaveBeenCalledWith(expect.stringContaining("Limiting from 15 to 5"));
  });

  it("should not truncate when under limit", async () => {
    mockRunFn.mockResolvedValue(buildLlmResponse([{
      scene_name: "small",
      memories: [
        { content: "记忆内容一号，包含了用户的编程偏好信息", type: "persona" },
        { content: "记忆内容二号，记录了用户今天完成的任务", type: "episodic" },
      ],
    }]));

    const result = await extractL1Memories({
      messages: [makeMessage("user", "Some information here")],
      sessionKey: "test", baseDir: testDir, config: {},
      options: { maxMemoriesPerSession: 10, enableDedup: false },
    });

    expect(result.extractedCount).toBe(2);
  });
});

describe("extractL1Memories: LLM failure handling", () => {
  it("should return success=false when LLM call fails", async () => {
    mockRunFn.mockRejectedValue(new Error("LLM service unavailable"));
    const log = logger();

    const result = await extractL1Memories({
      messages: [makeMessage("user", "Hello world test msg")],
      sessionKey: "test", baseDir: testDir, config: {},
      logger: log,
    });

    expect(result.success).toBe(false);
    expect(result.extractedCount).toBe(0);
    expect(result.storedCount).toBe(0);
    expect(result.records).toEqual([]);
    expect(log.error).toHaveBeenCalledWith(expect.stringContaining("LLM extraction failed"));
  });

  it("should return success=false when LLM throws non-Error", async () => {
    mockRunFn.mockRejectedValue("string error");
    const log = logger();

    const result = await extractL1Memories({
      messages: [makeMessage("user", "Test message content")],
      sessionKey: "test", baseDir: testDir, config: {},
      logger: log,
    });

    expect(result.success).toBe(false);
    expect(log.error).toHaveBeenCalled();
  });
});

describe("extractL1Memories: multiple scenes", () => {
  it("should track scene names from multiple scenes", async () => {
    mockRunFn.mockResolvedValue(buildLlmResponse([
      { scene_name: "编程讨论", memories: [{ content: "用户正在学习 Rust 编程语言的基础知识", type: "episodic" }] },
      { scene_name: "生活闲聊", memories: [{ content: "用户喜欢在周末去公园散步放松心情", type: "persona" }] },
    ]));

    const result = await extractL1Memories({
      messages: [makeMessage("user", "Multiple topics discussed")],
      sessionKey: "test", baseDir: testDir, config: {},
      options: { enableDedup: false },
    });

    expect(result.sceneNames).toEqual(["编程讨论", "生活闲聊"]);
    expect(result.lastSceneName).toBe("生活闲聊");
    expect(result.extractedCount).toBe(2);
  });

  it("should handle scene with all invalid types → extractedCount=0", async () => {
    const log = logger();
    mockRunFn.mockResolvedValue(buildLlmResponse([{
      scene_name: "bad-types",
      memories: [
        { content: "内容一：这条记忆使用了无效的类型分类标签", type: "unknown_type_xyz" },
        { content: "内容二：这条记忆也使用了不合法的类型分类", type: "foobar_invalid" },
      ],
    }]));

    const result = await extractL1Memories({
      messages: [makeMessage("user", "Test message")],
      sessionKey: "test", baseDir: testDir, config: {},
      options: { enableDedup: false }, logger: log,
    });

    expect(result.extractedCount).toBe(0);
    expect(result.sceneNames).toEqual(["bad-types"]);
    expect(result.lastSceneName).toBe("bad-types");
    expect(log.warn).toHaveBeenCalledWith(expect.stringContaining("invalid type"));
  });
});

describe("extractL1Memories: enableDedup=false", () => {
  it("should store all directly without calling batchDedup", async () => {
    mockRunFn.mockResolvedValue(buildLlmResponse([{
      scene_name: "scene1",
      memories: [
        { content: "记忆A：用户对代码质量有着非常高的要求", type: "persona" },
        { content: "记忆B：用户今天完成了代码重构的工作", type: "episodic" },
      ],
    }]));

    const result = await extractL1Memories({
      messages: [makeMessage("user", "I care about code quality")],
      sessionKey: "test", baseDir: testDir, config: {},
      options: { enableDedup: false },
    });

    expect(result.success).toBe(true);
    expect(result.storedCount).toBe(2);

    // Verify JSONL files are written
    const recordsDir = path.join(testDir, "records");
    const files = await fs.readdir(recordsDir);
    expect(files).toHaveLength(1);
    const lines = (await fs.readFile(path.join(recordsDir, files[0]), "utf-8")).split("\n").filter(Boolean);
    expect(lines).toHaveLength(2);
  });
});

describe("extractL1Memories: enableDedup=true with no existing records", () => {
  it("should succeed when batchDedup finds no conflicts", async () => {
    mockRunFn.mockResolvedValue(buildLlmResponse([{
      scene_name: "test",
      memories: [{ content: "用户偏好使用暗色主题的编辑器进行开发", type: "persona" }],
    }]));

    const result = await extractL1Memories({
      messages: [makeMessage("user", "I prefer dark theme")],
      sessionKey: "test", baseDir: testDir, config: {},
      options: { enableDedup: true },
    });

    expect(result.success).toBe(true);
    expect(result.storedCount).toBe(1);
  });
});

describe("extractL1Memories: memory record fields", () => {
  it("should assign source_message_ids from LLM response", async () => {
    mockRunFn.mockResolvedValue(buildLlmResponse([{
      scene_name: "scene1",
      memories: [{
        content: "用户喜欢在工作中使用 TypeScript 语言编程",
        type: "persona",
        priority: 75,
        source_message_ids: ["msg_1", "msg_2"],
      }],
    }]));

    const result = await extractL1Memories({
      messages: [makeMessage("user", "I use TypeScript a lot")],
      sessionKey: "test-sess", sessionId: "sid-42",
      baseDir: testDir, config: {},
      options: { enableDedup: false },
    });

    const r = result.records[0];
    expect(r.source_message_ids).toEqual(["msg_1", "msg_2"]);
    expect(r.sessionKey).toBe("test-sess");
    expect(r.sessionId).toBe("sid-42");
    expect(r.scene_name).toBe("scene1");
    expect(r.priority).toBe(75);
  });

  it("should handle non-array source_message_ids gracefully", async () => {
    mockRunFn.mockResolvedValue(
      '[{"scene_name":"s1","memories":[{"content":"用户对机器学习算法有着浓厚的研究兴趣","type":"persona","source_message_ids":"not-an-array"}]}]',
    );

    const result = await extractL1Memories({
      messages: [makeMessage("user", "ML is interesting")],
      sessionKey: "test", baseDir: testDir, config: {},
      options: { enableDedup: false },
    });

    expect(result.records[0].source_message_ids).toEqual([]);
  });

  it("should handle metadata from LLM response", async () => {
    mockRunFn.mockResolvedValue(buildLlmResponse([{
      scene_name: "meeting",
      memories: [{
        content: "用户今天下午两点到四点参加了项目评审会议",
        type: "episodic",
        metadata: { activity_start_time: "2026-03-17T14:00:00Z", activity_end_time: "2026-03-17T16:00:00Z" },
      }],
    }]));

    const result = await extractL1Memories({
      messages: [makeMessage("user", "Meeting from 2-4pm")],
      sessionKey: "test", baseDir: testDir, config: {},
      options: { enableDedup: false },
    });

    expect(result.records[0].metadata).toEqual({
      activity_start_time: "2026-03-17T14:00:00Z",
      activity_end_time: "2026-03-17T16:00:00Z",
    });
  });

  it("should default metadata to empty object when null or non-object", async () => {
    mockRunFn.mockResolvedValue(
      '[{"scene_name":"s1","memories":[{"content":"用户通常在早上九点开始一天的工作","type":"persona","metadata":null}]}]',
    );

    const result = await extractL1Memories({
      messages: [makeMessage("user", "I start work at 9am")],
      sessionKey: "test", baseDir: testDir, config: {},
      options: { enableDedup: false },
    });

    expect(result.records[0].metadata).toEqual({});
  });
});

describe("extractL1Memories: sessionId handling", () => {
  it("should pass sessionId to stored records", async () => {
    mockRunFn.mockResolvedValue(buildLlmResponse([{
      scene_name: "s",
      memories: [{ content: "用户在开发过程中经常使用 Git 进行版本控制", type: "persona" }],
    }]));

    const result = await extractL1Memories({
      messages: [makeMessage("user", "I use git daily")],
      sessionKey: "sk", sessionId: "sid-123",
      baseDir: testDir, config: {},
      options: { enableDedup: false },
    });

    expect(result.records[0].sessionId).toBe("sid-123");
  });

  it("should default sessionId to empty string when not provided", async () => {
    mockRunFn.mockResolvedValue(buildLlmResponse([{
      scene_name: "s",
      memories: [{ content: "用户习惯在深夜进行代码审查和合并请求", type: "persona" }],
    }]));

    const result = await extractL1Memories({
      messages: [makeMessage("user", "Late night code review")],
      sessionKey: "sk",
      baseDir: testDir, config: {},
      options: { enableDedup: false },
    });

    expect(result.records).toHaveLength(1);
    expect(result.records[0].sessionId).toBe("");
  });
});

describe("extractL1Memories: options defaults", () => {
  it("should use default options when options is undefined", async () => {
    mockRunFn.mockResolvedValue(buildLlmResponse([{
      scene_name: "s",
      memories: [{ content: "用户对自动化测试工具有着深入的了解", type: "persona" }],
    }]));

    const result = await extractL1Memories({
      messages: [makeMessage("user", "I know testing tools well")],
      sessionKey: "sk", baseDir: testDir, config: {},
      // no options → defaults (enableDedup=true, but no existing records → store all)
    });

    expect(result.success).toBe(true);
    expect(result.storedCount).toBe(1);
  });
});

describe("extractL1Memories: JSONL persistence", () => {
  it("should write records to JSONL file with correct fields", async () => {
    mockRunFn.mockResolvedValue(buildLlmResponse([{
      scene_name: "coding",
      memories: [{ content: "用户偏好使用 Vim 键绑定进行代码编辑", type: "instruction", priority: 90 }],
    }]));

    const result = await extractL1Memories({
      messages: [makeMessage("user", "I use Vim keybindings")],
      sessionKey: "my-session", sessionId: "sid-99",
      baseDir: testDir, config: {},
      options: { enableDedup: false },
    });

    expect(result.storedCount).toBe(1);

    // Read and verify JSONL
    const recordsDir = path.join(testDir, "records");
    const files = await fs.readdir(recordsDir);
    expect(files).toHaveLength(1);
    const line = JSON.parse((await fs.readFile(path.join(recordsDir, files[0]), "utf-8")).trim());
    expect(line.content).toBe("用户偏好使用 Vim 键绑定进行代码编辑");
    expect(line.type).toBe("instruction");
    expect(line.priority).toBe(90);
    expect(line.sessionKey).toBe("my-session");
    expect(line.sessionId).toBe("sid-99");
    expect(line.scene_name).toBe("coding");
    expect(line.id).toMatch(/^m_/);
    expect(line.createdAt).toBeDefined();
    expect(line.updatedAt).toBeDefined();
  });
});
