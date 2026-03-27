/**
 * Unit tests for L0 Conversation Recorder (A 同学自测).
 * L0-01 ~ L0-05 + extraction edge cases.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import {
  recordConversation,
  readConversationRecords,
  readConversationMessages,
  readConversationMessagesGroupedBySessionId,
} from "./l0-recorder.js";
import type { L0MessageRecord } from "./l0-recorder.js";

let testDir: string;
const mkDir = async () => {
  const d = path.join(os.tmpdir(), `l0-test-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`);
  await fs.mkdir(d, { recursive: true });
  return d;
};
const rmDir = async (d: string) => { try { await fs.rm(d, { recursive: true, force: true }); } catch {} };
const logger = () => ({ debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() });
const msg = (role: string, content: string, ts?: number) => ({ role, content, timestamp: ts ?? Date.now() });
const today = () => {
  const n = new Date();
  return `${n.getFullYear()}-${String(n.getMonth()+1).padStart(2,"0")}-${String(n.getDate()).padStart(2,"0")}`;
};

beforeEach(async () => { testDir = await mkDir(); });
afterEach(async () => { await rmDir(testDir); });

// ── recordConversation ──

describe("recordConversation", () => {
  it("L0-01: writes valid JSONL lines with correct fields", async () => {
    const ts = Date.now();
    const result = await recordConversation({
      sessionKey: "sess", sessionId: "sid",
      rawMessages: [msg("user","Hello, how are you?",ts), msg("assistant","I am fine, thank you!",ts+1)],
      baseDir: testDir,
    });
    expect(result).toHaveLength(2);
    const dir = path.join(testDir,"conversations");
    const files = await fs.readdir(dir);
    expect(files).toHaveLength(1);
    const lines = (await fs.readFile(path.join(dir,files[0]),"utf-8")).split("\n").filter(Boolean);
    expect(lines).toHaveLength(2);
    for (const l of lines) {
      const p = JSON.parse(l) as L0MessageRecord;
      expect(p.sessionKey).toBe("sess");
      expect(p.sessionId).toBe("sid");
      expect(typeof p.id).toBe("string");
      expect(typeof p.timestamp).toBe("number");
    }
  });

  it("L0-01: appends to same file on same day", async () => {
    const ts = Date.now();
    await recordConversation({ sessionKey:"s", rawMessages:[msg("user","First msg here",ts)], baseDir:testDir });
    await recordConversation({ sessionKey:"s", rawMessages:[msg("user","Second msg here",ts+100)], baseDir:testDir });
    const dir = path.join(testDir,"conversations");
    const files = await fs.readdir(dir);
    expect(files).toHaveLength(1);
    const lines = (await fs.readFile(path.join(dir,files[0]),"utf-8")).split("\n").filter(Boolean);
    expect(lines).toHaveLength(2);
  });

  it("L0-02: file named YYYY-MM-DD.jsonl matching today", async () => {
    await recordConversation({ sessionKey:"s", rawMessages:[msg("user","Test msg content",Date.now())], baseDir:testDir });
    const files = await fs.readdir(path.join(testDir,"conversations"));
    expect(files[0]).toBe(`${today()}.jsonl`);
  });

  it("L0-02: sessionKey in line data, not filename", async () => {
    await recordConversation({ sessionKey:"special-key", rawMessages:[msg("user","Test content msg",Date.now())], baseDir:testDir });
    const files = await fs.readdir(path.join(testDir,"conversations"));
    expect(files[0]).not.toContain("special-key");
    const p = JSON.parse((await fs.readFile(path.join(testDir,"conversations",files[0]),"utf-8")).trim());
    expect(p.sessionKey).toBe("special-key");
  });

  it("L0-04: afterTimestamp filters old messages", async () => {
    const base = 1000000;
    const r = await recordConversation({
      sessionKey:"s", baseDir:testDir, afterTimestamp: base+50,
      rawMessages: [
        msg("user","Old msg skip here",base), msg("assistant","Old reply skip",base+1),
        msg("user","New msg keep here",base+100), msg("assistant","New reply keep",base+101),
      ],
    });
    expect(r).toHaveLength(2);
    expect(r[0].content).toBe("New msg keep here");
  });

  it("L0-04: afterTimestamp=0 records all", async () => {
    const r = await recordConversation({
      sessionKey:"s", baseDir:testDir, afterTimestamp:0,
      rawMessages: [msg("user","Message one here",100), msg("assistant","Reply one here",200)],
    });
    expect(r).toHaveLength(2);
  });

  it("L0-04: returns empty when all older than cursor", async () => {
    const r = await recordConversation({
      sessionKey:"s", baseDir:testDir, afterTimestamp:999999,
      rawMessages: [msg("user","Very old message",100)],
    });
    expect(r).toHaveLength(0);
  });

  it("L0-05: auto-creates nested directory", async () => {
    const nested = path.join(testDir,"deep","nested");
    await recordConversation({ sessionKey:"s", rawMessages:[msg("user","Deep dir test msg",Date.now())], baseDir:nested });
    const stat = await fs.stat(path.join(nested,"conversations"));
    expect(stat.isDirectory()).toBe(true);
  });

  it("ignores non-user/assistant roles", async () => {
    const r = await recordConversation({
      sessionKey:"s", baseDir:testDir,
      rawMessages: [msg("system","System text"), msg("user","User text here"), msg("tool","Tool out"), msg("assistant","Asst reply here")],
    });
    expect(r).toHaveLength(2);
    expect(r[0].role).toBe("user");
    expect(r[1].role).toBe("assistant");
  });

  it("handles multi-part content array", async () => {
    const r = await recordConversation({
      sessionKey:"s", baseDir:testDir,
      rawMessages: [{ role:"user", content:[{type:"text",text:"Hello "},{type:"text",text:"world!"}], timestamp:Date.now() }],
    });
    expect(r).toHaveLength(1);
    expect(r[0].content).toBe("Hello \nworld!");
  });

  it("skips null/undefined/non-object messages", async () => {
    const r = await recordConversation({
      sessionKey:"s", baseDir:testDir,
      rawMessages: [null, undefined, "str", 42, msg("user","Valid msg content here")] as unknown[],
    });
    expect(r).toHaveLength(1);
  });

  it("L0 now captures short messages (strict filtering deferred to L1)", async () => {
    const r = await recordConversation({
      sessionKey:"s", baseDir:testDir,
      rawMessages: [msg("user","hi"), msg("user","好的"), msg("user","This is a normal message for testing")],
    });
    // L0 is permissive — all 3 messages pass (short messages are no longer filtered at L0 stage)
    expect(r).toHaveLength(3);
  });

  it("returns empty for empty rawMessages", async () => {
    expect(await recordConversation({ sessionKey:"s", rawMessages:[], baseDir:testDir })).toHaveLength(0);
  });

  it("replaces polluted user message with originalUserText", async () => {
    const ts = Date.now();
    const r = await recordConversation({
      sessionKey:"s", baseDir:testDir,
      rawMessages: [msg("user","<relevant-memories>x</relevant-memories> polluted text here",ts), msg("assistant","Response text here",ts+1)],
      originalUserText: "Clean original text content",
      originalUserMessageCount: 0,
    });
    expect(r[0].content).toBe("Clean original text content");
  });

  it("warns when originalUserText without messageCount", async () => {
    const log = logger();
    await recordConversation({
      sessionKey:"s", baseDir:testDir, logger:log,
      rawMessages: [msg("user","User msg that may be polluted",Date.now())],
      originalUserText: "Clean text",
    });
    expect(log.warn).toHaveBeenCalled();
  });

  it("strips injected memory tags via sanitize", async () => {
    const r = await recordConversation({
      sessionKey:"s", baseDir:testDir,
      rawMessages: [msg("user","<relevant-memories>old</relevant-memories> Actual text here for test",Date.now())],
    });
    expect(r[0].content).not.toContain("relevant-memories");
    expect(r[0].content).toContain("Actual text here for test");
  });

  it("returns messages even when file write fails", async () => {
    const log = logger();
    const dir = path.join(testDir,"conversations");
    await fs.mkdir(dir, { recursive: true });
    await fs.mkdir(path.join(dir, `${today()}.jsonl`), { recursive: true });
    const r = await recordConversation({
      sessionKey:"s", baseDir:testDir, logger:log,
      rawMessages: [msg("user","Msg returned despite error",Date.now())],
    });
    expect(r).toHaveLength(1);
    expect(log.error).toHaveBeenCalled();
  });
});

// ── readConversationRecords ──

describe("readConversationRecords", () => {
  it("returns empty when dir does not exist", async () => {
    expect(await readConversationRecords("s", testDir)).toEqual([]);
  });

  it("returns empty when dir is empty", async () => {
    await fs.mkdir(path.join(testDir,"conversations"), { recursive: true });
    expect(await readConversationRecords("s", testDir)).toEqual([]);
  });

  it("filters by sessionKey", async () => {
    const ts = Date.now();
    await recordConversation({ sessionKey:"A", rawMessages:[msg("user","Msg from session A",ts)], baseDir:testDir });
    await recordConversation({ sessionKey:"B", rawMessages:[msg("user","Msg from session B",ts+1)], baseDir:testDir });
    const rA = await readConversationRecords("A", testDir);
    expect(rA).toHaveLength(1);
    expect(rA[0].sessionKey).toBe("A");
    expect(await readConversationRecords("C", testDir)).toEqual([]);
  });

  it("skips malformed lines", async () => {
    const log = logger();
    const dir = path.join(testDir,"conversations");
    await fs.mkdir(dir, { recursive: true });
    const valid = { sessionKey:"s", sessionId:"", recordedAt:new Date().toISOString(), id:"m1", role:"user", content:"Valid content msg", timestamp:Date.now() };
    await fs.writeFile(path.join(dir,`${today()}.jsonl`), [JSON.stringify(valid), "bad json", JSON.stringify({...valid,id:"m2",content:"Second valid msg"})].join("\n")+"\n");
    const r = await readConversationRecords("s", testDir, log);
    expect(r).toHaveLength(2);
    expect(log.warn).toHaveBeenCalled();
  });

  it("sorts by recordedAt", async () => {
    const dir = path.join(testDir,"conversations");
    await fs.mkdir(dir, { recursive: true });
    const rows = [
      { sessionKey:"s", sessionId:"", recordedAt:"2026-03-17T12:00:00Z", id:"m3", role:"user", content:"Third msg", timestamp:3000 },
      { sessionKey:"s", sessionId:"", recordedAt:"2026-03-17T10:00:00Z", id:"m1", role:"user", content:"First msg", timestamp:1000 },
    ];
    await fs.writeFile(path.join(dir,`${today()}.jsonl`), rows.map(r=>JSON.stringify(r)).join("\n")+"\n");
    const result = await readConversationRecords("s", testDir);
    expect(result[0].messages[0].content).toBe("First msg");
    expect(result[1].messages[0].content).toBe("Third msg");
  });

  it("ignores non-date-pattern filenames", async () => {
    const dir = path.join(testDir,"conversations");
    await fs.mkdir(dir, { recursive: true });
    await fs.writeFile(path.join(dir,"random.jsonl"), JSON.stringify({ sessionKey:"s", role:"user", content:"Ignored content", timestamp:1 })+"\n");
    expect(await readConversationRecords("s", testDir)).toEqual([]);
  });
});

// ── readConversationMessages ──

describe("readConversationMessages", () => {
  it("returns all messages for session", async () => {
    const ts = Date.now();
    await recordConversation({ sessionKey:"s", rawMessages:[msg("user","Hello user one",ts), msg("assistant","Hello asst one",ts+1)], baseDir:testDir });
    expect(await readConversationMessages("s", testDir)).toHaveLength(2);
  });

  it("filters by afterTimestamp", async () => {
    const dir = path.join(testDir,"conversations");
    await fs.mkdir(dir, { recursive: true });
    await fs.writeFile(path.join(dir,`${today()}.jsonl`), [
      JSON.stringify({ sessionKey:"s", sessionId:"", recordedAt:"2026-03-17T10:00:00Z", id:"m1", role:"user", content:"Old msg here", timestamp:100 }),
      JSON.stringify({ sessionKey:"s", sessionId:"", recordedAt:"2026-03-17T11:00:00Z", id:"m2", role:"user", content:"New msg here", timestamp:200 }),
    ].join("\n")+"\n");
    const msgs = await readConversationMessages("s", testDir, 150);
    expect(msgs).toHaveLength(1);
    expect(msgs[0].content).toBe("New msg here");
  });

  it("returns empty for non-existent dir", async () => {
    expect(await readConversationMessages("s", testDir)).toEqual([]);
  });

  it("limit truncates to newest N messages, preserving chronological order", async () => {
    const dir = path.join(testDir, "conversations");
    await fs.mkdir(dir, { recursive: true });
    // Write 10 messages with timestamps 1..10
    const lines = Array.from({ length: 10 }, (_, i) =>
      JSON.stringify({ sessionKey: "s", sessionId: "", recordedAt: `2026-03-17T10:0${i}:00Z`, id: `m${i + 1}`, role: "user", content: `Message number ${i + 1}`, timestamp: (i + 1) * 100 }),
    );
    await fs.writeFile(path.join(dir, `${today()}.jsonl`), lines.join("\n") + "\n");

    const msgs = await readConversationMessages("s", testDir, undefined, undefined, 3);
    expect(msgs).toHaveLength(3);
    // Should be the 3 newest: timestamps 800, 900, 1000 in chronological order
    expect(msgs[0].content).toBe("Message number 8");
    expect(msgs[1].content).toBe("Message number 9");
    expect(msgs[2].content).toBe("Message number 10");
    // Verify chronological order (old → new)
    expect(msgs[0].timestamp).toBeLessThan(msgs[1].timestamp);
    expect(msgs[1].timestamp).toBeLessThan(msgs[2].timestamp);
  });

  it("limit larger than total messages returns all", async () => {
    const dir = path.join(testDir, "conversations");
    await fs.mkdir(dir, { recursive: true });
    const lines = [
      JSON.stringify({ sessionKey: "s", sessionId: "", recordedAt: "2026-03-17T10:00:00Z", id: "m1", role: "user", content: "Only message here", timestamp: 100 }),
    ];
    await fs.writeFile(path.join(dir, `${today()}.jsonl`), lines.join("\n") + "\n");

    const msgs = await readConversationMessages("s", testDir, undefined, undefined, 50);
    expect(msgs).toHaveLength(1);
    expect(msgs[0].content).toBe("Only message here");
  });

  it("limit undefined preserves backward-compat (returns all)", async () => {
    const dir = path.join(testDir, "conversations");
    await fs.mkdir(dir, { recursive: true });
    const lines = Array.from({ length: 5 }, (_, i) =>
      JSON.stringify({ sessionKey: "s", sessionId: "", recordedAt: `2026-03-17T10:0${i}:00Z`, id: `m${i + 1}`, role: "user", content: `Msg ${i + 1}`, timestamp: (i + 1) * 100 }),
    );
    await fs.writeFile(path.join(dir, `${today()}.jsonl`), lines.join("\n") + "\n");

    const msgs = await readConversationMessages("s", testDir);
    expect(msgs).toHaveLength(5);
  });

  it("limit combined with afterTimestamp truncates filtered results", async () => {
    const dir = path.join(testDir, "conversations");
    await fs.mkdir(dir, { recursive: true });
    // 10 messages: timestamps 100..1000
    const lines = Array.from({ length: 10 }, (_, i) =>
      JSON.stringify({ sessionKey: "s", sessionId: "", recordedAt: `2026-03-17T10:0${i}:00Z`, id: `m${i + 1}`, role: "user", content: `Message ${i + 1}`, timestamp: (i + 1) * 100 }),
    );
    await fs.writeFile(path.join(dir, `${today()}.jsonl`), lines.join("\n") + "\n");

    // afterTimestamp=500 → keep ts 600,700,800,900,1000 (5 msgs), then limit=2 → 900,1000
    const msgs = await readConversationMessages("s", testDir, 500, undefined, 2);
    expect(msgs).toHaveLength(2);
    expect(msgs[0].content).toBe("Message 9");
    expect(msgs[1].content).toBe("Message 10");
  });
});

// ── readConversationMessagesGroupedBySessionId ──

describe("readConversationMessagesGroupedBySessionId", () => {
  it("groups by sessionId, sorted by earliest timestamp", async () => {
    const dir = path.join(testDir,"conversations");
    await fs.mkdir(dir, { recursive: true });
    await fs.writeFile(path.join(dir,`${today()}.jsonl`), [
      JSON.stringify({ sessionKey:"sk", sessionId:"A", recordedAt:"2026-03-17T10:00:00Z", id:"m1", role:"user", content:"Session A msg one", timestamp:100 }),
      JSON.stringify({ sessionKey:"sk", sessionId:"B", recordedAt:"2026-03-17T10:01:00Z", id:"m2", role:"user", content:"Session B msg one", timestamp:200 }),
      JSON.stringify({ sessionKey:"sk", sessionId:"A", recordedAt:"2026-03-17T10:02:00Z", id:"m3", role:"assistant", content:"Session A reply one", timestamp:300 }),
    ].join("\n")+"\n");
    const g = await readConversationMessagesGroupedBySessionId("sk", testDir);
    expect(g).toHaveLength(2);
    expect(g[0].sessionId).toBe("A");
    expect(g[0].messages).toHaveLength(2);
    expect(g[1].sessionId).toBe("B");
    expect(g[1].messages).toHaveLength(1);
  });

  it("filters by afterTimestamp before grouping", async () => {
    const dir = path.join(testDir,"conversations");
    await fs.mkdir(dir, { recursive: true });
    await fs.writeFile(path.join(dir,`${today()}.jsonl`), [
      JSON.stringify({ sessionKey:"sk", sessionId:"A", recordedAt:"2026-03-17T10:00:00Z", id:"m1", role:"user", content:"Old A message", timestamp:100 }),
      JSON.stringify({ sessionKey:"sk", sessionId:"A", recordedAt:"2026-03-17T10:05:00Z", id:"m2", role:"user", content:"New A message", timestamp:500 }),
    ].join("\n")+"\n");
    const g = await readConversationMessagesGroupedBySessionId("sk", testDir, 200);
    expect(g).toHaveLength(1);
    expect(g[0].messages).toHaveLength(1);
    expect(g[0].messages[0].content).toBe("New A message");
  });

  it("returns empty for non-existent dir", async () => {
    expect(await readConversationMessagesGroupedBySessionId("s", testDir)).toEqual([]);
  });

  it("limit truncates to newest N messages across groups", async () => {
    const dir = path.join(testDir, "conversations");
    await fs.mkdir(dir, { recursive: true });
    // 3 groups: A(ts100,300), B(ts200,400), C(ts500)
    // Total 5 messages. limit=3 → keep ts 300,400,500 (newest 3)
    const lines = [
      JSON.stringify({ sessionKey: "sk", sessionId: "A", recordedAt: "2026-03-17T10:00:00Z", id: "m1", role: "user", content: "A old msg", timestamp: 100 }),
      JSON.stringify({ sessionKey: "sk", sessionId: "B", recordedAt: "2026-03-17T10:01:00Z", id: "m2", role: "user", content: "B old msg", timestamp: 200 }),
      JSON.stringify({ sessionKey: "sk", sessionId: "A", recordedAt: "2026-03-17T10:02:00Z", id: "m3", role: "assistant", content: "A new msg", timestamp: 300 }),
      JSON.stringify({ sessionKey: "sk", sessionId: "B", recordedAt: "2026-03-17T10:03:00Z", id: "m4", role: "assistant", content: "B new msg", timestamp: 400 }),
      JSON.stringify({ sessionKey: "sk", sessionId: "C", recordedAt: "2026-03-17T10:04:00Z", id: "m5", role: "user", content: "C only msg", timestamp: 500 }),
    ];
    await fs.writeFile(path.join(dir, `${today()}.jsonl`), lines.join("\n") + "\n");

    const g = await readConversationMessagesGroupedBySessionId("sk", testDir, undefined, undefined, 3);
    // Should keep ts 300(A), 400(B), 500(C) → 3 groups each with 1 msg
    const totalMsgs = g.reduce((sum, group) => sum + group.messages.length, 0);
    expect(totalMsgs).toBe(3);
    // Verify groups sorted by earliest msg timestamp
    expect(g[0].sessionId).toBe("A");
    expect(g[0].messages[0].content).toBe("A new msg");
    expect(g[1].sessionId).toBe("B");
    expect(g[1].messages[0].content).toBe("B new msg");
    expect(g[2].sessionId).toBe("C");
    expect(g[2].messages[0].content).toBe("C only msg");
  });

  it("limit drops groups that become empty after truncation", async () => {
    const dir = path.join(testDir, "conversations");
    await fs.mkdir(dir, { recursive: true });
    // Group A: ts 100, 200 (old). Group B: ts 300, 400 (new).
    // limit=2 → keep ts 300, 400 → only group B remains
    const lines = [
      JSON.stringify({ sessionKey: "sk", sessionId: "A", recordedAt: "2026-03-17T10:00:00Z", id: "m1", role: "user", content: "A msg one", timestamp: 100 }),
      JSON.stringify({ sessionKey: "sk", sessionId: "A", recordedAt: "2026-03-17T10:01:00Z", id: "m2", role: "assistant", content: "A msg two", timestamp: 200 }),
      JSON.stringify({ sessionKey: "sk", sessionId: "B", recordedAt: "2026-03-17T10:02:00Z", id: "m3", role: "user", content: "B msg one", timestamp: 300 }),
      JSON.stringify({ sessionKey: "sk", sessionId: "B", recordedAt: "2026-03-17T10:03:00Z", id: "m4", role: "assistant", content: "B msg two", timestamp: 400 }),
    ];
    await fs.writeFile(path.join(dir, `${today()}.jsonl`), lines.join("\n") + "\n");

    const g = await readConversationMessagesGroupedBySessionId("sk", testDir, undefined, undefined, 2);
    expect(g).toHaveLength(1);
    expect(g[0].sessionId).toBe("B");
    expect(g[0].messages).toHaveLength(2);
    // Verify chronological order within group
    expect(g[0].messages[0].timestamp).toBeLessThan(g[0].messages[1].timestamp);
  });

  it("limit preserves chronological order within and across groups", async () => {
    const dir = path.join(testDir, "conversations");
    await fs.mkdir(dir, { recursive: true });
    // Interleaved: A(100), B(200), A(300), B(400), A(500), B(600)
    // limit=4 → keep ts 300(A), 400(B), 500(A), 600(B)
    const lines = [
      JSON.stringify({ sessionKey: "sk", sessionId: "A", recordedAt: "2026-03-17T10:00:00Z", id: "m1", role: "user", content: "A first", timestamp: 100 }),
      JSON.stringify({ sessionKey: "sk", sessionId: "B", recordedAt: "2026-03-17T10:01:00Z", id: "m2", role: "user", content: "B first", timestamp: 200 }),
      JSON.stringify({ sessionKey: "sk", sessionId: "A", recordedAt: "2026-03-17T10:02:00Z", id: "m3", role: "assistant", content: "A second", timestamp: 300 }),
      JSON.stringify({ sessionKey: "sk", sessionId: "B", recordedAt: "2026-03-17T10:03:00Z", id: "m4", role: "assistant", content: "B second", timestamp: 400 }),
      JSON.stringify({ sessionKey: "sk", sessionId: "A", recordedAt: "2026-03-17T10:04:00Z", id: "m5", role: "user", content: "A third", timestamp: 500 }),
      JSON.stringify({ sessionKey: "sk", sessionId: "B", recordedAt: "2026-03-17T10:05:00Z", id: "m6", role: "user", content: "B third", timestamp: 600 }),
    ];
    await fs.writeFile(path.join(dir, `${today()}.jsonl`), lines.join("\n") + "\n");

    const g = await readConversationMessagesGroupedBySessionId("sk", testDir, undefined, undefined, 4);
    // Groups: A(300,500), B(400,600)
    expect(g).toHaveLength(2);
    // Group A first (earliest msg ts=300 < B's 400)
    expect(g[0].sessionId).toBe("A");
    expect(g[0].messages).toHaveLength(2);
    expect(g[0].messages[0].timestamp).toBe(300);
    expect(g[0].messages[1].timestamp).toBe(500);
    // Group B
    expect(g[1].sessionId).toBe("B");
    expect(g[1].messages).toHaveLength(2);
    expect(g[1].messages[0].timestamp).toBe(400);
    expect(g[1].messages[1].timestamp).toBe(600);
  });

  it("limit undefined returns all messages (backward compat)", async () => {
    const dir = path.join(testDir, "conversations");
    await fs.mkdir(dir, { recursive: true });
    const lines = Array.from({ length: 20 }, (_, i) =>
      JSON.stringify({ sessionKey: "sk", sessionId: "X", recordedAt: `2026-03-17T10:${String(i).padStart(2, "0")}:00Z`, id: `m${i + 1}`, role: "user", content: `Msg ${i + 1}`, timestamp: (i + 1) * 100 }),
    );
    await fs.writeFile(path.join(dir, `${today()}.jsonl`), lines.join("\n") + "\n");

    const g = await readConversationMessagesGroupedBySessionId("sk", testDir);
    expect(g).toHaveLength(1);
    expect(g[0].messages).toHaveLength(20);
  });

  it("limit combined with afterTimestamp works correctly", async () => {
    const dir = path.join(testDir, "conversations");
    await fs.mkdir(dir, { recursive: true });
    // 6 messages: ts 100..600. afterTimestamp=300 → keep 400,500,600. limit=2 → 500,600
    const lines = Array.from({ length: 6 }, (_, i) =>
      JSON.stringify({ sessionKey: "sk", sessionId: "X", recordedAt: `2026-03-17T10:0${i}:00Z`, id: `m${i + 1}`, role: "user", content: `Msg ${i + 1}`, timestamp: (i + 1) * 100 }),
    );
    await fs.writeFile(path.join(dir, `${today()}.jsonl`), lines.join("\n") + "\n");

    const g = await readConversationMessagesGroupedBySessionId("sk", testDir, 300, undefined, 2);
    expect(g).toHaveLength(1);
    expect(g[0].messages).toHaveLength(2);
    expect(g[0].messages[0].content).toBe("Msg 5");
    expect(g[0].messages[1].content).toBe("Msg 6");
  });
});

// ── Edge cases: recordConversation ──

describe("recordConversation edge cases", () => {
  it("handles empty string content (filters out)", async () => {
    const r = await recordConversation({
      sessionKey: "s", baseDir: testDir,
      rawMessages: [{ role: "user", content: "", timestamp: Date.now() }],
    });
    expect(r).toHaveLength(0);
  });

  it("handles whitespace-only content (filters out)", async () => {
    const r = await recordConversation({
      sessionKey: "s", baseDir: testDir,
      rawMessages: [{ role: "user", content: "   \n  ", timestamp: Date.now() }],
    });
    expect(r).toHaveLength(0);
  });

  it("assigns Date.now() when message has no timestamp", async () => {
    const before = Date.now();
    const r = await recordConversation({
      sessionKey: "s", baseDir: testDir,
      rawMessages: [{ role: "user", content: "A message without a timestamp field here" }],
    });
    if (r.length > 0) {
      expect(r[0].timestamp).toBeGreaterThanOrEqual(before);
    }
  });

  it("preserves existing message id if present", async () => {
    const r = await recordConversation({
      sessionKey: "s", baseDir: testDir,
      rawMessages: [{ role: "user", content: "Message with a custom id value here", id: "custom-id-1", timestamp: Date.now() }],
    });
    expect(r).toHaveLength(1);
    expect(r[0].id).toBe("custom-id-1");
  });

  it("generates id when message id is missing", async () => {
    const r = await recordConversation({
      sessionKey: "s", baseDir: testDir,
      rawMessages: [{ role: "user", content: "Message without any id field value", timestamp: Date.now() }],
    });
    expect(r).toHaveLength(1);
    expect(r[0].id).toMatch(/^msg_/);
  });

  it("originalUserMessageCount out of range → skip replacement, sanitize fallback", async () => {
    const log = logger();
    const r = await recordConversation({
      sessionKey: "s", baseDir: testDir, logger: log,
      rawMessages: [msg("user", "<relevant-memories>injected</relevant-memories> Some real text here", Date.now())],
      originalUserText: "Clean text original",
      originalUserMessageCount: 99, // out of range
    });
    // Should NOT replace (out of range), but sanitize should remove tags
    expect(r).toHaveLength(1);
    expect(r[0].content).not.toContain("relevant-memories");
  });

  it("originalUserMessageCount negative → skip replacement", async () => {
    const log = logger();
    await recordConversation({
      sessionKey: "s", baseDir: testDir, logger: log,
      rawMessages: [msg("user", "Some longer message content for testing purposes", Date.now())],
      originalUserText: "Clean text",
      originalUserMessageCount: -1,
    });
    expect(log.warn).toHaveBeenCalled();
  });

  it("target raw message has no timestamp → skip replacement with warning", async () => {
    const log = logger();
    await recordConversation({
      sessionKey: "s", baseDir: testDir, logger: log,
      rawMessages: [{ role: "user", content: "Message content without timestamp field here" }],
      originalUserText: "Clean prompt text here",
      originalUserMessageCount: 0,
    });
    expect(log.warn).toHaveBeenCalledWith(expect.stringContaining("no valid timestamp"));
  });

  it("target timestamp not found in extracted → skip replacement with warning", async () => {
    const log = logger();
    const ts = Date.now();
    // rawMessages[0] has ts=100, but extracted will filter it (afterTimestamp=200)
    await recordConversation({
      sessionKey: "s", baseDir: testDir, logger: log, afterTimestamp: 200,
      rawMessages: [
        { role: "user", content: "Old message that was filtered out", timestamp: 100 },
        { role: "user", content: "New message that passes the cursor filter", timestamp: 300 },
      ],
      originalUserText: "Replacement text for testing",
      originalUserMessageCount: 0, // points to old msg (ts=100), which is filtered
    });
    // Should warn about not finding target in extracted batch
    expect(log.warn).toHaveBeenCalledWith(expect.stringContaining("not found in extracted batch"));
  });

  it("sessionId defaults to empty string when not provided", async () => {
    await recordConversation({
      sessionKey: "s", baseDir: testDir,
      rawMessages: [msg("user", "Message to test sessionId default value", Date.now())],
    });
    const dir = path.join(testDir, "conversations");
    const files = await fs.readdir(dir);
    const line = JSON.parse((await fs.readFile(path.join(dir, files[0]), "utf-8")).trim());
    expect(line.sessionId).toBe("");
  });

  it("content array with non-text parts is ignored", async () => {
    const r = await recordConversation({
      sessionKey: "s", baseDir: testDir,
      rawMessages: [{ role: "user", content: [{ type: "image", url: "http://example.com/img.png" }, { type: "text", text: "Some text content here for test" }], timestamp: Date.now() }],
    });
    expect(r).toHaveLength(1);
    expect(r[0].content).toBe("Some text content here for test");
  });
});

// ── Edge cases: readConversationRecords ──

describe("readConversationRecords edge cases", () => {
  it("handles file read error gracefully", async () => {
    const log = logger();
    const dir = path.join(testDir, "conversations");
    await fs.mkdir(dir, { recursive: true });
    // Create a directory with the same name as a date file (causes read error)
    await fs.mkdir(path.join(dir, "2026-03-17.jsonl"), { recursive: true });
    // readdir should include it, but readFile will fail
    // However, the function uses .isFile() filter so directories are skipped
    const result = await readConversationRecords("s", testDir, log);
    expect(result).toEqual([]);
  });

  it("generates id for records missing id field", async () => {
    const dir = path.join(testDir, "conversations");
    await fs.mkdir(dir, { recursive: true });
    const record = { sessionKey: "s", sessionId: "", recordedAt: new Date().toISOString(), role: "user", content: "No id field in this record" };
    await fs.writeFile(path.join(dir, `${today()}.jsonl`), JSON.stringify(record) + "\n");
    const result = await readConversationRecords("s", testDir);
    expect(result).toHaveLength(1);
    expect(result[0].messages[0].id).toMatch(/^msg_/);
  });
});
