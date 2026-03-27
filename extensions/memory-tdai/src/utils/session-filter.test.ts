import { describe, it, expect } from "vitest";
import { SessionFilter } from "./session-filter.js";
import type { AgentHookContext } from "./session-filter.js";

// ────────────────────────────────────────
// Built-in skip rules
// ────────────────────────────────────────
describe("built-in skip rules", () => {
  const filter = new SessionFilter();

  it("should skip scene extraction runner sessions", () => {
    expect(filter.shouldSkip("agent:abc:memory-scene-extract-xyz")).toBe(true);
  });

  it("should skip subagent sessions", () => {
    expect(filter.shouldSkip("agent:main:subagent:task1")).toBe(true);
  });

  it("should skip temp: sessions", () => {
    expect(filter.shouldSkip("temp:slug-generator")).toBe(true);
  });

  it("should NOT skip normal user sessions", () => {
    expect(filter.shouldSkip("agent:user123:telegram")).toBe(false);
  });
});

// ────────────────────────────────────────
// User-configured excludeAgents (glob patterns)
// ────────────────────────────────────────
describe("user-configured excludeAgents", () => {
  it("should skip sessions matching a simple glob pattern", () => {
    const filter = new SessionFilter(["bench-judge-*"]);
    expect(filter.shouldSkip("agent:bench-judge-001:session")).toBe(true);
    expect(filter.shouldSkip("agent:normal-user:session")).toBe(false);
  });

  it("should match substring (glob without wildcard matches anywhere in key)", () => {
    const filter = new SessionFilter(["test-bot"]);
    expect(filter.shouldSkip("agent:test-bot:session")).toBe(true);
    // Note: no anchoring — "test-bot" substring appears in "test-bot-2" too
    expect(filter.shouldSkip("agent:test-bot-2:session")).toBe(true);
    // But a completely different key should not match
    expect(filter.shouldSkip("agent:other-user:session")).toBe(false);
  });

  it("should handle multiple exclude patterns", () => {
    const filter = new SessionFilter(["bench-*", "ci-runner-*"]);
    expect(filter.shouldSkip("agent:bench-eval:s1")).toBe(true);
    expect(filter.shouldSkip("agent:ci-runner-42:s2")).toBe(true);
    expect(filter.shouldSkip("agent:real-user:s3")).toBe(false);
  });

  it("should ignore empty/whitespace patterns", () => {
    const filter = new SessionFilter(["  ", "", "valid-*"]);
    expect(filter.shouldSkip("agent:valid-match:s")).toBe(true);
    // Empty patterns should not cause everything to match
    expect(filter.shouldSkip("agent:normal:s")).toBe(false);
  });

  it("should escape regex special characters in patterns", () => {
    const filter = new SessionFilter(["agent.special+id"]);
    // The dot and plus should be literal, not regex wildcards
    expect(filter.shouldSkip("agent.special+id")).toBe(true);
    expect(filter.shouldSkip("agentXspecialXid")).toBe(false);
  });
});

// ────────────────────────────────────────
// shouldSkipCtx
// ────────────────────────────────────────
describe("shouldSkipCtx", () => {
  const filter = new SessionFilter();

  it("should skip when sessionKey is missing", () => {
    const ctx: AgentHookContext = {};
    expect(filter.shouldSkipCtx(ctx)).toBe(true);
  });

  it("should skip when sessionId starts with 'memory-'", () => {
    const ctx: AgentHookContext = {
      sessionKey: "agent:user:telegram",
      sessionId: "memory-scene-extract",
    };
    expect(filter.shouldSkipCtx(ctx)).toBe(true);
  });

  it("should NOT skip normal context", () => {
    const ctx: AgentHookContext = {
      sessionKey: "agent:user:telegram",
      sessionId: "normal-session-123",
    };
    expect(filter.shouldSkipCtx(ctx)).toBe(false);
  });

  it("should apply built-in rules via shouldSkip delegation", () => {
    const ctx: AgentHookContext = {
      sessionKey: "temp:some-task",
      sessionId: "normal-id",
    };
    expect(filter.shouldSkipCtx(ctx)).toBe(true);
  });

  it("should apply user exclude rules via shouldSkip delegation", () => {
    const f = new SessionFilter(["excluded-bot-*"]);
    const ctx: AgentHookContext = {
      sessionKey: "agent:excluded-bot-1:session",
      sessionId: "sid-1",
    };
    expect(f.shouldSkipCtx(ctx)).toBe(true);
  });
});
