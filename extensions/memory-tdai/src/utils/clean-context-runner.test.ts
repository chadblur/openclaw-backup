import { describe, it, expect } from "vitest";
import {
  parseModelRef,
  resolveModelFromMainConfig,
  CleanContextRunner,
} from "./clean-context-runner.js";

// ── parseModelRef ──

describe("parseModelRef", () => {
  it("should parse standard provider/model", () => {
    expect(parseModelRef("azure/gpt-5.2-chat")).toEqual({
      provider: "azure",
      model: "gpt-5.2-chat",
    });
  });

  it("should parse provider with nested model path", () => {
    expect(parseModelRef("custom-host/org/model-v2")).toEqual({
      provider: "custom-host",
      model: "org/model-v2",
    });
  });

  it("should return undefined for undefined input", () => {
    expect(parseModelRef(undefined)).toBeUndefined();
  });

  it("should return undefined for empty string", () => {
    expect(parseModelRef("")).toBeUndefined();
  });

  it("should return undefined for whitespace-only", () => {
    expect(parseModelRef("   ")).toBeUndefined();
  });

  it("should return undefined for bare model name (no slash)", () => {
    expect(parseModelRef("bare-model-name")).toBeUndefined();
  });

  it("should return undefined for leading slash", () => {
    // slashIdx === 0
    expect(parseModelRef("/model")).toBeUndefined();
  });

  it("should return undefined for trailing slash", () => {
    // slashIdx === length - 1
    expect(parseModelRef("provider/")).toBeUndefined();
  });

  it("should trim whitespace before parsing", () => {
    expect(parseModelRef("  openai/gpt-4  ")).toEqual({
      provider: "openai",
      model: "gpt-4",
    });
  });
});

// ── resolveModelFromMainConfig ──

describe("resolveModelFromMainConfig", () => {
  it("should return undefined for null/undefined config", () => {
    expect(resolveModelFromMainConfig(null)).toBeUndefined();
    expect(resolveModelFromMainConfig(undefined)).toBeUndefined();
  });

  it("should return undefined for non-object config", () => {
    expect(resolveModelFromMainConfig("string")).toBeUndefined();
    expect(resolveModelFromMainConfig(42)).toBeUndefined();
  });

  it("should return undefined when agents is missing", () => {
    expect(resolveModelFromMainConfig({})).toBeUndefined();
  });

  it("should return undefined when agents.defaults is missing", () => {
    expect(resolveModelFromMainConfig({ agents: {} })).toBeUndefined();
  });

  it("should return undefined when agents.defaults.model is missing", () => {
    expect(resolveModelFromMainConfig({ agents: { defaults: {} } })).toBeUndefined();
  });

  it("should parse direct provider/model string", () => {
    const config = {
      agents: { defaults: { model: "openai/gpt-4" } },
    };
    expect(resolveModelFromMainConfig(config)).toEqual({
      provider: "openai",
      model: "gpt-4",
    });
  });

  it("should parse model from { primary } object form", () => {
    const config = {
      agents: { defaults: { model: { primary: "azure/gpt-5.2-chat" } } },
    };
    expect(resolveModelFromMainConfig(config)).toEqual({
      provider: "azure",
      model: "gpt-5.2-chat",
    });
  });

  it("should return undefined when primary is missing in object form", () => {
    const config = {
      agents: { defaults: { model: { secondary: "something" } } },
    };
    expect(resolveModelFromMainConfig(config)).toBeUndefined();
  });

  it("should return undefined for bare model name without alias table", () => {
    // "my-model" has no "/" and no models alias table → undefined
    const config = {
      agents: { defaults: { model: "my-model" } },
    };
    expect(resolveModelFromMainConfig(config)).toBeUndefined();
  });

  it("should resolve alias from agents.defaults.models table", () => {
    const config = {
      agents: {
        defaults: {
          model: "fast",
          models: {
            "openai/gpt-4-turbo": { alias: "fast" },
            "anthropic/claude-opus-4-6": { alias: "smart" },
          },
        },
      },
    };
    expect(resolveModelFromMainConfig(config)).toEqual({
      provider: "openai",
      model: "gpt-4-turbo",
    });
  });

  it("should do case-insensitive alias matching", () => {
    const config = {
      agents: {
        defaults: {
          model: "FAST",
          models: {
            "openai/gpt-4-turbo": { alias: "fast" },
          },
        },
      },
    };
    expect(resolveModelFromMainConfig(config)).toEqual({
      provider: "openai",
      model: "gpt-4-turbo",
    });
  });

  it("should return undefined when alias is not found", () => {
    const config = {
      agents: {
        defaults: {
          model: "nonexistent",
          models: {
            "openai/gpt-4-turbo": { alias: "fast" },
          },
        },
      },
    };
    expect(resolveModelFromMainConfig(config)).toBeUndefined();
  });

  it("should skip models entries without string alias", () => {
    const config = {
      agents: {
        defaults: {
          model: "test",
          models: {
            "openai/gpt-4": { alias: 123 },
            "anthropic/claude-opus-4-6": null,
            "azure/gpt-5": "not-an-object",
          },
        },
      },
    };
    expect(resolveModelFromMainConfig(config)).toBeUndefined();
  });

  it("should return undefined for empty model string", () => {
    const config = {
      agents: { defaults: { model: "" } },
    };
    expect(resolveModelFromMainConfig(config)).toBeUndefined();
  });

  it("should return undefined for whitespace-only model string", () => {
    const config = {
      agents: { defaults: { model: "   " } },
    };
    expect(resolveModelFromMainConfig(config)).toBeUndefined();
  });
});

// ── CleanContextRunner constructor — model resolution priority ──

describe("CleanContextRunner constructor — model resolution priority", () => {
  const baseConfig = { agents: { defaults: { model: "anthropic/claude-opus-4-6" } } };

  it("priority 1: modelRef takes precedence over everything", () => {
    const runner = new CleanContextRunner({
      config: baseConfig,
      modelRef: "azure/gpt-5",
      provider: "ignored-provider",
      model: "ignored-model",
    });
    // Access private fields via type casting for testing
    const r = runner as unknown as Record<string, unknown>;
    expect(r.resolvedProvider).toBe("azure");
    expect(r.resolvedModel).toBe("gpt-5");
  });

  it("priority 2: explicit provider + model when no modelRef", () => {
    const runner = new CleanContextRunner({
      config: baseConfig,
      provider: "openai",
      model: "gpt-4",
    });
    const r = runner as unknown as Record<string, unknown>;
    expect(r.resolvedProvider).toBe("openai");
    expect(r.resolvedModel).toBe("gpt-4");
  });

  it("priority 2b: explicit provider alone (model undefined)", () => {
    const runner = new CleanContextRunner({
      config: baseConfig,
      provider: "openai",
    });
    const r = runner as unknown as Record<string, unknown>;
    expect(r.resolvedProvider).toBe("openai");
    expect(r.resolvedModel).toBeUndefined();
  });

  it("priority 3: falls back to main config agents.defaults.model", () => {
    const runner = new CleanContextRunner({
      config: { agents: { defaults: { model: "deepseek/chat-v3" } } },
    });
    const r = runner as unknown as Record<string, unknown>;
    expect(r.resolvedProvider).toBe("deepseek");
    expect(r.resolvedModel).toBe("chat-v3");
  });

  it("priority 4: all undefined when config has no model", () => {
    const runner = new CleanContextRunner({ config: {} });
    const r = runner as unknown as Record<string, unknown>;
    expect(r.resolvedProvider).toBeUndefined();
    expect(r.resolvedModel).toBeUndefined();
  });

  it("ignores invalid modelRef and falls back to explicit fields", () => {
    const runner = new CleanContextRunner({
      config: baseConfig,
      modelRef: "no-slash",
      provider: "openai",
      model: "gpt-4",
    });
    const r = runner as unknown as Record<string, unknown>;
    expect(r.resolvedProvider).toBe("openai");
    expect(r.resolvedModel).toBe("gpt-4");
  });

  it("ignores invalid modelRef and no explicit fields → config fallback", () => {
    const runner = new CleanContextRunner({
      config: { agents: { defaults: { model: "openai/o1" } } },
      modelRef: "bare-name",
    });
    const r = runner as unknown as Record<string, unknown>;
    expect(r.resolvedProvider).toBe("openai");
    expect(r.resolvedModel).toBe("o1");
  });
});
