/**
 * Unit tests for parseConfig (config.ts).
 *
 * Covers:
 * - CFG-01: Zero-config defaults (provider="none", embedding disabled)
 * - CFG-02: Explicit "none" provider disables embedding
 * - CFG-03: "local" provider is blocked at entry level (treated as "none")
 * - CFG-04: Remote provider with complete config enables embedding
 * - CFG-05: Remote provider with missing fields disables embedding with configError
 * - CFG-06: Dimensions default to 768 for "none" provider (placeholder)
 * - CFG-07: Explicit enabled=false overrides remote provider
 */

import { describe, it, expect } from "vitest";
import { parseConfig } from "./config.js";

describe("parseConfig — embedding provider entry-level gating", () => {
  // CFG-01: Zero config → provider defaults to "none", embedding disabled
  describe("CFG-01: zero-config defaults", () => {
    it("should default provider to 'none' and disable embedding", () => {
      const cfg = parseConfig({});
      expect(cfg.embedding.provider).toBe("none");
      expect(cfg.embedding.enabled).toBe(false);
      expect(cfg.embedding.dimensions).toBe(0); // no placeholder — vec0 tables deferred
    });

    it("should work with undefined input", () => {
      const cfg = parseConfig(undefined);
      expect(cfg.embedding.provider).toBe("none");
      expect(cfg.embedding.enabled).toBe(false);
    });
  });

  // CFG-02: Explicit "none"
  describe("CFG-02: explicit 'none' provider", () => {
    it("should disable embedding when provider is explicitly 'none'", () => {
      const cfg = parseConfig({
        embedding: { provider: "none" },
      });
      expect(cfg.embedding.provider).toBe("none");
      expect(cfg.embedding.enabled).toBe(false);
      expect(cfg.embedding.configError).toBeUndefined();
    });
  });

  // CFG-03: "local" provider is blocked
  describe("CFG-03: 'local' provider is blocked at entry level", () => {
    it("should treat 'local' as 'none' and disable embedding", () => {
      const cfg = parseConfig({
        embedding: { provider: "local" },
      });
      expect(cfg.embedding.provider).toBe("none");
      expect(cfg.embedding.enabled).toBe(false);
    });

    it("should set configError explaining local is not available", () => {
      const cfg = parseConfig({
        embedding: { provider: "local" },
      });
      expect(cfg.embedding.configError).toBeDefined();
      expect(cfg.embedding.configError).toContain("not available");
    });

    it("should use 0 dimensions (deferred) for blocked local", () => {
      const cfg = parseConfig({
        embedding: { provider: "local" },
      });
      expect(cfg.embedding.dimensions).toBe(0);
    });
  });

  // CFG-04: Remote provider with complete config
  describe("CFG-04: remote provider with complete config", () => {
    it("should enable embedding for openai with all required fields", () => {
      const cfg = parseConfig({
        embedding: {
          provider: "openai",
          apiKey: "sk-test-key",
          baseUrl: "https://api.openai.com/v1",
          model: "text-embedding-3-small",
          dimensions: 1536,
        },
      });
      expect(cfg.embedding.provider).toBe("openai");
      expect(cfg.embedding.enabled).toBe(true);
      expect(cfg.embedding.dimensions).toBe(1536);
      expect(cfg.embedding.configError).toBeUndefined();
    });

    it("should enable embedding for deepseek with all required fields", () => {
      const cfg = parseConfig({
        embedding: {
          provider: "deepseek",
          apiKey: "sk-ds-key",
          baseUrl: "https://api.deepseek.com/v1",
          model: "deepseek-embedding",
          dimensions: 1024,
        },
      });
      expect(cfg.embedding.provider).toBe("deepseek");
      expect(cfg.embedding.enabled).toBe(true);
      expect(cfg.embedding.dimensions).toBe(1024);
    });
  });

  // CFG-05: Remote provider with missing fields
  describe("CFG-05: remote provider with missing fields", () => {
    it("should disable embedding and set configError when apiKey is missing", () => {
      const cfg = parseConfig({
        embedding: {
          provider: "openai",
          baseUrl: "https://api.openai.com/v1",
          model: "text-embedding-3-small",
          dimensions: 1536,
        },
      });
      expect(cfg.embedding.enabled).toBe(false);
      expect(cfg.embedding.configError).toBeDefined();
      expect(cfg.embedding.configError).toContain("apiKey");
    });

    it("should list all missing fields in configError", () => {
      const cfg = parseConfig({
        embedding: {
          provider: "openai",
          // Missing: apiKey, baseUrl, model, dimensions
        },
      });
      expect(cfg.embedding.enabled).toBe(false);
      expect(cfg.embedding.configError).toContain("apiKey");
      expect(cfg.embedding.configError).toContain("baseUrl");
      expect(cfg.embedding.configError).toContain("model");
      expect(cfg.embedding.configError).toContain("dimensions");
    });
  });

  // CFG-06: Dimensions placeholder
  describe("CFG-06: dimensions defaults", () => {
    it("should use 0 as deferred dimension for 'none' provider", () => {
      const cfg = parseConfig({});
      expect(cfg.embedding.dimensions).toBe(0);
    });

    it("should use user-specified dimensions for remote provider", () => {
      const cfg = parseConfig({
        embedding: {
          provider: "openai",
          apiKey: "sk-key",
          baseUrl: "https://api.openai.com/v1",
          model: "text-embedding-3-small",
          dimensions: 3072,
        },
      });
      expect(cfg.embedding.dimensions).toBe(3072);
    });
  });

  // CFG-07: Explicit enabled=false
  describe("CFG-07: explicit enabled=false overrides", () => {
    it("should respect enabled=false even with complete remote config", () => {
      const cfg = parseConfig({
        embedding: {
          enabled: false,
          provider: "openai",
          apiKey: "sk-key",
          baseUrl: "https://api.openai.com/v1",
          model: "text-embedding-3-small",
          dimensions: 1536,
        },
      });
      // enabled=false is read from config, but then overridden by provider validation.
      // For remote provider with complete config, enabled stays as the user set it.
      // Actually, the current logic: embeddingEnabled starts as bool(enabled) ?? true,
      // then only gets set to false for "none"/"local" providers or missing remote fields.
      // For complete remote config, it respects the user's enabled=false.
      expect(cfg.embedding.enabled).toBe(false);
    });
  });
});

describe("parseConfig — non-embedding defaults", () => {
  it("should have sensible defaults for all config groups", () => {
    const cfg = parseConfig({});
    expect(cfg.capture.enabled).toBe(true);
    expect(cfg.extraction.enabled).toBe(true);
    expect(cfg.recall.enabled).toBe(true);
    expect(cfg.recall.strategy).toBe("hybrid");
    expect(cfg.pipeline.everyNConversations).toBe(5);
    expect(cfg.persona.triggerEveryN).toBe(50);
  });
});
