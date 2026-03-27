/**
 * Unit tests for embedding.ts.
 *
 * EMB-01: OpenAIEmbeddingService — constructor validation
 * EMB-02: OpenAIEmbeddingService — embed() single text
 * EMB-03: OpenAIEmbeddingService — embedBatch() multiple texts
 * EMB-04: OpenAIEmbeddingService — embedBatch() large batch splitting
 * EMB-05: OpenAIEmbeddingService — API error (4xx) no retry
 * EMB-06: OpenAIEmbeddingService — API error (5xx) retry + succeed
 * EMB-07: OpenAIEmbeddingService — API error (429) retry
 * EMB-08: OpenAIEmbeddingService — timeout triggers retry
 * EMB-09: OpenAIEmbeddingService — all retries exhausted
 * EMB-10: OpenAIEmbeddingService — malformed response
 * EMB-11: OpenAIEmbeddingService — normalize output (NaN/Inf sanitization)
 * EMB-12: OpenAIEmbeddingService — dimensions only sent when non-default
 * EMB-13: OpenAIEmbeddingService — getDimensions / getProviderInfo
 * EMB-14: LocalEmbeddingService — embed() with mocked node-llama-cpp
 * EMB-15: LocalEmbeddingService — embedBatch()
 * EMB-16: LocalEmbeddingService — truncates long input
 * EMB-17: LocalEmbeddingService — getDimensions / getProviderInfo
 * EMB-18: LocalEmbeddingService — lazy init, double embed does not re-init
 * EMB-19: createEmbeddingService — OpenAI config
 * EMB-20: createEmbeddingService — local config
 * EMB-21: createEmbeddingService — fallback to local (no config)
 * EMB-22: createEmbeddingService — fallback to local (empty apiKey)
 * EMB-23: OpenAIEmbeddingService — embedBatch empty array
 * EMB-24: OpenAIEmbeddingService — response index reordering
 * EMB-25: LocalEmbeddingService — isReady() states
 * EMB-26: LocalEmbeddingService — embed() before warmup throws EmbeddingNotReadyError
 * EMB-27: LocalEmbeddingService — startWarmup() idempotent
 * EMB-28: LocalEmbeddingService — warmup failure sets failed state, retry via startWarmup()
 * EMB-29: OpenAIEmbeddingService — isReady() always true, startWarmup() no-op
 * EMB-30: createEmbeddingService — does NOT auto-call startWarmup() for local (lazy init)
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  OpenAIEmbeddingService,
  LocalEmbeddingService,
  EmbeddingNotReadyError,
  createEmbeddingService,
} from "./embedding.js";
import type { OpenAIEmbeddingConfig, LocalEmbeddingConfig } from "./embedding.js";

// ── Helpers ──

const mkLogger = () => ({
  debug: vi.fn(),
  info: vi.fn(),
  warn: vi.fn(),
  error: vi.fn(),
});

/** Build a valid OpenAI API JSON response. */
function makeOpenAIResponse(embeddings: number[][], startIndex = 0) {
  return {
    data: embeddings.map((emb, i) => ({
      index: startIndex + i,
      embedding: emb,
    })),
    usage: { prompt_tokens: 10, total_tokens: 10 },
  };
}

/** Create a simple normalized vector for testing. */
function simpleVec(dims: number, val = 1): number[] {
  const v = new Array(dims).fill(0);
  v[0] = val;
  return v;
}

/** Default config for OpenAI tests. */
const defaultOpenAIConfig: OpenAIEmbeddingConfig = {
  provider: "openai",
  apiKey: "test-api-key",
  baseUrl: "https://test.example.com/v1",
  model: "text-embedding-3-large",
  dimensions: 3072,
};

// ── OpenAI Embedding Service ──

describe("OpenAIEmbeddingService", () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  // EMB-01
  describe("EMB-01: constructor validation", () => {
    it("should throw if apiKey is missing", () => {
      expect(
        () =>
          new OpenAIEmbeddingService({
            provider: "openai",
            apiKey: "",
            baseUrl: "https://api.openai.com/v1",
            model: "text-embedding-3-large",
            dimensions: 3072,
          }),
      ).toThrow("apiKey is required");
    });

    it("should throw if baseUrl is missing", () => {
      expect(
        () =>
          new OpenAIEmbeddingService({
            provider: "openai",
            apiKey: "key",
            baseUrl: "",
            model: "text-embedding-3-large",
            dimensions: 3072,
          }),
      ).toThrow("baseUrl is required");
    });

    it("should throw if model is missing", () => {
      expect(
        () =>
          new OpenAIEmbeddingService({
            provider: "openai",
            apiKey: "key",
            baseUrl: "https://api.openai.com/v1",
            model: "",
            dimensions: 3072,
          }),
      ).toThrow("model is required");
    });

    it("should throw if dimensions is missing or invalid", () => {
      expect(
        () =>
          new OpenAIEmbeddingService({
            provider: "openai",
            apiKey: "key",
            baseUrl: "https://api.openai.com/v1",
            model: "text-embedding-3-large",
            dimensions: 0,
          }),
      ).toThrow("dimensions is required");
    });

    it("should accept valid config", () => {
      const svc = new OpenAIEmbeddingService(defaultOpenAIConfig);
      expect(svc).toBeDefined();
    });

    it("should strip trailing slashes from baseUrl", () => {
      const svc = new OpenAIEmbeddingService({
        provider: "openai",
        apiKey: "key",
        baseUrl: "https://example.com/v1///",
        model: "text-embedding-3-large",
        dimensions: 3072,
      });
      expect(svc).toBeDefined();
    });
  });

  // EMB-02
  describe("EMB-02: embed() single text", () => {
    it("should return a Float32Array for a single text", async () => {
      const vec = simpleVec(4, 2);
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => makeOpenAIResponse([vec]),
      });

      const svc = new OpenAIEmbeddingService({
        ...defaultOpenAIConfig,
        dimensions: 4,
      });
      const result = await svc.embed("hello world");

      expect(result).toBeInstanceOf(Float32Array);
      expect(result).toHaveLength(4);
      // Should be L2-normalized
      const norm = Math.sqrt(Array.from(result).reduce((s, v) => s + v * v, 0));
      expect(norm).toBeCloseTo(1.0, 4);
    });
  });

  // EMB-03
  describe("EMB-03: embedBatch() multiple texts", () => {
    it("should return correct number of embeddings", async () => {
      const vecs = [simpleVec(4, 1), simpleVec(4, 2), simpleVec(4, 3)];
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => makeOpenAIResponse(vecs),
      });

      const svc = new OpenAIEmbeddingService({
        ...defaultOpenAIConfig,
        dimensions: 4,
      });
      const results = await svc.embedBatch(["a", "b", "c"]);

      expect(results).toHaveLength(3);
      for (const r of results) {
        expect(r).toBeInstanceOf(Float32Array);
        expect(r).toHaveLength(4);
      }
    });
  });

  // EMB-04
  describe("EMB-04: embedBatch() large batch splitting", () => {
    it("should split batches larger than 256", async () => {
      const callCount = { value: 0 };
      globalThis.fetch = vi.fn().mockImplementation(async () => {
        callCount.value++;
        // Return embeddings matching the batch size from the request body
        return {
          ok: true,
          json: async () => {
            // We need to return the right number of embeddings
            // The mock doesn't see the request, so we use a fixed small vec
            const count = callCount.value === 1 ? 256 : 44; // 300 total = 256 + 44
            const vecs = Array.from({ length: count }, () => simpleVec(4));
            return makeOpenAIResponse(vecs);
          },
        };
      });

      const svc = new OpenAIEmbeddingService({
        ...defaultOpenAIConfig,
        dimensions: 4,
      });
      const texts = Array.from({ length: 300 }, (_, i) => `text-${i}`);
      const results = await svc.embedBatch(texts);

      expect(results).toHaveLength(300);
      // Should have made 2 API calls (256 + 44)
      expect(callCount.value).toBe(2);
    });
  });

  // EMB-05
  describe("EMB-05: API error (4xx) no retry", () => {
    it("should not retry on 400 client error", async () => {
      const fetchMock = vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        statusText: "Bad Request",
        text: async () => "invalid input",
      });
      globalThis.fetch = fetchMock;

      const svc = new OpenAIEmbeddingService({
        ...defaultOpenAIConfig,
        dimensions: 4,
      });

      await expect(svc.embed("test")).rejects.toThrow("HTTP 400");
      expect(fetchMock).toHaveBeenCalledTimes(1); // no retry
    });

    it("should not retry on 401 unauthorized", async () => {
      const fetchMock = vi.fn().mockResolvedValue({
        ok: false,
        status: 401,
        statusText: "Unauthorized",
        text: async () => "invalid api key",
      });
      globalThis.fetch = fetchMock;

      const svc = new OpenAIEmbeddingService({
        ...defaultOpenAIConfig,
        dimensions: 4,
      });

      await expect(svc.embed("test")).rejects.toThrow("HTTP 401");
      expect(fetchMock).toHaveBeenCalledTimes(1);
    });
  });

  // EMB-06
  describe("EMB-06: API error (5xx) retry + succeed", () => {
    it("should retry on 500 and succeed on second attempt", async () => {
      let callNum = 0;
      globalThis.fetch = vi.fn().mockImplementation(async () => {
        callNum++;
        if (callNum === 1) {
          return {
            ok: false,
            status: 500,
            statusText: "Internal Server Error",
            text: async () => "server error",
          };
        }
        return {
          ok: true,
          json: async () => makeOpenAIResponse([simpleVec(4)]),
        };
      });

      const svc = new OpenAIEmbeddingService({
        ...defaultOpenAIConfig,
        dimensions: 4,
      });
      const result = await svc.embed("test");

      expect(result).toBeInstanceOf(Float32Array);
      expect(callNum).toBe(2); // first failed, second succeeded
    });
  });

  // EMB-07
  describe("EMB-07: API error (429) retry", () => {
    it("should retry on 429 rate limit", async () => {
      let callNum = 0;
      globalThis.fetch = vi.fn().mockImplementation(async () => {
        callNum++;
        if (callNum <= 2) {
          return {
            ok: false,
            status: 429,
            statusText: "Too Many Requests",
            text: async () => "rate limited",
          };
        }
        return {
          ok: true,
          json: async () => makeOpenAIResponse([simpleVec(4)]),
        };
      });

      const svc = new OpenAIEmbeddingService({
        ...defaultOpenAIConfig,
        dimensions: 4,
      });
      const result = await svc.embed("test");

      expect(result).toBeInstanceOf(Float32Array);
      expect(callNum).toBe(3); // 2 rate-limited + 1 success
    });
  });

  // EMB-08
  describe("EMB-08: timeout triggers retry", () => {
    it("should retry on AbortError (timeout)", async () => {
      let callNum = 0;
      globalThis.fetch = vi.fn().mockImplementation(async ({ signal }: { signal?: AbortSignal }) => {
        callNum++;
        if (callNum === 1) {
          // Simulate abort/timeout
          const err = new DOMException("The operation was aborted", "AbortError");
          throw err;
        }
        return {
          ok: true,
          json: async () => makeOpenAIResponse([simpleVec(4)]),
        };
      });

      const svc = new OpenAIEmbeddingService({
        ...defaultOpenAIConfig,
        dimensions: 4,
      });
      const result = await svc.embed("test");

      expect(result).toBeInstanceOf(Float32Array);
      expect(callNum).toBe(2);
    });
  });

  // EMB-09
  describe("EMB-09: all retries exhausted", () => {
    it("should throw after MAX_RETRIES+1 attempts", async () => {
      const fetchMock = vi.fn().mockRejectedValue(new Error("network error"));
      globalThis.fetch = fetchMock;

      const svc = new OpenAIEmbeddingService({
        ...defaultOpenAIConfig,
        dimensions: 4,
      });

      await expect(svc.embed("test")).rejects.toThrow("network error");
      // 1 initial + 2 retries = 3 calls
      expect(fetchMock).toHaveBeenCalledTimes(3);
    });
  });

  // EMB-10
  describe("EMB-10: malformed response", () => {
    it("should throw on missing data array", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ usage: {} }), // no 'data' field
      });

      const svc = new OpenAIEmbeddingService({
        ...defaultOpenAIConfig,
        dimensions: 4,
      });

      await expect(svc.embed("test")).rejects.toThrow("missing 'data' array");
    });

    it("should throw on non-array data", async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ data: "not an array" }),
      });

      const svc = new OpenAIEmbeddingService({
        ...defaultOpenAIConfig,
        dimensions: 4,
      });

      await expect(svc.embed("test")).rejects.toThrow("missing 'data' array");
    });
  });

  // EMB-11
  describe("EMB-11: normalize output (NaN/Inf sanitization)", () => {
    it("should sanitize NaN values to 0 and normalize", async () => {
      const vecWithNaN = [NaN, 3, 0, 4]; // NaN→0, then normalize [0,3,0,4]
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => makeOpenAIResponse([vecWithNaN]),
      });

      const svc = new OpenAIEmbeddingService({
        ...defaultOpenAIConfig,
        dimensions: 4,
      });
      const result = await svc.embed("test");

      expect(result[0]).toBeCloseTo(0, 5); // NaN → 0
      // magnitude = sqrt(0 + 9 + 0 + 16) = 5
      expect(result[1]).toBeCloseTo(3 / 5, 5);
      expect(result[3]).toBeCloseTo(4 / 5, 5);
    });

    it("should sanitize Infinity values", async () => {
      const vecWithInf = [Infinity, 0, -Infinity, 1];
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => makeOpenAIResponse([vecWithInf]),
      });

      const svc = new OpenAIEmbeddingService({
        ...defaultOpenAIConfig,
        dimensions: 4,
      });
      const result = await svc.embed("test");

      expect(result[0]).toBeCloseTo(0, 5); // Inf → 0
      expect(result[2]).toBeCloseTo(0, 5); // -Inf → 0
      // Only [3]=1 is non-zero, so normalized to 1.0
      expect(result[3]).toBeCloseTo(1.0, 5);
    });

    it("should handle all-zero vector gracefully (no division by zero)", async () => {
      const zeroVec = [0, 0, 0, 0];
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => makeOpenAIResponse([zeroVec]),
      });

      const svc = new OpenAIEmbeddingService({
        ...defaultOpenAIConfig,
        dimensions: 4,
      });
      const result = await svc.embed("test");

      // All zeros — magnitude < 1e-10, returned as-is
      for (let i = 0; i < 4; i++) {
        expect(result[i]).toBe(0);
      }
    });
  });

  // EMB-12
  describe("EMB-12: dimensions always sent in request body", () => {
    it("should always send dimensions in request body", async () => {
      let capturedBody: string | undefined;
      globalThis.fetch = vi.fn().mockImplementation(async (_url: string, init: RequestInit) => {
        capturedBody = init.body as string;
        return {
          ok: true,
          json: async () => makeOpenAIResponse([simpleVec(4)]),
        };
      });

      const svc = new OpenAIEmbeddingService({
        ...defaultOpenAIConfig,
        dimensions: 3072,
      });
      await svc.embed("test");

      const body = JSON.parse(capturedBody!);
      expect(body.dimensions).toBe(3072);
    });

    it("should send custom dimensions in request body", async () => {
      let capturedBody: string | undefined;
      globalThis.fetch = vi.fn().mockImplementation(async (_url: string, init: RequestInit) => {
        capturedBody = init.body as string;
        return {
          ok: true,
          json: async () => makeOpenAIResponse([simpleVec(4)]),
        };
      });

      const svc = new OpenAIEmbeddingService({
        ...defaultOpenAIConfig,
        dimensions: 1536,
      });
      await svc.embed("test");

      const body = JSON.parse(capturedBody!);
      expect(body.dimensions).toBe(1536);
    });
  });

  // EMB-13
  describe("EMB-13: getDimensions / getProviderInfo", () => {
    it("should return configured dimensions", () => {
      const svc = new OpenAIEmbeddingService({
        ...defaultOpenAIConfig,
        dimensions: 1536,
      });
      expect(svc.getDimensions()).toBe(1536);
    });

    it("should return provider info", () => {
      const svc = new OpenAIEmbeddingService(defaultOpenAIConfig);
      const info = svc.getProviderInfo();
      expect(info.provider).toBe("openai");
      expect(info.model).toBe("text-embedding-3-large");
    });
  });

  // EMB-23
  describe("EMB-23: embedBatch empty array", () => {
    it("should return empty array for empty input", async () => {
      const svc = new OpenAIEmbeddingService(defaultOpenAIConfig);
      const results = await svc.embedBatch([]);
      expect(results).toEqual([]);
    });
  });

  // EMB-24
  describe("EMB-24: response index reordering", () => {
    it("should sort by index to ensure correct order", async () => {
      // API returns out-of-order indices
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          data: [
            { index: 2, embedding: [0, 0, 1, 0] },
            { index: 0, embedding: [1, 0, 0, 0] },
            { index: 1, embedding: [0, 1, 0, 0] },
          ],
          usage: { prompt_tokens: 10, total_tokens: 10 },
        }),
      });

      const svc = new OpenAIEmbeddingService({
        ...defaultOpenAIConfig,
        dimensions: 4,
      });
      const results = await svc.embedBatch(["a", "b", "c"]);

      expect(results).toHaveLength(3);
      // After sorting by index: [1,0,0,0], [0,1,0,0], [0,0,1,0]
      // After normalize: each already unit vector
      expect(results[0][0]).toBeCloseTo(1.0, 5);
      expect(results[1][1]).toBeCloseTo(1.0, 5);
      expect(results[2][2]).toBeCloseTo(1.0, 5);
    });
  });

  // EMB-29
  describe("EMB-29: OpenAI — isReady() always true, startWarmup() no-op", () => {
    it("should always report ready (stateless HTTP)", () => {
      const svc = new OpenAIEmbeddingService(defaultOpenAIConfig);
      expect(svc.isReady()).toBe(true);
    });

    it("should not throw on startWarmup()", () => {
      const svc = new OpenAIEmbeddingService(defaultOpenAIConfig);
      expect(() => svc.startWarmup()).not.toThrow();
      expect(svc.isReady()).toBe(true);
    });
  });

  // Additional: verify Authorization header and URL
  describe("EMB-extra: request format", () => {
    it("should send correct URL and headers", async () => {
      let capturedUrl = "";
      let capturedHeaders: Record<string, string> = {};
      globalThis.fetch = vi.fn().mockImplementation(async (url: string, init: RequestInit) => {
        capturedUrl = url;
        capturedHeaders = Object.fromEntries(
          Object.entries(init.headers as Record<string, string>),
        );
        return {
          ok: true,
          json: async () => makeOpenAIResponse([simpleVec(4)]),
        };
      });

      const svc = new OpenAIEmbeddingService({
        provider: "openai",
        apiKey: "sk-test-123",
        baseUrl: "https://my-api.example.com/v1",
        model: "text-embedding-3-large",
        dimensions: 4,
      });
      await svc.embed("hello");

      expect(capturedUrl).toBe("https://my-api.example.com/v1/embeddings");
      expect(capturedHeaders["Authorization"]).toBe("Bearer sk-test-123");
      expect(capturedHeaders["Content-Type"]).toBe("application/json");
    });

    it("should include model in request body", async () => {
      let capturedBody = "";
      globalThis.fetch = vi.fn().mockImplementation(async (_url: string, init: RequestInit) => {
        capturedBody = init.body as string;
        return {
          ok: true,
          json: async () => makeOpenAIResponse([simpleVec(4)]),
        };
      });

      const svc = new OpenAIEmbeddingService({
        ...defaultOpenAIConfig,
        model: "my-custom-model",
        dimensions: 4,
      });
      await svc.embed("hello");

      const body = JSON.parse(capturedBody);
      expect(body.model).toBe("my-custom-model");
      expect(body.input).toEqual(["hello"]);
    });
  });
});

// ── Local Embedding Service ──

describe("LocalEmbeddingService", () => {
  // EMB-14
  describe("EMB-14: embed() with mocked node-llama-cpp", () => {
    it("should return normalized Float32Array from local model", async () => {
      // Mock dynamic import of node-llama-cpp
      const mockGetEmbeddingFor = vi.fn().mockResolvedValue({
        vector: new Float32Array([3, 4, 0, 0]),
      });

      const mockModel = {
        createEmbeddingContext: vi.fn().mockResolvedValue({
          getEmbeddingFor: mockGetEmbeddingFor,
        }),
      };

      const mockLlama = {
        loadModel: vi.fn().mockResolvedValue(mockModel),
      };

      // We need to mock the dynamic import. Use vi.mock for the module.
      vi.doMock("node-llama-cpp", () => ({
        getLlama: vi.fn().mockResolvedValue(mockLlama),
        resolveModelFile: vi.fn().mockResolvedValue("/tmp/model.gguf"),
        LlamaLogLevel: { error: 2 },
      }));

      // Re-import to pick up mocked module
      const { LocalEmbeddingService: MockedLocal } = await import("./embedding.js");
      const svc = new MockedLocal(undefined, mkLogger());
      svc.startWarmup();
      await svc.waitForReady();
      const result = await svc.embed("test text");

      expect(result).toBeInstanceOf(Float32Array);
      // [3,4,0,0] → magnitude=5 → normalized [0.6, 0.8, 0, 0]
      expect(result[0]).toBeCloseTo(0.6, 4);
      expect(result[1]).toBeCloseTo(0.8, 4);
      expect(mockGetEmbeddingFor).toHaveBeenCalledWith("test text");

      vi.doUnmock("node-llama-cpp");
    });
  });

  // EMB-15
  describe("EMB-15: embedBatch()", () => {
    it("should embed each text sequentially", async () => {
      let callCount = 0;
      const mockGetEmbeddingFor = vi.fn().mockImplementation(async () => {
        callCount++;
        return { vector: new Float32Array([callCount, 0, 0, 0]) };
      });

      const mockModel = {
        createEmbeddingContext: vi.fn().mockResolvedValue({
          getEmbeddingFor: mockGetEmbeddingFor,
        }),
      };

      vi.doMock("node-llama-cpp", () => ({
        getLlama: vi.fn().mockResolvedValue({
          loadModel: vi.fn().mockResolvedValue(mockModel),
        }),
        resolveModelFile: vi.fn().mockResolvedValue("/tmp/model.gguf"),
        LlamaLogLevel: { error: 2 },
      }));

      const { LocalEmbeddingService: MockedLocal } = await import("./embedding.js");
      const svc = new MockedLocal(undefined, mkLogger());
      svc.startWarmup();
      await svc.waitForReady();
      const results = await svc.embedBatch(["a", "b", "c"]);

      expect(results).toHaveLength(3);
      expect(mockGetEmbeddingFor).toHaveBeenCalledTimes(3);

      vi.doUnmock("node-llama-cpp");
    });

    it("should return empty array for empty input", async () => {
      const svc = new LocalEmbeddingService();
      const results = await svc.embedBatch([]);
      expect(results).toEqual([]);
    });
  });

  // EMB-16
  describe("EMB-16: truncates long input", () => {
    it("should truncate text longer than 512 chars", async () => {
      let capturedText = "";
      const mockGetEmbeddingFor = vi.fn().mockImplementation(async (text: string) => {
        capturedText = text;
        return { vector: new Float32Array([1, 0, 0, 0]) };
      });

      const mockModel = {
        createEmbeddingContext: vi.fn().mockResolvedValue({
          getEmbeddingFor: mockGetEmbeddingFor,
        }),
      };

      vi.doMock("node-llama-cpp", () => ({
        getLlama: vi.fn().mockResolvedValue({
          loadModel: vi.fn().mockResolvedValue(mockModel),
        }),
        resolveModelFile: vi.fn().mockResolvedValue("/tmp/model.gguf"),
        LlamaLogLevel: { error: 2 },
      }));

      const { LocalEmbeddingService: MockedLocal } = await import("./embedding.js");
      const log = mkLogger();
      const svc = new MockedLocal(undefined, log);
      svc.startWarmup();
      await svc.waitForReady();

      const longText = "A".repeat(1000);
      await svc.embed(longText);

      expect(capturedText).toHaveLength(512);
      expect(log.debug).toHaveBeenCalled();

      vi.doUnmock("node-llama-cpp");
    });

    it("should NOT truncate text within limit", async () => {
      let capturedText = "";
      const mockGetEmbeddingFor = vi.fn().mockImplementation(async (text: string) => {
        capturedText = text;
        return { vector: new Float32Array([1, 0, 0, 0]) };
      });

      const mockModel = {
        createEmbeddingContext: vi.fn().mockResolvedValue({
          getEmbeddingFor: mockGetEmbeddingFor,
        }),
      };

      vi.doMock("node-llama-cpp", () => ({
        getLlama: vi.fn().mockResolvedValue({
          loadModel: vi.fn().mockResolvedValue(mockModel),
        }),
        resolveModelFile: vi.fn().mockResolvedValue("/tmp/model.gguf"),
        LlamaLogLevel: { error: 2 },
      }));

      const { LocalEmbeddingService: MockedLocal } = await import("./embedding.js");
      const svc = new MockedLocal(undefined, mkLogger());
      svc.startWarmup();
      await svc.waitForReady();

      const shortText = "Hello world";
      await svc.embed(shortText);

      expect(capturedText).toBe(shortText);

      vi.doUnmock("node-llama-cpp");
    });
  });

  // EMB-17
  describe("EMB-17: getDimensions / getProviderInfo", () => {
    it("should return 768 dimensions", () => {
      const svc = new LocalEmbeddingService();
      expect(svc.getDimensions()).toBe(768);
    });

    it("should return local provider info with default model", () => {
      const svc = new LocalEmbeddingService();
      const info = svc.getProviderInfo();
      expect(info.provider).toBe("local");
      expect(info.model).toContain("embeddinggemma");
    });

    it("should return custom model path in provider info", () => {
      const svc = new LocalEmbeddingService({
        provider: "local",
        modelPath: "/custom/model.gguf",
      });
      expect(svc.getProviderInfo().model).toBe("/custom/model.gguf");
    });
  });

  // EMB-18
  describe("EMB-18: startWarmup + double embed does not re-init", () => {
    it("should only initialize once across multiple embed calls", async () => {
      let initCount = 0;
      const mockGetEmbeddingFor = vi.fn().mockResolvedValue({
        vector: new Float32Array([1, 0, 0, 0]),
      });

      const mockModel = {
        createEmbeddingContext: vi.fn().mockImplementation(async () => {
          initCount++;
          return { getEmbeddingFor: mockGetEmbeddingFor };
        }),
      };

      vi.doMock("node-llama-cpp", () => ({
        getLlama: vi.fn().mockResolvedValue({
          loadModel: vi.fn().mockResolvedValue(mockModel),
        }),
        resolveModelFile: vi.fn().mockResolvedValue("/tmp/model.gguf"),
        LlamaLogLevel: { error: 2 },
      }));

      const { LocalEmbeddingService: MockedLocal } = await import("./embedding.js");
      const svc = new MockedLocal(undefined, mkLogger());
      svc.startWarmup();
      await svc.waitForReady();

      await svc.embed("first");
      await svc.embed("second");
      await svc.embed("third");

      // createEmbeddingContext should only be called once
      expect(initCount).toBe(1);
      expect(mockGetEmbeddingFor).toHaveBeenCalledTimes(3);

      vi.doUnmock("node-llama-cpp");
    });
  });

  // EMB-25
  describe("EMB-25: isReady() states", () => {
    it("should return false before startWarmup()", () => {
      const svc = new LocalEmbeddingService();
      expect(svc.isReady()).toBe(false);
    });

    it("should return true after successful warmup", async () => {
      const mockModel = {
        createEmbeddingContext: vi.fn().mockResolvedValue({
          getEmbeddingFor: vi.fn().mockResolvedValue({ vector: new Float32Array([1, 0]) }),
        }),
      };

      vi.doMock("node-llama-cpp", () => ({
        getLlama: vi.fn().mockResolvedValue({
          loadModel: vi.fn().mockResolvedValue(mockModel),
        }),
        resolveModelFile: vi.fn().mockResolvedValue("/tmp/model.gguf"),
        LlamaLogLevel: { error: 2 },
      }));

      const { LocalEmbeddingService: MockedLocal } = await import("./embedding.js");
      const svc = new MockedLocal(undefined, mkLogger());
      expect(svc.isReady()).toBe(false);

      svc.startWarmup();
      // During warmup, might still be false
      await svc.waitForReady();
      expect(svc.isReady()).toBe(true);

      vi.doUnmock("node-llama-cpp");
    });

    it("should return false after close()", async () => {
      const mockModel = {
        createEmbeddingContext: vi.fn().mockResolvedValue({
          getEmbeddingFor: vi.fn().mockResolvedValue({ vector: new Float32Array([1, 0]) }),
        }),
      };

      vi.doMock("node-llama-cpp", () => ({
        getLlama: vi.fn().mockResolvedValue({
          loadModel: vi.fn().mockResolvedValue(mockModel),
        }),
        resolveModelFile: vi.fn().mockResolvedValue("/tmp/model.gguf"),
        LlamaLogLevel: { error: 2 },
      }));

      const { LocalEmbeddingService: MockedLocal } = await import("./embedding.js");
      const svc = new MockedLocal(undefined, mkLogger());
      svc.startWarmup();
      await svc.waitForReady();
      expect(svc.isReady()).toBe(true);

      svc.close();
      expect(svc.isReady()).toBe(false);

      vi.doUnmock("node-llama-cpp");
    });
  });

  // EMB-26
  describe("EMB-26: embed() before warmup throws EmbeddingNotReadyError", () => {
    it("should throw EmbeddingNotReadyError when warmup not started (idle)", async () => {
      const { EmbeddingNotReadyError: ENRE } = await import("./embedding.js");
      const svc = new LocalEmbeddingService();
      await expect(svc.embed("test")).rejects.toThrow(ENRE);
      await expect(svc.embed("test")).rejects.toThrow("warmup has not been started");
    });

    it("should throw EmbeddingNotReadyError when warmup is still in progress", async () => {
      // Use a mock that takes a long time to resolve
      let resolveInit!: () => void;
      const initBlocker = new Promise<void>((resolve) => { resolveInit = resolve; });

      vi.doMock("node-llama-cpp", () => ({
        getLlama: vi.fn().mockImplementation(async () => {
          await initBlocker;
          return {
            loadModel: vi.fn().mockResolvedValue({
              createEmbeddingContext: vi.fn().mockResolvedValue({
                getEmbeddingFor: vi.fn(),
              }),
            }),
          };
        }),
        resolveModelFile: vi.fn().mockResolvedValue("/tmp/model.gguf"),
        LlamaLogLevel: { error: 2 },
      }));

      const { LocalEmbeddingService: MockedLocal, EmbeddingNotReadyError: ENRE } = await import("./embedding.js");
      const svc = new MockedLocal(undefined, mkLogger());
      svc.startWarmup();

      // embed() should throw while still initializing
      await expect(svc.embed("test")).rejects.toThrow(ENRE);
      await expect(svc.embed("test")).rejects.toThrow("still loading");

      // Unblock init and clean up
      resolveInit();
      await svc.waitForReady();

      vi.doUnmock("node-llama-cpp");
    });
  });

  // EMB-27
  describe("EMB-27: startWarmup() idempotent", () => {
    it("should not re-initialize when called multiple times", async () => {
      let initCount = 0;
      const mockModel = {
        createEmbeddingContext: vi.fn().mockImplementation(async () => {
          initCount++;
          return { getEmbeddingFor: vi.fn().mockResolvedValue({ vector: new Float32Array([1, 0]) }) };
        }),
      };

      vi.doMock("node-llama-cpp", () => ({
        getLlama: vi.fn().mockResolvedValue({
          loadModel: vi.fn().mockResolvedValue(mockModel),
        }),
        resolveModelFile: vi.fn().mockResolvedValue("/tmp/model.gguf"),
        LlamaLogLevel: { error: 2 },
      }));

      const { LocalEmbeddingService: MockedLocal } = await import("./embedding.js");
      const svc = new MockedLocal(undefined, mkLogger());

      svc.startWarmup();
      svc.startWarmup(); // second call should be no-op
      svc.startWarmup(); // third call should be no-op
      await svc.waitForReady();

      expect(initCount).toBe(1);

      // After ready, startWarmup should also be no-op
      svc.startWarmup();
      expect(initCount).toBe(1);

      vi.doUnmock("node-llama-cpp");
    });
  });

  // EMB-28
  describe("EMB-28: warmup failure sets failed state, retry via startWarmup()", () => {
    it("should set failed state on warmup error and allow retry", async () => {
      let callCount = 0;
      const mockModel = {
        createEmbeddingContext: vi.fn().mockResolvedValue({
          getEmbeddingFor: vi.fn().mockResolvedValue({ vector: new Float32Array([1, 0]) }),
        }),
      };

      vi.doMock("node-llama-cpp", () => ({
        getLlama: vi.fn().mockImplementation(async () => {
          callCount++;
          if (callCount === 1) {
            throw new Error("GPU not available");
          }
          return {
            loadModel: vi.fn().mockResolvedValue(mockModel),
          };
        }),
        resolveModelFile: vi.fn().mockResolvedValue("/tmp/model.gguf"),
        LlamaLogLevel: { error: 2 },
      }));

      const { LocalEmbeddingService: MockedLocal, EmbeddingNotReadyError: ENRE } = await import("./embedding.js");
      const svc = new MockedLocal(undefined, mkLogger());

      // First warmup: fails
      svc.startWarmup();
      await svc.waitForReady().catch(() => {}); // swallow the error in the test
      expect(svc.isReady()).toBe(false);

      // embed should throw with "failed" info
      await expect(svc.embed("test")).rejects.toThrow(ENRE);
      await expect(svc.embed("test")).rejects.toThrow("initialization failed");

      // Retry warmup: should succeed now
      svc.startWarmup();
      await svc.waitForReady();
      expect(svc.isReady()).toBe(true);

      vi.doUnmock("node-llama-cpp");
    });
  });
});

// ── Factory: createEmbeddingService ──

describe("createEmbeddingService", () => {
  // EMB-19
  describe("EMB-19: OpenAI config", () => {
    it("should create OpenAIEmbeddingService for openai config with apiKey", () => {
      const log = mkLogger();
      const svc = createEmbeddingService(
        {
          provider: "openai",
          apiKey: "sk-test",
          baseUrl: "https://api.openai.com/v1",
          model: "text-embedding-3-small",
          dimensions: 1536,
        },
        log,
      );
      expect(svc).toBeInstanceOf(OpenAIEmbeddingService);
      expect(svc.getProviderInfo().provider).toBe("openai");
      expect(log.info).toHaveBeenCalledWith(expect.stringContaining("remote embedding"));
    });

    it("should create OpenAIEmbeddingService for any non-local provider with apiKey", () => {
      const log = mkLogger();
      const svc = createEmbeddingService(
        {
          provider: "deepseek",
          apiKey: "sk-ds-test",
          baseUrl: "https://api.deepseek.com/v1",
          model: "deepseek-embedding",
          dimensions: 1024,
        },
        log,
      );
      expect(svc).toBeInstanceOf(OpenAIEmbeddingService);
      expect(svc.getProviderInfo().provider).toBe("deepseek");
      expect(log.info).toHaveBeenCalledWith(expect.stringContaining("provider=deepseek"));
    });
  });

  // EMB-20
  describe("EMB-20: local config", () => {
    it("should create LocalEmbeddingService for local config", () => {
      const log = mkLogger();
      const svc = createEmbeddingService({ provider: "local" }, log);
      expect(svc).toBeInstanceOf(LocalEmbeddingService);
      expect(svc.getProviderInfo().provider).toBe("local");
      expect(log.info).toHaveBeenCalledWith(expect.stringContaining("local embedding"));
    });
  });

  // EMB-21
  describe("EMB-21: fallback to local (no config)", () => {
    it("should fall back to LocalEmbeddingService when config is undefined", () => {
      const log = mkLogger();
      const svc = createEmbeddingService(undefined, log);
      expect(svc).toBeInstanceOf(LocalEmbeddingService);
      expect(log.info).toHaveBeenCalledWith(expect.stringContaining("falling back"));
    });
  });

  // EMB-22
  describe("EMB-22: fallback to local (empty apiKey)", () => {
    it("should fall back to LocalEmbeddingService when apiKey is empty", () => {
      const log = mkLogger();
      const svc = createEmbeddingService(
        { provider: "openai", apiKey: "" } as OpenAIEmbeddingConfig,
        log,
      );
      // Empty apiKey → falls through to local fallback
      expect(svc).toBeInstanceOf(LocalEmbeddingService);
    });
  });

  // EMB-30
  describe("EMB-30: createEmbeddingService does NOT auto-call startWarmup() for local", () => {
    it("should NOT start warmup automatically — caller is responsible", () => {
      const log = mkLogger();
      const svc = createEmbeddingService({ provider: "local" }, log);

      // Service should be created but NOT ready (warmup not triggered)
      expect(svc).toBeInstanceOf(LocalEmbeddingService);
      expect(svc.isReady()).toBe(false);
    });

    it("should NOT start warmup for fallback-to-local path either", () => {
      const log = mkLogger();
      const svc = createEmbeddingService(undefined, log);

      expect(svc).toBeInstanceOf(LocalEmbeddingService);
      expect(svc.isReady()).toBe(false);
    });
  });
});
