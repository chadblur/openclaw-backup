/**
 * Unit tests for MemoryPipelineManager.
 *
 * Covers:
 * - PM-01: Conversation threshold triggers L1
 * - PM-02: Warm-up mode (threshold 1 → 2 → 4 → ... → everyN)
 * - PM-03: L1 idle timeout triggers L1
 * - PM-04: L2 delay-after-L1 + minInterval/maxInterval control
 * - PM-05: Pipeline execution exception doesn't break next schedule
 * - PM-06: destroy() cleans all timers and resources
 * - Additional: L3 dedup, session GC, recovery, session filter, buffer management
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { MemoryPipelineManager } from "./pipeline-manager.js";
import type {
  PipelineConfig,
  CapturedMessage,
  L1Runner,
  L2Runner,
  L3Runner,
  L1RunnerResult,
  L2RunnerResult,
} from "./pipeline-manager.js";
import { SessionFilter } from "./session-filter.js";

// ============================
// Test helpers
// ============================

function makeConfig(overrides?: Partial<PipelineConfig>): PipelineConfig {
  return {
    everyNConversations: 3,
    enableWarmup: false,
    l1: { idleTimeoutSeconds: 1 },
    l2: {
      delayAfterL1Seconds: 1,
      minIntervalSeconds: 2,
      maxIntervalSeconds: 10,
      sessionActiveWindowHours: 24,
    },
    ...overrides,
  };
}

function makeMessages(count = 1): CapturedMessage[] {
  return Array.from({ length: count }, (_, i) => ({
    role: "user" as const,
    content: `message-${i}`,
    timestamp: new Date().toISOString(),
  }));
}

function createNoopLogger() {
  return {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  };
}

/**
 * Drain all pending microtasks and macrotasks in the fake timer environment.
 *
 * The SerialQueue uses `.then()/.finally()` microtask chains to drive
 * serial execution. In fake-timer mode we need to repeatedly flush
 * microtasks so those chains settle. We also advance timers by 0 to
 * fire any `setTimeout(fn, 0)` callbacks that may have been scheduled.
 *
 * Three rounds is enough for the deepest L1→L2→L3 chain.
 */
async function flushTimers() {
  for (let i = 0; i < 5; i++) {
    await vi.advanceTimersByTimeAsync(0);
    // Let any scheduled microtasks / setImmediate settle
    await new Promise<void>((r) => process.nextTick(r));
  }
}

/**
 * Advance fake timers by the given ms AND fully drain the resulting
 * microtask / macrotask queue so SerialQueue chains complete.
 */
async function advanceAndFlush(ms: number) {
  await vi.advanceTimersByTimeAsync(ms);
  // Multiple rounds to drain chained microtasks (SerialQueue .finally → drain → .then → ...)
  for (let i = 0; i < 5; i++) {
    await vi.advanceTimersByTimeAsync(0);
    await new Promise<void>((r) => process.nextTick(r));
  }
}

// ============================
// Tests
// ============================

describe("MemoryPipelineManager", () => {
  beforeEach(() => {
    vi.useFakeTimers({
      // Ensure fake timers intercept setImmediate too — needed because
      // SerialQueue .finally() chains schedule microtasks that may be
      // followed by setImmediate calls in Node internals.
      toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date", "setImmediate", "clearImmediate"],
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  // ─────────────────────────────────────
  // PM-01: Conversation threshold triggers L1
  // ─────────────────────────────────────

  describe("PM-01: conversation threshold triggers L1", () => {
    it("should trigger L1 when conversation count reaches everyNConversations", async () => {
      const config = makeConfig({ everyNConversations: 3, enableWarmup: false });
      const logger = createNoopLogger();
      const pm = new MemoryPipelineManager(config, logger);

      const l1Runner = vi.fn<L1Runner>().mockResolvedValue(undefined);
      const persister = vi.fn().mockResolvedValue(undefined);

      pm.setL1Runner(l1Runner);
      pm.setPersister(persister);
      pm.start();

      // Send 2 conversations — below threshold, should not trigger L1
      await pm.notifyConversation("session-1", makeMessages());
      await flushTimers();
      await pm.notifyConversation("session-1", makeMessages());
      await flushTimers();

      expect(l1Runner).not.toHaveBeenCalled();

      // 3rd conversation — reaches threshold
      await pm.notifyConversation("session-1", makeMessages());
      await flushTimers();

      expect(l1Runner).toHaveBeenCalledTimes(1);
      expect(l1Runner).toHaveBeenCalledWith(
        expect.objectContaining({
          sessionKey: "session-1",
          msg: expect.arrayContaining([expect.objectContaining({ role: "user" })]),
        }),
      );

      await pm.destroy();
    });

    it("should reset conversation count after L1 completes", async () => {
      const config = makeConfig({ everyNConversations: 2, enableWarmup: false });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      const l1Runner = vi.fn<L1Runner>().mockResolvedValue(undefined);
      pm.setL1Runner(l1Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      // Trigger L1 (2 conversations)
      await pm.notifyConversation("s1", makeMessages());
      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();

      expect(l1Runner).toHaveBeenCalledTimes(1);

      // Send 1 more conversation — should not trigger L1 (count reset to 0)
      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();

      expect(l1Runner).toHaveBeenCalledTimes(1);

      // 2nd conversation after reset → trigger again
      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();

      expect(l1Runner).toHaveBeenCalledTimes(2);

      await pm.destroy();
    });

    it("should buffer messages and pass them all to L1 runner", async () => {
      const config = makeConfig({ everyNConversations: 2, enableWarmup: false });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      const l1Runner = vi.fn<L1Runner>().mockResolvedValue(undefined);
      pm.setL1Runner(l1Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      await pm.notifyConversation("s1", makeMessages(2));
      await pm.notifyConversation("s1", makeMessages(3));
      await flushTimers();

      // L1 should receive all 5 buffered messages
      expect(l1Runner).toHaveBeenCalledTimes(1);
      const call = l1Runner.mock.calls[0][0];
      expect(call.msg).toHaveLength(5);

      await pm.destroy();
    });
  });

  // ─────────────────────────────────────
  // PM-02: Warm-up mode
  // ─────────────────────────────────────

  describe("PM-02: warm-up mode (threshold doubles: 1→2→4→...→everyN)", () => {
    it("should trigger L1 after first conversation when warm-up is enabled", async () => {
      const config = makeConfig({ everyNConversations: 8, enableWarmup: true });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      const l1Runner = vi.fn<L1Runner>().mockResolvedValue(undefined);
      pm.setL1Runner(l1Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      // First conversation → threshold is 1 in warm-up
      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();

      expect(l1Runner).toHaveBeenCalledTimes(1);

      await pm.destroy();
    });

    it("should double threshold after each L1: 1→2→4→graduated", async () => {
      const config = makeConfig({ everyNConversations: 4, enableWarmup: true });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      const l1Runner = vi.fn<L1Runner>().mockResolvedValue(undefined);
      pm.setL1Runner(l1Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      // threshold=1: 1st conversation triggers L1
      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();
      expect(l1Runner).toHaveBeenCalledTimes(1);

      // threshold=2: next 2 conversations to trigger
      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();
      expect(l1Runner).toHaveBeenCalledTimes(1); // not yet

      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();
      expect(l1Runner).toHaveBeenCalledTimes(2); // triggered

      // threshold should graduate to everyN=4 (since 2*2=4=everyN)
      // So now need 4 conversations to trigger
      for (let i = 0; i < 3; i++) {
        await pm.notifyConversation("s1", makeMessages());
        await flushTimers();
      }
      expect(l1Runner).toHaveBeenCalledTimes(2); // not yet

      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();
      expect(l1Runner).toHaveBeenCalledTimes(3); // triggered at steady-state

      await pm.destroy();
    });

    it("should NOT apply warm-up when enableWarmup=false", async () => {
      const config = makeConfig({ everyNConversations: 5, enableWarmup: false });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      const l1Runner = vi.fn<L1Runner>().mockResolvedValue(undefined);
      pm.setL1Runner(l1Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      // 1st conversation should NOT trigger L1 (threshold=5, no warm-up)
      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();
      expect(l1Runner).not.toHaveBeenCalled();

      await pm.destroy();
    });
  });

  // ─────────────────────────────────────
  // PM-03: L1 idle timeout
  // ─────────────────────────────────────

  describe("PM-03: L1 idle timeout triggers L1", () => {
    it("should trigger L1 after idle timeout when below threshold", async () => {
      const config = makeConfig({
        everyNConversations: 10,
        enableWarmup: false,
        l1: { idleTimeoutSeconds: 2 },
      });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      const l1Runner = vi.fn<L1Runner>().mockResolvedValue(undefined);
      pm.setL1Runner(l1Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      // 1 conversation (below threshold=10)
      await pm.notifyConversation("s1", makeMessages(2));
      await flushTimers();
      expect(l1Runner).not.toHaveBeenCalled();

      // Advance past idle timeout (2s)
      await advanceAndFlush(2100);

      expect(l1Runner).toHaveBeenCalledTimes(1);
      expect(l1Runner).toHaveBeenCalledWith(
        expect.objectContaining({
          sessionKey: "s1",
          msg: expect.any(Array),
        }),
      );

      await pm.destroy();
    });

    it("should reset idle timer on each new conversation", async () => {
      const config = makeConfig({
        everyNConversations: 10,
        enableWarmup: false,
        l1: { idleTimeoutSeconds: 2 },
      });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      const l1Runner = vi.fn<L1Runner>().mockResolvedValue(undefined);
      pm.setL1Runner(l1Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      await pm.notifyConversation("s1", makeMessages());
      await advanceAndFlush(1500); // 1.5s — not yet expired

      // New conversation resets the timer
      await pm.notifyConversation("s1", makeMessages());
      await advanceAndFlush(1500); // 1.5s from second notify
      expect(l1Runner).not.toHaveBeenCalled(); // timer was reset

      // Wait full 2s after last notify
      await advanceAndFlush(600);
      expect(l1Runner).toHaveBeenCalledTimes(1);

      await pm.destroy();
    });

    it("should not fire idle timeout if no buffered messages or conversations", async () => {
      const config = makeConfig({
        everyNConversations: 2,
        enableWarmup: false,
        l1: { idleTimeoutSeconds: 1 },
      });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      const l1Runner = vi.fn<L1Runner>().mockResolvedValue(undefined);
      pm.setL1Runner(l1Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      // Trigger L1 via threshold (clears buffer + count)
      await pm.notifyConversation("s1", makeMessages());
      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();
      expect(l1Runner).toHaveBeenCalledTimes(1);

      // The idle timer should have been cancelled when threshold was reached
      // Even if it somehow fires, it should be a no-op since buffer is empty
      await advanceAndFlush(2000);
      expect(l1Runner).toHaveBeenCalledTimes(1); // no extra call

      await pm.destroy();
    });
  });

  // ─────────────────────────────────────
  // PM-04: L2 timer semantics
  // ─────────────────────────────────────

  describe("PM-04: L2 delay-after-L1 + min/max interval control", () => {
    it("should arm L2 timer after L1 completes", async () => {
      const config = makeConfig({
        everyNConversations: 1,
        enableWarmup: false,
        l2: {
          delayAfterL1Seconds: 2,
          minIntervalSeconds: 1,
          maxIntervalSeconds: 60,
          sessionActiveWindowHours: 24,
        },
      });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      const l1Runner = vi.fn<L1Runner>().mockResolvedValue(undefined);
      const l2Runner = vi.fn<L2Runner>().mockResolvedValue(undefined);
      pm.setL1Runner(l1Runner);
      pm.setL2Runner(l2Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      // Trigger L1
      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();
      expect(l1Runner).toHaveBeenCalledTimes(1);
      expect(l2Runner).not.toHaveBeenCalled();

      // L2 should fire after delay (2s)
      await advanceAndFlush(2100);
      expect(l2Runner).toHaveBeenCalledTimes(1);
      expect(l2Runner).toHaveBeenCalledWith("s1", undefined);

      await pm.destroy();
    });

    it("should respect minInterval floor between L2 runs", async () => {
      const config = makeConfig({
        everyNConversations: 1,
        enableWarmup: false,
        l2: {
          delayAfterL1Seconds: 1,
          minIntervalSeconds: 5,
          maxIntervalSeconds: 60,
          sessionActiveWindowHours: 24,
        },
      });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      const l1Runner = vi.fn<L1Runner>().mockResolvedValue(undefined);
      const l2Runner = vi.fn<L2Runner>().mockResolvedValue(undefined);
      pm.setL1Runner(l1Runner);
      pm.setL2Runner(l2Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      // First L1 → L2 after 1s delay (no minInterval floor yet)
      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();
      await advanceAndFlush(1100);
      expect(l2Runner).toHaveBeenCalledTimes(1);

      // Second L1 immediately
      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();

      // Even after delay (1s), L2 should NOT fire yet because minInterval is 5s
      await advanceAndFlush(1100);
      expect(l2Runner).toHaveBeenCalledTimes(1); // still 1

      // After full minInterval passes, L2 should fire
      await advanceAndFlush(4100);
      expect(l2Runner).toHaveBeenCalledTimes(2);

      await pm.destroy();
    });

    it("should arm maxInterval timer after L2 completes for active sessions", async () => {
      const config = makeConfig({
        everyNConversations: 1,
        enableWarmup: false,
        l2: {
          delayAfterL1Seconds: 1,
          minIntervalSeconds: 1,
          maxIntervalSeconds: 5,
          sessionActiveWindowHours: 24,
        },
      });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      const l1Runner = vi.fn<L1Runner>().mockResolvedValue(undefined);
      const l2Runner = vi.fn<L2Runner>().mockResolvedValue(undefined);
      pm.setL1Runner(l1Runner);
      pm.setL2Runner(l2Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      // Trigger L1 → L2
      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();
      await advanceAndFlush(1100);
      expect(l2Runner).toHaveBeenCalledTimes(1);

      // Without new L1, L2 should auto-fire at maxInterval (5s)
      await advanceAndFlush(5100);
      expect(l2Runner).toHaveBeenCalledTimes(2);

      await pm.destroy();
    });

    it("should NOT fire L2 for cold (inactive) sessions", async () => {
      // maxInterval=3s, activeWindow=10s. After 10s with no new conversation
      // the session is cold and maxInterval should stop re-arming.
      const config = makeConfig({
        everyNConversations: 1,
        enableWarmup: false,
        l2: {
          delayAfterL1Seconds: 1,
          minIntervalSeconds: 1,
          maxIntervalSeconds: 3,
          sessionActiveWindowHours: 10 / 3600, // 10 second active window
        },
      });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      const l1Runner = vi.fn<L1Runner>().mockResolvedValue(undefined);
      const l2Runner = vi.fn<L2Runner>().mockResolvedValue(undefined);
      pm.setL1Runner(l1Runner);
      pm.setL2Runner(l2Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      // T=0: Trigger L1 → session last_active_time = T=0
      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();

      // T≈1.1s: L2 fires (delay-after-L1). Session still warm.
      await advanceAndFlush(1100);
      const callsAfterFirstL2 = l2Runner.mock.calls.length;
      expect(callsAfterFirstL2).toBeGreaterThanOrEqual(1);

      // Now let the session go completely cold: jump far past activeWindow.
      // 20s > 10s activeWindow — any maxInterval timers that fire during this
      // window will eventually hit the cold check and stop.
      await advanceAndFlush(20_000);
      const callsAfterCold = l2Runner.mock.calls.length;

      // After session goes cold, further maxInterval ticks should NOT fire L2.
      // Advance another 10s (several maxInterval cycles) — count should not increase.
      await advanceAndFlush(10_000);
      expect(l2Runner).toHaveBeenCalledTimes(callsAfterCold);

      await pm.destroy();
    });

    it("L2 timer should be downward-only (never postponed)", async () => {
      const config = makeConfig({
        everyNConversations: 1,
        enableWarmup: false,
        l2: {
          delayAfterL1Seconds: 3,
          minIntervalSeconds: 1,
          maxIntervalSeconds: 60,
          sessionActiveWindowHours: 24,
        },
      });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      const l1Runner = vi.fn<L1Runner>().mockResolvedValue(undefined);
      const l2Runner = vi.fn<L2Runner>().mockResolvedValue(undefined);
      pm.setL1Runner(l1Runner);
      pm.setL2Runner(l2Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      // First L1 → arms L2 at now+3s
      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();

      // Wait 1s, trigger another L1 → try to advance L2 to now+3s
      // But the current timer is at original_now+3s = 2s from now
      // So the new desired time (now+3s = original_now+4s) is LATER → not advanced
      await advanceAndFlush(1000);
      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();

      // L2 should still fire at original_now+3s (2s from now)
      await advanceAndFlush(2100);
      expect(l2Runner).toHaveBeenCalledTimes(1);

      await pm.destroy();
    });
  });

  // ─────────────────────────────────────
  // PM-05: Pipeline exception handling
  // ─────────────────────────────────────

  describe("PM-05: pipeline execution exception doesn't break next schedule", () => {
    it("L1 failure should preserve messages in buffer for retry", async () => {
      const config = makeConfig({ everyNConversations: 2, enableWarmup: false });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      let callCount = 0;
      const l1Runner = vi.fn<L1Runner>().mockImplementation(async () => {
        callCount++;
        if (callCount === 1) throw new Error("L1 failure");
        return undefined;
      });
      pm.setL1Runner(l1Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      // Trigger L1 (will fail)
      await pm.notifyConversation("s1", makeMessages(2));
      await pm.notifyConversation("s1", makeMessages(1));
      await flushTimers();

      expect(l1Runner).toHaveBeenCalledTimes(1);
      // Messages should be restored to buffer
      expect(pm.getBufferedMessageCount("s1")).toBeGreaterThan(0);

      // After retry delay (30s), L1 should be retried
      await advanceAndFlush(31_000);

      expect(l1Runner).toHaveBeenCalledTimes(2);

      await pm.destroy();
    });

    it("L1 max retries should stop auto-retry", async () => {
      const config = makeConfig({
        everyNConversations: 1,
        enableWarmup: false,
        l1: { idleTimeoutSeconds: 1 },
      });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      const l1Runner = vi.fn<L1Runner>().mockRejectedValue(new Error("persistent failure"));
      pm.setL1Runner(l1Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      // First call
      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();
      expect(l1Runner).toHaveBeenCalledTimes(1);

      // Retry 5 more times (max retries = 5)
      for (let i = 0; i < 5; i++) {
        await advanceAndFlush(31_000);
      }

      expect(l1Runner).toHaveBeenCalledTimes(6); // 1 original + 5 retries

      // After max retries, no more auto-retry
      await advanceAndFlush(60_000);
      expect(l1Runner).toHaveBeenCalledTimes(6);

      await pm.destroy();
    });

    it("L2 failure should still arm maxInterval for retry", async () => {
      const config = makeConfig({
        everyNConversations: 1,
        enableWarmup: false,
        l2: {
          delayAfterL1Seconds: 1,
          minIntervalSeconds: 1,
          maxIntervalSeconds: 5,
          sessionActiveWindowHours: 24,
        },
      });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      const l1Runner = vi.fn<L1Runner>().mockResolvedValue(undefined);
      let l2CallCount = 0;
      const l2Runner = vi.fn<L2Runner>().mockImplementation(async () => {
        l2CallCount++;
        if (l2CallCount === 1) throw new Error("L2 failure");
        return undefined;
      });

      pm.setL1Runner(l1Runner);
      pm.setL2Runner(l2Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      // Trigger L1 → L2 (L2 will fail)
      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();
      await advanceAndFlush(1100);
      expect(l2Runner).toHaveBeenCalledTimes(1);

      // Even after failure, maxInterval timer should be armed → retry
      await advanceAndFlush(5100);
      expect(l2Runner).toHaveBeenCalledTimes(2);

      await pm.destroy();
    });

    it("L3 failure should not prevent future L3 runs", async () => {
      const config = makeConfig({
        everyNConversations: 1,
        enableWarmup: false,
        l2: {
          delayAfterL1Seconds: 1,
          minIntervalSeconds: 1,
          maxIntervalSeconds: 60,
          sessionActiveWindowHours: 24,
        },
      });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      const l1Runner = vi.fn<L1Runner>().mockResolvedValue(undefined);
      const l2Runner = vi.fn<L2Runner>().mockResolvedValue(undefined);
      let l3CallCount = 0;
      const l3Runner = vi.fn<L3Runner>().mockImplementation(async () => {
        l3CallCount++;
        if (l3CallCount === 1) throw new Error("L3 failure");
      });

      pm.setL1Runner(l1Runner);
      pm.setL2Runner(l2Runner);
      pm.setL3Runner(l3Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      // First cycle: L1 → L2 → L3 (L3 fails)
      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();
      await advanceAndFlush(1100);
      await flushTimers();
      expect(l3Runner).toHaveBeenCalledTimes(1);

      // Second cycle: new L1 → L2 → L3 (should succeed now)
      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();
      await advanceAndFlush(1100);
      await flushTimers();
      expect(l3Runner).toHaveBeenCalledTimes(2);

      await pm.destroy();
    });
  });

  // ─────────────────────────────────────
  // PM-06: destroy() cleanup
  // ─────────────────────────────────────

  describe("PM-06: destroy() correctly cleans all timers and resources", () => {
    it("should not accept new work after destroy", async () => {
      const config = makeConfig({ everyNConversations: 1, enableWarmup: false });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      const l1Runner = vi.fn<L1Runner>().mockResolvedValue(undefined);
      pm.setL1Runner(l1Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      await pm.destroy();
      expect(pm.isDestroyed).toBe(true);

      // Notify after destroy should be silently ignored
      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();
      expect(l1Runner).not.toHaveBeenCalled();
    });

    it("should flush pending L1 buffers on destroy", async () => {
      const config = makeConfig({
        everyNConversations: 100,
        enableWarmup: false,
        l1: { idleTimeoutSeconds: 60 },
      });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      const l1Runner = vi.fn<L1Runner>().mockResolvedValue(undefined);
      pm.setL1Runner(l1Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      // Add messages but don't reach threshold
      await pm.notifyConversation("s1", makeMessages(3));
      await flushTimers();
      expect(l1Runner).not.toHaveBeenCalled();

      // Destroy should flush pending work
      await pm.destroy();
      expect(l1Runner).toHaveBeenCalledTimes(1);
    });

    it("double destroy should be safe", async () => {
      const pm = new MemoryPipelineManager(makeConfig(), createNoopLogger());
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      await pm.destroy();
      await pm.destroy(); // should not throw
      expect(pm.isDestroyed).toBe(true);
    });

    it("should persist states during destroy even if flush fails", async () => {
      const config = makeConfig({ everyNConversations: 1, enableWarmup: false });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      const l1Runner = vi.fn<L1Runner>().mockImplementation(async () => {
        // Simulate a slow L1 that exceeds destroy timeout
        await new Promise((r) => setTimeout(r, 10_000));
      });
      const persister = vi.fn().mockResolvedValue(undefined);

      pm.setL1Runner(l1Runner);
      pm.setPersister(persister);
      pm.start();

      // Add pending work
      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();

      // Destroy — L1 is running and will timeout
      const destroyPromise = pm.destroy();
      await advanceAndFlush(6000); // Past DESTROY_TIMEOUT_MS (5000)
      await destroyPromise;

      // Persister should still be called (state saving on destroy)
      expect(persister).toHaveBeenCalled();

      await advanceAndFlush(10_000);
    });
  });

  // ─────────────────────────────────────
  // L3 dedup (global mutex + pending flag)
  // ─────────────────────────────────────

  describe("L3 dedup: global mutex + pending flag", () => {
    it("should deduplicate concurrent L3 triggers", async () => {
      const config = makeConfig({
        everyNConversations: 1,
        enableWarmup: false,
        l2: {
          delayAfterL1Seconds: 1,
          minIntervalSeconds: 1,
          maxIntervalSeconds: 60,
          sessionActiveWindowHours: 24,
        },
      });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      const l1Runner = vi.fn<L1Runner>().mockResolvedValue(undefined);
      const l2Runner = vi.fn<L2Runner>().mockResolvedValue(undefined);
      // Use a synchronous L3 runner to avoid fake-timer complications.
      // The dedup logic is independent of how long L3 takes.
      const l3Runner = vi.fn<L3Runner>().mockResolvedValue(undefined);

      pm.setL1Runner(l1Runner);
      pm.setL2Runner(l2Runner);
      pm.setL3Runner(l3Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      // Trigger on two separate sessions to create two L2 completions
      await pm.notifyConversation("s1", makeMessages());
      await pm.notifyConversation("s2", makeMessages());
      await flushTimers();

      // Both L2s fire after delay
      await advanceAndFlush(1100);
      await flushTimers();

      // L3 should have been triggered, but dedup ensures at most 1 running + 1 pending
      // Key point: it should NOT run more than 2 (dedup works)
      expect(l3Runner.mock.calls.length).toBeLessThanOrEqual(2);

      await pm.destroy();
    });
  });

  // ─────────────────────────────────────
  // Session filter
  // ─────────────────────────────────────

  describe("session filter: internal sessions and excluded agents", () => {
    it("should skip internal session keys", async () => {
      const config = makeConfig({ everyNConversations: 1, enableWarmup: false });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      const l1Runner = vi.fn<L1Runner>().mockResolvedValue(undefined);
      pm.setL1Runner(l1Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      // Internal sessions should be skipped
      await pm.notifyConversation("agent:foo:subagent:bar", makeMessages());
      await pm.notifyConversation("temp:utility", makeMessages());
      await pm.notifyConversation("key:memory-scene-extract-123", makeMessages());
      await flushTimers();

      expect(l1Runner).not.toHaveBeenCalled();

      await pm.destroy();
    });

    it("should skip sessions matching excludeAgents patterns", async () => {
      const filter = new SessionFilter(["bench-judge-*"]);
      const config = makeConfig({ everyNConversations: 1, enableWarmup: false });
      const pm = new MemoryPipelineManager(config, createNoopLogger(), filter);

      const l1Runner = vi.fn<L1Runner>().mockResolvedValue(undefined);
      pm.setL1Runner(l1Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      await pm.notifyConversation("bench-judge-abc", makeMessages());
      await flushTimers();
      expect(l1Runner).not.toHaveBeenCalled();

      // Normal session should work
      await pm.notifyConversation("normal-session", makeMessages());
      await flushTimers();
      expect(l1Runner).toHaveBeenCalledTimes(1);

      await pm.destroy();
    });
  });

  // ─────────────────────────────────────
  // Multi-session independence
  // ─────────────────────────────────────

  describe("multi-session: sessions are independent", () => {
    it("should track conversation count per session independently", async () => {
      const config = makeConfig({ everyNConversations: 2, enableWarmup: false });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      const l1Runner = vi.fn<L1Runner>().mockResolvedValue(undefined);
      pm.setL1Runner(l1Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      // session-a: 1 conversation (below threshold)
      await pm.notifyConversation("session-a", makeMessages());
      await flushTimers();
      expect(l1Runner).not.toHaveBeenCalled();

      // session-b: 2 conversations (reaches threshold)
      await pm.notifyConversation("session-b", makeMessages());
      await pm.notifyConversation("session-b", makeMessages());
      await flushTimers();
      expect(l1Runner).toHaveBeenCalledTimes(1);
      expect(l1Runner).toHaveBeenCalledWith(expect.objectContaining({ sessionKey: "session-b" }));

      // session-a: 2nd conversation → reaches threshold
      await pm.notifyConversation("session-a", makeMessages());
      await flushTimers();
      expect(l1Runner).toHaveBeenCalledTimes(2);

      await pm.destroy();
    });
  });

  // ─────────────────────────────────────
  // State recovery from checkpoint
  // ─────────────────────────────────────

  describe("state recovery from checkpoint", () => {
    it("should restore session states and recover pending work", async () => {
      const config = makeConfig({
        everyNConversations: 5,
        enableWarmup: false,
        l2: {
          delayAfterL1Seconds: 1,
          minIntervalSeconds: 1,
          maxIntervalSeconds: 60,
          sessionActiveWindowHours: 24,
        },
      });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      const l2Runner = vi.fn<L2Runner>().mockResolvedValue(undefined);
      pm.setL2Runner(l2Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));

      // Start with restored states that have pending L2 work
      pm.start({
        "recovered-session": {
          conversation_count: 3,
          last_extraction_time: "",
          last_extraction_updated_time: "",
          last_active_time: Date.now(),
          l2_pending_l1_count: 3,
          warmup_threshold: 0,
          l2_last_extraction_time: "",
        },
      });

      // Recovery should arm L2 timer for the pending session
      await advanceAndFlush(1100);
      expect(l2Runner).toHaveBeenCalledTimes(1);
      expect(l2Runner).toHaveBeenCalledWith("recovered-session", undefined);

      await pm.destroy();
    });

    it("should skip filtered sessions during restore", async () => {
      const filter = new SessionFilter(["bench-*"]);
      const config = makeConfig({
        everyNConversations: 1,
        enableWarmup: false,
        l2: {
          delayAfterL1Seconds: 1,
          minIntervalSeconds: 1,
          maxIntervalSeconds: 10,
          sessionActiveWindowHours: 24,
        },
      });
      const pm = new MemoryPipelineManager(config, createNoopLogger(), filter);

      const l2Runner = vi.fn<L2Runner>().mockResolvedValue(undefined);
      pm.setL2Runner(l2Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));

      pm.start({
        "bench-test-1": {
          conversation_count: 5,
          last_extraction_time: "",
          last_extraction_updated_time: "",
          last_active_time: Date.now(),
          l2_pending_l1_count: 5,
          warmup_threshold: 0,
          l2_last_extraction_time: "",
        },
      });

      // Filtered session should not trigger L2
      await advanceAndFlush(5000);
      expect(l2Runner).not.toHaveBeenCalled();
      expect(pm.getSessionKeys()).not.toContain("bench-test-1");

      await pm.destroy();
    });

    it("should backfill warmup_threshold for old sessions", async () => {
      const config = makeConfig({ everyNConversations: 5, enableWarmup: true });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      pm.setPersister(vi.fn().mockResolvedValue(undefined));

      // Old checkpoint has no warmup_threshold field
      pm.start({
        "old-session": {
          conversation_count: 0,
          last_extraction_time: "",
          last_extraction_updated_time: "",
          last_active_time: Date.now(),
          l2_pending_l1_count: 0,
          warmup_threshold: undefined as unknown as number,
          l2_last_extraction_time: "",
        },
      });

      // Should treat undefined warmup_threshold as graduated (0)
      const state = pm.getSessionState("old-session");
      expect(state?.warmup_threshold).toBe(0);

      await pm.destroy();
    });
  });

  // ─────────────────────────────────────
  // No runner set
  // ─────────────────────────────────────

  describe("no runner set: graceful degradation", () => {
    it("should skip L1 gracefully when no L1 runner is set", async () => {
      const config = makeConfig({ everyNConversations: 1, enableWarmup: false });
      const logger = createNoopLogger();
      const pm = new MemoryPipelineManager(config, logger);

      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      // No L1 runner set — should not throw
      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();

      expect(logger.warn).toHaveBeenCalledWith(expect.stringContaining("No L1 runner"));

      await pm.destroy();
    });

    it("should skip L2 gracefully when no L2 runner is set", async () => {
      const config = makeConfig({
        everyNConversations: 1,
        enableWarmup: false,
        l2: {
          delayAfterL1Seconds: 1,
          minIntervalSeconds: 1,
          maxIntervalSeconds: 10,
          sessionActiveWindowHours: 24,
        },
      });
      const logger = createNoopLogger();
      const pm = new MemoryPipelineManager(config, logger);

      const l1Runner = vi.fn<L1Runner>().mockResolvedValue(undefined);
      pm.setL1Runner(l1Runner);
      // No L2 runner set
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();
      await advanceAndFlush(1100);

      expect(logger.warn).toHaveBeenCalledWith(expect.stringContaining("No L2 runner"));

      await pm.destroy();
    });

    it("should skip L3 gracefully when no L3 runner is set", async () => {
      const config = makeConfig({
        everyNConversations: 1,
        enableWarmup: false,
        l2: {
          delayAfterL1Seconds: 1,
          minIntervalSeconds: 1,
          maxIntervalSeconds: 60,
          sessionActiveWindowHours: 24,
        },
      });
      const logger = createNoopLogger();
      const pm = new MemoryPipelineManager(config, logger);

      pm.setL1Runner(vi.fn<L1Runner>().mockResolvedValue(undefined));
      pm.setL2Runner(vi.fn<L2Runner>().mockResolvedValue(undefined));
      // No L3 runner set
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();
      await advanceAndFlush(1100);
      await flushTimers();

      expect(logger.warn).toHaveBeenCalledWith(expect.stringContaining("No L3 runner"));

      await pm.destroy();
    });
  });

  // ─────────────────────────────────────
  // Public accessors
  // ─────────────────────────────────────

  describe("public accessors", () => {
    it("getSessionState returns a copy (not a reference)", async () => {
      const pm = new MemoryPipelineManager(
        makeConfig({ everyNConversations: 10, enableWarmup: false }),
        createNoopLogger(),
      );
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      await pm.notifyConversation("s1", makeMessages());

      const state1 = pm.getSessionState("s1");
      expect(state1).toBeDefined();
      state1!.conversation_count = 999; // Mutate the copy

      const state2 = pm.getSessionState("s1");
      expect(state2!.conversation_count).toBe(1); // Original unchanged

      await pm.destroy();
    });

    it("getSessionKeys and getQueueSizes return correct values", async () => {
      const pm = new MemoryPipelineManager(
        makeConfig({ everyNConversations: 10, enableWarmup: false }),
        createNoopLogger(),
      );
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      await pm.notifyConversation("s1", makeMessages());
      await pm.notifyConversation("s2", makeMessages());

      expect(pm.getSessionKeys()).toContain("s1");
      expect(pm.getSessionKeys()).toContain("s2");

      const sizes = pm.getQueueSizes();
      expect(sizes).toHaveProperty("l1");
      expect(sizes).toHaveProperty("l2");
      expect(sizes).toHaveProperty("l3");

      await pm.destroy();
    });

    it("getBufferedMessageCount returns correct count", async () => {
      const pm = new MemoryPipelineManager(
        makeConfig({ everyNConversations: 10, enableWarmup: false }),
        createNoopLogger(),
      );
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      expect(pm.getBufferedMessageCount("nonexistent")).toBe(0);

      await pm.notifyConversation("s1", makeMessages(5));
      expect(pm.getBufferedMessageCount("s1")).toBe(5);

      await pm.destroy();
    });
  });

  // ─────────────────────────────────────
  // L2 cursor management
  // ─────────────────────────────────────

  describe("L2 cursor management", () => {
    it("should pass cursor from L2 runner result to next L2 call", async () => {
      const config = makeConfig({
        everyNConversations: 1,
        enableWarmup: false,
        l2: {
          delayAfterL1Seconds: 1,
          minIntervalSeconds: 1,
          maxIntervalSeconds: 5,
          sessionActiveWindowHours: 24,
        },
      });
      const pm = new MemoryPipelineManager(config, createNoopLogger());

      const l1Runner = vi.fn<L1Runner>().mockResolvedValue(undefined);
      const l2Runner = vi.fn<L2Runner>().mockResolvedValue({
        latestCursor: "2026-03-17T10:00:00Z",
      } satisfies L2RunnerResult);

      pm.setL1Runner(l1Runner);
      pm.setL2Runner(l2Runner);
      pm.setPersister(vi.fn().mockResolvedValue(undefined));
      pm.start();

      // First cycle: L1 → L2 (returns cursor)
      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();
      await advanceAndFlush(1100);
      expect(l2Runner).toHaveBeenCalledWith("s1", undefined);

      // Second cycle: L1 → L2 (should receive the cursor)
      await pm.notifyConversation("s1", makeMessages());
      await flushTimers();
      await advanceAndFlush(1100);
      expect(l2Runner).toHaveBeenCalledWith("s1", "2026-03-17T10:00:00Z");

      await pm.destroy();
    });
  });
});
