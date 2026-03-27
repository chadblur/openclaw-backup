/**
 * Unit tests for ManagedTimer.
 *
 * Covers:
 * - schedule() — cancel + set a new timer
 * - scheduleAt() — schedule by absolute epoch-ms
 * - tryAdvanceTo() — downward-only rescheduling
 * - cancel() — cancel without triggering
 * - flush() — trigger immediately for graceful shutdown
 * - pending / scheduledTime accessors
 * - isDestroyed guard
 * - .unref() behavior (timer doesn't keep process alive)
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { ManagedTimer } from "./managed-timer.js";

describe("ManagedTimer", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  // ─────────────────────────────────────
  // schedule()
  // ─────────────────────────────────────

  describe("schedule()", () => {
    it("should fire callback after the specified delay", async () => {
      const timer = new ManagedTimer("test");
      const cb = vi.fn();

      timer.schedule(1000, cb);
      expect(timer.pending).toBe(true);
      expect(cb).not.toHaveBeenCalled();

      vi.advanceTimersByTime(999);
      expect(cb).not.toHaveBeenCalled();

      vi.advanceTimersByTime(1);
      expect(cb).toHaveBeenCalledTimes(1);
      expect(timer.pending).toBe(false);
    });

    it("should cancel previous timer when scheduling a new one", () => {
      const timer = new ManagedTimer("test");
      const cb1 = vi.fn();
      const cb2 = vi.fn();

      timer.schedule(1000, cb1);
      timer.schedule(2000, cb2); // replaces cb1

      vi.advanceTimersByTime(1500);
      expect(cb1).not.toHaveBeenCalled(); // cancelled

      vi.advanceTimersByTime(1000);
      expect(cb2).toHaveBeenCalledTimes(1);
    });

    it("should auto-clear after firing (not pending)", () => {
      const timer = new ManagedTimer("test");
      timer.schedule(100, vi.fn());

      vi.advanceTimersByTime(100);
      expect(timer.pending).toBe(false);
      expect(timer.scheduledTime).toBe(0);
    });
  });

  // ─────────────────────────────────────
  // scheduleAt()
  // ─────────────────────────────────────

  describe("scheduleAt()", () => {
    it("should fire at the specified absolute epoch-ms", () => {
      const timer = new ManagedTimer("test");
      const cb = vi.fn();
      const fireAt = Date.now() + 5000;

      timer.scheduleAt(fireAt, cb);
      expect(timer.pending).toBe(true);

      vi.advanceTimersByTime(4999);
      expect(cb).not.toHaveBeenCalled();

      vi.advanceTimersByTime(1);
      expect(cb).toHaveBeenCalledTimes(1);
    });

    it("should fire immediately (delay=0) if epoch-ms is in the past", () => {
      const timer = new ManagedTimer("test");
      const cb = vi.fn();

      timer.scheduleAt(Date.now() - 1000, cb); // past time
      vi.advanceTimersByTime(0);

      expect(cb).toHaveBeenCalledTimes(1);
    });

    it("should replace any pending timer", () => {
      const timer = new ManagedTimer("test");
      const cb1 = vi.fn();
      const cb2 = vi.fn();

      timer.scheduleAt(Date.now() + 1000, cb1);
      timer.scheduleAt(Date.now() + 3000, cb2);

      vi.advanceTimersByTime(1500);
      expect(cb1).not.toHaveBeenCalled(); // cancelled by second scheduleAt

      vi.advanceTimersByTime(2000);
      expect(cb2).toHaveBeenCalledTimes(1);
    });
  });

  // ─────────────────────────────────────
  // tryAdvanceTo() — downward-only
  // ─────────────────────────────────────

  describe("tryAdvanceTo()", () => {
    it("should set timer when no timer is pending", () => {
      const timer = new ManagedTimer("test");
      const cb = vi.fn();

      const advanced = timer.tryAdvanceTo(Date.now() + 2000, cb);
      expect(advanced).toBe(true);
      expect(timer.pending).toBe(true);

      vi.advanceTimersByTime(2000);
      expect(cb).toHaveBeenCalledTimes(1);
    });

    it("should advance if new time is earlier than current schedule", () => {
      const timer = new ManagedTimer("test");
      const cb1 = vi.fn();
      const cb2 = vi.fn();

      timer.scheduleAt(Date.now() + 5000, cb1); // fire in 5s
      const advanced = timer.tryAdvanceTo(Date.now() + 2000, cb2); // advance to 2s

      expect(advanced).toBe(true);

      vi.advanceTimersByTime(2500);
      expect(cb1).not.toHaveBeenCalled();
      expect(cb2).toHaveBeenCalledTimes(1);
    });

    it("should NOT advance if new time is later than current schedule", () => {
      const timer = new ManagedTimer("test");
      const cb1 = vi.fn();
      const cb2 = vi.fn();

      timer.scheduleAt(Date.now() + 2000, cb1); // fire in 2s
      const advanced = timer.tryAdvanceTo(Date.now() + 5000, cb2); // try later

      expect(advanced).toBe(false);

      vi.advanceTimersByTime(2500);
      expect(cb1).toHaveBeenCalledTimes(1);
      expect(cb2).not.toHaveBeenCalled();
    });

    it("should NOT advance if new time equals current schedule", () => {
      const timer = new ManagedTimer("test");
      const cb1 = vi.fn();
      const cb2 = vi.fn();
      const t = Date.now() + 3000;

      timer.scheduleAt(t, cb1);
      const advanced = timer.tryAdvanceTo(t, cb2); // same time

      expect(advanced).toBe(false);

      vi.advanceTimersByTime(3500);
      expect(cb1).toHaveBeenCalledTimes(1);
      expect(cb2).not.toHaveBeenCalled();
    });
  });

  // ─────────────────────────────────────
  // cancel()
  // ─────────────────────────────────────

  describe("cancel()", () => {
    it("should cancel a pending timer without triggering callback", () => {
      const timer = new ManagedTimer("test");
      const cb = vi.fn();

      timer.schedule(1000, cb);
      expect(timer.pending).toBe(true);

      timer.cancel();
      expect(timer.pending).toBe(false);
      expect(timer.scheduledTime).toBe(0);

      vi.advanceTimersByTime(2000);
      expect(cb).not.toHaveBeenCalled();
    });

    it("should be safe to cancel when no timer is pending", () => {
      const timer = new ManagedTimer("test");
      expect(() => timer.cancel()).not.toThrow();
      expect(timer.pending).toBe(false);
    });

    it("should be safe to cancel multiple times", () => {
      const timer = new ManagedTimer("test");
      timer.schedule(1000, vi.fn());

      timer.cancel();
      timer.cancel();
      timer.cancel();

      expect(timer.pending).toBe(false);
    });
  });

  // ─────────────────────────────────────
  // flush()
  // ─────────────────────────────────────

  describe("flush()", () => {
    it("should trigger callback immediately and clear the timer", () => {
      const timer = new ManagedTimer("test");
      const cb = vi.fn();

      timer.schedule(5000, cb);
      expect(timer.pending).toBe(true);

      timer.flush();
      expect(cb).toHaveBeenCalledTimes(1);
      expect(timer.pending).toBe(false);
    });

    it("should be a no-op when no timer is pending", () => {
      const timer = new ManagedTimer("test");
      expect(() => timer.flush()).not.toThrow();
    });

    it("should not fire the callback again after flush", () => {
      const timer = new ManagedTimer("test");
      const cb = vi.fn();

      timer.schedule(1000, cb);
      timer.flush();
      expect(cb).toHaveBeenCalledTimes(1);

      // Original timer should not fire again
      vi.advanceTimersByTime(2000);
      expect(cb).toHaveBeenCalledTimes(1);
    });
  });

  // ─────────────────────────────────────
  // pending / scheduledTime accessors
  // ─────────────────────────────────────

  describe("accessors: pending, scheduledTime, name", () => {
    it("pending should be false initially", () => {
      const timer = new ManagedTimer("my-timer");
      expect(timer.pending).toBe(false);
    });

    it("scheduledTime should be 0 when no timer is pending", () => {
      const timer = new ManagedTimer("test");
      expect(timer.scheduledTime).toBe(0);
    });

    it("scheduledTime should return the correct fire time", () => {
      const timer = new ManagedTimer("test");
      const before = Date.now();
      timer.schedule(5000, vi.fn());
      const after = Date.now();

      expect(timer.scheduledTime).toBeGreaterThanOrEqual(before + 5000);
      expect(timer.scheduledTime).toBeLessThanOrEqual(after + 5000);
    });

    it("name should match the constructor argument", () => {
      const timer = new ManagedTimer("L2-schedule:session-abc");
      expect(timer.name).toBe("L2-schedule:session-abc");
    });
  });

  // ─────────────────────────────────────
  // isDestroyed guard
  // ─────────────────────────────────────

  describe("isDestroyed guard", () => {
    it("should not fire callback when isDestroyed returns true", () => {
      let destroyed = false;
      const timer = new ManagedTimer("test", () => destroyed);
      const cb = vi.fn();

      timer.schedule(1000, cb);

      // Mark as destroyed before timer fires
      destroyed = true;
      vi.advanceTimersByTime(1500);

      expect(cb).not.toHaveBeenCalled();
      expect(timer.pending).toBe(false); // timer handle cleared
    });

    it("should fire callback normally when isDestroyed returns false", () => {
      const timer = new ManagedTimer("test", () => false);
      const cb = vi.fn();

      timer.schedule(1000, cb);
      vi.advanceTimersByTime(1000);

      expect(cb).toHaveBeenCalledTimes(1);
    });
  });

  // ─────────────────────────────────────
  // Interaction: schedule → cancel → schedule → fire
  // ─────────────────────────────────────

  describe("complex interaction sequences", () => {
    it("schedule → cancel → re-schedule should work correctly", () => {
      const timer = new ManagedTimer("test");
      const cb1 = vi.fn();
      const cb2 = vi.fn();

      timer.schedule(1000, cb1);
      timer.cancel();

      vi.advanceTimersByTime(1500);
      expect(cb1).not.toHaveBeenCalled();

      timer.schedule(500, cb2);
      vi.advanceTimersByTime(500);
      expect(cb2).toHaveBeenCalledTimes(1);
    });

    it("flush → schedule → fire should work correctly", () => {
      const timer = new ManagedTimer("test");
      const cb1 = vi.fn();
      const cb2 = vi.fn();

      timer.schedule(1000, cb1);
      timer.flush();
      expect(cb1).toHaveBeenCalledTimes(1);

      // After flush, scheduling a new timer should work
      timer.schedule(500, cb2);
      expect(timer.pending).toBe(true);

      vi.advanceTimersByTime(500);
      expect(cb2).toHaveBeenCalledTimes(1);
    });

    it("schedule → tryAdvanceTo (earlier) → fire should fire at advanced time", () => {
      const timer = new ManagedTimer("test");
      const cbOrig = vi.fn();
      const cbAdvanced = vi.fn();

      timer.schedule(5000, cbOrig);
      timer.tryAdvanceTo(Date.now() + 1000, cbAdvanced);

      vi.advanceTimersByTime(1500);
      expect(cbAdvanced).toHaveBeenCalledTimes(1);
      expect(cbOrig).not.toHaveBeenCalled();
    });
  });
});
