import { describe, it, expect, vi } from "vitest";
import { SerialQueue } from "./serial-queue.js";

// Helper: create a delayed task that records execution order
function delayTask(ms: number, log: string[], label: string) {
  return () =>
    new Promise<string>((resolve) => {
      setTimeout(() => {
        log.push(label);
        resolve(label);
      }, ms);
    });
}

// ────────────────────────────────────────
// Basic serial execution
// ────────────────────────────────────────
describe("serial execution", () => {
  it("should execute tasks in FIFO order", async () => {
    const q = new SerialQueue("test");
    const log: string[] = [];

    q.add(delayTask(30, log, "A"));
    q.add(delayTask(10, log, "B"));
    q.add(delayTask(10, log, "C"));

    await q.onIdle();
    expect(log).toEqual(["A", "B", "C"]);
  });

  it("should return the task result via add()", async () => {
    const q = new SerialQueue("test");
    const result = await q.add(async () => 42);
    expect(result).toBe(42);
  });

  it("should reject the add() promise when a task throws", async () => {
    const q = new SerialQueue("test");
    await expect(
      q.add(async () => {
        throw new Error("boom");
      }),
    ).rejects.toThrow("boom");
  });

  it("should continue executing subsequent tasks after one fails", async () => {
    const q = new SerialQueue("test");
    const log: string[] = [];

    const p1 = q.add(async () => {
      throw new Error("fail");
    });
    const p2 = q.add(async () => {
      log.push("ok");
      return "done";
    });

    await expect(p1).rejects.toThrow("fail");
    await expect(p2).resolves.toBe("done");
    expect(log).toEqual(["ok"]);
  });
});

// ────────────────────────────────────────
// size / pending
// ────────────────────────────────────────
describe("size and pending", () => {
  it("should report correct size and pending state", async () => {
    const q = new SerialQueue("test");
    expect(q.size).toBe(0);
    expect(q.pending).toBe(false);

    let resolveFirst!: () => void;
    const blocker = new Promise<void>((r) => {
      resolveFirst = r;
    });

    q.add(() => blocker);
    q.add(async () => {});
    q.add(async () => {});

    // First task is running, 2 are queued
    await new Promise((r) => setTimeout(r, 10));
    expect(q.pending).toBe(true);
    expect(q.size).toBe(2);

    resolveFirst();
    await q.onIdle();
    expect(q.size).toBe(0);
    expect(q.pending).toBe(false);
  });
});

// ────────────────────────────────────────
// pause / start
// ────────────────────────────────────────
describe("pause and start", () => {
  it("should not start new tasks while paused", async () => {
    const q = new SerialQueue("test");
    const log: string[] = [];

    q.pause();
    q.add(async () => {
      log.push("A");
    });

    await new Promise((r) => setTimeout(r, 50));
    expect(log).toEqual([]); // nothing executed

    q.start();
    await q.onIdle();
    expect(log).toEqual(["A"]);
  });

  it("should let a running task finish after pause", async () => {
    const q = new SerialQueue("test");
    const log: string[] = [];

    q.add(async () => {
      log.push("running");
      await new Promise((r) => setTimeout(r, 30));
      log.push("done");
    });
    q.add(async () => {
      log.push("second");
    });

    // Pause immediately — first task already started
    await new Promise((r) => setTimeout(r, 5));
    q.pause();

    await new Promise((r) => setTimeout(r, 50));
    // First task should finish; second should NOT have started
    expect(log).toEqual(["running", "done"]);

    q.start();
    await q.onIdle();
    expect(log).toEqual(["running", "done", "second"]);
  });
});

// ────────────────────────────────────────
// onIdle
// ────────────────────────────────────────
describe("onIdle", () => {
  it("should resolve immediately when queue is empty and not running", async () => {
    const q = new SerialQueue("test");
    await q.onIdle(); // should not hang
  });

  it("should resolve after all tasks complete", async () => {
    const q = new SerialQueue("test");
    let counter = 0;

    q.add(async () => {
      counter++;
    });
    q.add(async () => {
      counter++;
    });
    q.add(async () => {
      counter++;
    });

    await q.onIdle();
    expect(counter).toBe(3);
  });
});

// ────────────────────────────────────────
// clear
// ────────────────────────────────────────
describe("clear", () => {
  it("should reject all pending tasks and empty the queue", async () => {
    const q = new SerialQueue("test");

    let resolveFirst!: () => void;
    const blocker = new Promise<void>((r) => {
      resolveFirst = r;
    });

    const p1 = q.add(() => blocker);
    const p2 = q.add(async () => "should-not-run");
    const p3 = q.add(async () => "should-not-run-either");

    await new Promise((r) => setTimeout(r, 10));
    q.clear();
    expect(q.size).toBe(0);

    // Pending tasks should be rejected
    await expect(p2).rejects.toThrow("Queue cleared");
    await expect(p3).rejects.toThrow("Queue cleared");

    // First task (already running) should still complete
    resolveFirst();
    await expect(p1).resolves.toBeUndefined();
  });
});

// ────────────────────────────────────────
// debug logger
// ────────────────────────────────────────
describe("debug logger", () => {
  it("should call debugFn on enqueue/dequeue/complete", async () => {
    const q = new SerialQueue("myq");
    const logs: string[] = [];
    q.setDebugLogger((msg) => logs.push(msg));

    await q.add(async () => "x");
    await q.onIdle();

    expect(logs.some((m) => m.includes("enqueued"))).toBe(true);
    expect(logs.some((m) => m.includes("dequeued"))).toBe(true);
    expect(logs.some((m) => m.includes("completed"))).toBe(true);
    expect(logs.every((m) => m.includes("[queue:myq]"))).toBe(true);
  });
});

// ────────────────────────────────────────
// name
// ────────────────────────────────────────
describe("constructor", () => {
  it("should default name to 'unnamed'", () => {
    const q = new SerialQueue();
    expect(q.name).toBe("unnamed");
  });

  it("should accept a custom name", () => {
    const q = new SerialQueue("my-queue");
    expect(q.name).toBe("my-queue");
  });
});
