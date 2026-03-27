/**
 * Unit tests for text-utils.ts.
 *
 * TU-01: extractWords — Latin words extraction
 * TU-02: extractWords — CJK character extraction
 * TU-03: extractWords — CJK 2-gram generation
 * TU-04: extractWords — mixed CJK and Latin
 * TU-05: extractWords — empty / whitespace input
 * TU-06: extractWords — single character Latin (below threshold)
 * TU-07: extractWords — numbers and alphanumeric
 * TU-08: extractWords — symbols / punctuation only
 * TU-09: extractWords — deduplication
 * TU-10: extractWords — Japanese Hiragana / Katakana
 * TU-11: extractWords — Korean Hangul
 */
import { describe, it, expect } from "vitest";
import { extractWords } from "./text-utils.js";

describe("extractWords", () => {
  // TU-01
  describe("TU-01: Latin words extraction", () => {
    it("should extract lowercase Latin words of 2+ chars", () => {
      const result = extractWords("Hello World");
      expect(result.has("hello")).toBe(true);
      expect(result.has("world")).toBe(true);
    });

    it("should convert to lowercase", () => {
      const result = extractWords("TypeScript React");
      expect(result.has("typescript")).toBe(true);
      expect(result.has("react")).toBe(true);
      // Original case should NOT be present
      expect(result.has("TypeScript")).toBe(false);
    });

    it("should extract words from sentences with punctuation", () => {
      const result = extractWords("hello, world! how are you?");
      expect(result.has("hello")).toBe(true);
      expect(result.has("world")).toBe(true);
      expect(result.has("how")).toBe(true);
      expect(result.has("are")).toBe(true);
      expect(result.has("you")).toBe(true);
    });
  });

  // TU-02
  describe("TU-02: CJK character extraction", () => {
    it("should extract individual CJK characters", () => {
      const result = extractWords("你好世界");
      expect(result.has("你")).toBe(true);
      expect(result.has("好")).toBe(true);
      expect(result.has("世")).toBe(true);
      expect(result.has("界")).toBe(true);
    });
  });

  // TU-03
  describe("TU-03: CJK 2-gram generation", () => {
    it("should generate 2-grams from adjacent CJK chars", () => {
      const result = extractWords("你好世界");
      // 2-grams: 你好, 好世, 世界
      expect(result.has("你好")).toBe(true);
      expect(result.has("好世")).toBe(true);
      expect(result.has("世界")).toBe(true);
    });

    it("should NOT generate 2-gram for single CJK char", () => {
      const result = extractWords("你");
      expect(result.has("你")).toBe(true);
      expect(result.size).toBe(1); // only the single char
    });

    it("should generate exactly one 2-gram for two CJK chars", () => {
      const result = extractWords("你好");
      expect(result.has("你")).toBe(true);
      expect(result.has("好")).toBe(true);
      expect(result.has("你好")).toBe(true);
      expect(result.size).toBe(3); // 2 chars + 1 bigram
    });
  });

  // TU-04
  describe("TU-04: mixed CJK and Latin", () => {
    it("should extract both CJK and Latin words", () => {
      const result = extractWords("用户喜欢TypeScript编程");
      // Latin
      expect(result.has("typescript")).toBe(true);
      // CJK chars
      expect(result.has("用")).toBe(true);
      expect(result.has("户")).toBe(true);
      expect(result.has("喜")).toBe(true);
      expect(result.has("欢")).toBe(true);
      expect(result.has("编")).toBe(true);
      expect(result.has("程")).toBe(true);
      // CJK 2-grams
      expect(result.has("用户")).toBe(true);
      expect(result.has("喜欢")).toBe(true);
      expect(result.has("编程")).toBe(true);
    });
  });

  // TU-05
  describe("TU-05: empty / whitespace input", () => {
    it("should return empty set for empty string", () => {
      expect(extractWords("").size).toBe(0);
    });

    it("should return empty set for whitespace only", () => {
      expect(extractWords("   \n\t  ").size).toBe(0);
    });
  });

  // TU-06
  describe("TU-06: single character Latin (below threshold)", () => {
    it("should NOT extract single-char Latin words", () => {
      const result = extractWords("I a");
      // "I" and "a" are 1-char, below the 2-char threshold
      expect(result.has("i")).toBe(false);
      expect(result.has("a")).toBe(false);
      expect(result.size).toBe(0);
    });
  });

  // TU-07
  describe("TU-07: numbers and alphanumeric", () => {
    it("should extract numbers as part of words", () => {
      const result = extractWords("python3 es2024 v8");
      expect(result.has("python3")).toBe(true);
      expect(result.has("es2024")).toBe(true);
      expect(result.has("v8")).toBe(true);
    });

    it("should extract pure numeric strings of 2+ digits", () => {
      const result = extractWords("version 42 is out");
      expect(result.has("version")).toBe(true);
      expect(result.has("42")).toBe(true);
      expect(result.has("is")).toBe(true);
      expect(result.has("out")).toBe(true);
    });
  });

  // TU-08
  describe("TU-08: symbols / punctuation only", () => {
    it("should return empty set for symbols only", () => {
      const result = extractWords("!@#$%^&*()");
      expect(result.size).toBe(0);
    });

    it("should return empty set for emoji only", () => {
      const result = extractWords("🎉🚀✨");
      expect(result.size).toBe(0);
    });
  });

  // TU-09
  describe("TU-09: deduplication", () => {
    it("should not duplicate words that appear multiple times", () => {
      const result = extractWords("hello hello hello");
      expect(result.has("hello")).toBe(true);
      // Count occurrences in set — should be exactly 1
      let count = 0;
      for (const w of result) {
        if (w === "hello") count++;
      }
      expect(count).toBe(1);
    });
  });

  // TU-10
  describe("TU-10: Japanese Hiragana / Katakana", () => {
    it("should extract Hiragana characters", () => {
      const result = extractWords("こんにちは");
      expect(result.has("こ")).toBe(true);
      expect(result.has("ん")).toBe(true);
      expect(result.has("に")).toBe(true);
      expect(result.has("ち")).toBe(true);
      expect(result.has("は")).toBe(true);
      // 2-grams
      expect(result.has("こん")).toBe(true);
    });

    it("should extract Katakana characters", () => {
      const result = extractWords("カタカナ");
      expect(result.has("カ")).toBe(true);
      expect(result.has("タ")).toBe(true);
      expect(result.has("ナ")).toBe(true);
      // 2-grams
      expect(result.has("カタ")).toBe(true);
      expect(result.has("タカ")).toBe(true);
      expect(result.has("カナ")).toBe(true);
    });
  });

  // TU-11
  describe("TU-11: Korean Hangul", () => {
    it("should extract Korean characters", () => {
      const result = extractWords("안녕하세요");
      expect(result.has("안")).toBe(true);
      expect(result.has("녕")).toBe(true);
      expect(result.has("하")).toBe(true);
      expect(result.has("세")).toBe(true);
      expect(result.has("요")).toBe(true);
      // 2-grams
      expect(result.has("안녕")).toBe(true);
      expect(result.has("하세")).toBe(true);
    });
  });

  // Additional: return type verification
  describe("Return type", () => {
    it("should return a Set<string>", () => {
      const result = extractWords("hello world");
      expect(result).toBeInstanceOf(Set);
    });
  });
});
