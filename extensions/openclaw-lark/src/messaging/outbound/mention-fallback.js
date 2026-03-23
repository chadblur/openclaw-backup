"use strict";
// SPDX-License-Identifier: MIT
/**
 * Outbound @mention fallback.
 *
 * Feishu/Lark clients do NOT treat plain text `@ou_xxx` as a mention.
 * The correct syntaxes are:
 * - Post/Text (md): `<at user_id="ou_xxx"></at>` (name optional)
 * - Interactive Card markdown: `<at id=ou_xxx></at>`
 *
 * This module provides conservative conversion of `@ou_...` into proper
 * mention tags while avoiding common false positives (URLs/emails).
 */

/** @typedef {'text'|'card'} MentionMode */

const DEFAULT_ENABLED = true;

/**
 * Decide whether mention fallback is enabled.
 * Config key: channels.feishu.mentionFallback (boolean)
 * - undefined => enabled (default)
 * - true/false => explicit
 */
export function isMentionFallbackEnabled(larkClient) {
  // larkClient is a LarkClient instance; its account.config is merged account config.
  const v = larkClient?.account?.config?.mentionFallback;
  if (typeof v === 'boolean') return v;
  return DEFAULT_ENABLED;
}

function isBoundaryChar(ch) {
  if (!ch) return true;
  // whitespace
  if (/\s/.test(ch)) return true;
  // common punctuation (ASCII + CJK)
  return /[\.,;:!\?\)\]\}>"'，。；：！？”’、】【】》）]/.test(ch);
}

function isAllowedPrecedingChar(ch) {
  if (!ch) return true;
  if (/\s/.test(ch)) return true;
  // conservative: allow only opening punctuation to avoid URLs (/,=,? are NOT allowed)
  return /[\(\[\{<"'“‘（【《]/.test(ch);
}

/**
 * Convert `@ou_...` to Feishu mention tags.
 *
 * Notes:
 * - Conservative boundary rules: only converts when `@` is at start or preceded
 *   by whitespace / opening punctuation.
 * - Requires a boundary after the id.
 * - Does NOT try to resolve user display names.
 */
export function convertOpenIdMentions(text, mode) {
  if (!text || typeof text !== 'string') return text;

  /** @type {MentionMode} */
  const m = mode;

  // Quick check to avoid extra work.
  if (!text.includes('@ou_')) return text;

  const re = /@ou_[0-9a-zA-Z]+/g;
  return text.replace(re, (match, offset) => {
    // Do not touch existing XML/HTML-like tags, especially `<at ...>`.
    // Prevents corrupting attributes like `<at user_id="@ou_xxx"></at>`.
    const lt = text.lastIndexOf('<', offset);
    const gt = text.lastIndexOf('>', offset);
    if (lt > gt) return match; // inside a tag

    const before = offset > 0 ? text[offset - 1] : '';
    const after = offset + match.length < text.length ? text[offset + match.length] : '';

    if (!isAllowedPrecedingChar(before)) return match;
    if (!isBoundaryChar(after)) return match;

    const openId = match.slice(1); // drop leading '@'

    if (m === 'card') {
      return `<at id=${openId}></at>`;
    }
    // text/post
    return `<at user_id="${openId}"></at>`;
  });
}

const CARD_SKIP_KEYS = new Set([
  'url',
  'href',
  'src',
  'img_key',
  'image_key',
  'template_id',
  'file_key',
  'icon',
  'token',
]);

function looksLikeUrlOrEmail(s) {
  const t = s.trim();
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(t)) return true; // scheme:
  if (t.includes('://')) return true;
  // simple email heuristic
  if (/\b\S+@\S+\.[A-Za-z]{2,}\b/.test(t)) return true;
  return false;
}

/**
 * Deeply apply mention fallback conversion to a card JSON object.
 *
 * We only mutate string leaf nodes; keys that are typically URL/media identifiers
 * are skipped.
 */
export function convertOpenIdMentionsInCard(card) {
  const walk = (v, parentKey) => {
    if (typeof v === 'string') {
      if (parentKey && CARD_SKIP_KEYS.has(parentKey)) return v;
      if (looksLikeUrlOrEmail(v)) return v;
      return convertOpenIdMentions(v, 'card');
    }
    if (Array.isArray(v)) {
      return v.map((x) => walk(x, parentKey));
    }
    if (v && typeof v === 'object') {
      const out = Array.isArray(v) ? [] : {};
      for (const [k, val] of Object.entries(v)) {
        out[k] = walk(val, k);
      }
      return out;
    }
    return v;
  };

  return walk(card, undefined);
}
