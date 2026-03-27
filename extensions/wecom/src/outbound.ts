import type { ChannelOutboundAdapter, ChannelOutboundContext } from "openclaw/plugin-sdk";

import { sendText as sendAgentText, sendMedia as sendAgentMedia, uploadMedia } from "./agent/api-client.js";
import { resolveWecomAccount, resolveWecomAccountConflict, resolveWecomAccounts } from "./config/index.js";
import { getWecomRuntime } from "./runtime.js";
import { getWsClient, waitForWsConnection } from "./ws-adapter.js";
import { uploadAndSendMediaBuffer } from "./media/index.js";

import { resolveWecomTarget } from "./target.js";

// ─── MIME 类型映射表（扩展名 → Content-Type）──────────────────────────

const MIME_BY_EXT: Record<string, string> = {
  jpg: "image/jpeg", jpeg: "image/jpeg", png: "image/png", gif: "image/gif",
  webp: "image/webp", bmp: "image/bmp", mp3: "audio/mpeg", wav: "audio/wav",
  amr: "audio/amr", mp4: "video/mp4", mov: "video/quicktime",
  pdf: "application/pdf", doc: "application/msword",
  docx: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  xls: "application/vnd.ms-excel", xlsx: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  ppt: "application/vnd.ms-powerpoint", pptx: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  txt: "text/plain", csv: "text/csv", tsv: "text/tab-separated-values", md: "text/markdown", json: "application/json",
  xml: "application/xml", yaml: "application/yaml", yml: "application/yaml",
  zip: "application/zip", rar: "application/vnd.rar", "7z": "application/x-7z-compressed",
  tar: "application/x-tar", gz: "application/gzip", tgz: "application/gzip",
  rtf: "application/rtf", odt: "application/vnd.oasis.opendocument.text",
};

// ─── 共享的媒体加载逻辑 ────────────────────────────────────────────

/**
 * 从 URL 或本地文件路径加载媒体文件，返回 buffer、contentType、filename。
 * 供 Bot WS 和 Agent 两种发送模式共用。
 */
async function loadMediaBuffer(mediaUrl: string): Promise<{
  buffer: Buffer;
  contentType: string;
  filename: string;
}> {
  const isRemoteUrl = /^https?:\/\//i.test(mediaUrl);

  if (isRemoteUrl) {
    const res = await fetch(mediaUrl, { signal: AbortSignal.timeout(30_000) });
    if (!res.ok) {
      throw new Error(`Failed to download media: ${res.status}`);
    }
    const buffer = Buffer.from(await res.arrayBuffer());
    const contentType = res.headers.get("content-type") || "application/octet-stream";
    const urlPath = new URL(mediaUrl).pathname;
    const filename = urlPath.split("/").pop() || "media";
    return { buffer, contentType, filename };
  }

  // 本地文件路径
  const fs = await import("node:fs/promises");
  const path = await import("node:path");
  const buffer = await fs.readFile(mediaUrl);
  const filename = path.basename(mediaUrl);
  const ext = path.extname(mediaUrl).slice(1).toLowerCase();
  const contentType = MIME_BY_EXT[ext] || "application/octet-stream";
  console.log(`[wecom-outbound] Reading local file: ${mediaUrl}, ext=${ext}, contentType=${contentType}`);
  return { buffer, contentType, filename };
}

function resolveAgentConfigOrThrow(params: {
  cfg: ChannelOutboundContext["cfg"];
  accountId?: string | null;
}) {
  const resolvedAccounts = resolveWecomAccounts(params.cfg);
  const conflictAccountId = params.accountId?.trim() || resolvedAccounts.defaultAccountId;
  const conflict = resolveWecomAccountConflict({
    cfg: params.cfg,
    accountId: conflictAccountId,
  });
  if (conflict) {
    throw new Error(conflict.message);
  }

  const requestedAccountId = params.accountId?.trim();
  if (requestedAccountId) {
    if (!resolvedAccounts.accounts[requestedAccountId]) {
      throw new Error(
        `WeCom outbound account "${requestedAccountId}" not found. Configure channels.wecom.accounts.${requestedAccountId} or use an existing accountId.`,
      );
    }
  }
  const account = resolveWecomAccount({
    cfg: params.cfg,
    accountId: params.accountId,
  }).agent;
  if (!account?.configured) {
    throw new Error(
      `WeCom outbound requires Agent mode for account=${params.accountId ?? "default"}. Configure channels.wecom.accounts.<accountId>.agent (or legacy channels.wecom.agent).`,
    );
  }
  if (typeof account.agentId !== "number" || !Number.isFinite(account.agentId)) {
    throw new Error(
      `WeCom outbound requires channels.wecom.accounts.<accountId>.agent.agentId (or legacy channels.wecom.agent.agentId) for account=${params.accountId ?? account.accountId}.`,
    );
  }
  // 注意：不要在日志里输出 corpSecret 等敏感信息
  console.log(`[wecom-outbound] Using agent config: accountId=${account.accountId}, corpId=${account.corpId}, agentId=${account.agentId}`);
  return account;
}

export const wecomOutbound: ChannelOutboundAdapter = {
  deliveryMode: "direct",
  chunkerMode: "text",
  textChunkLimit: 20480,
  chunker: (text, limit) => {
    try {
      return getWecomRuntime().channel.text.chunkText(text, limit);
    } catch {
      return [text];
    }
  },
  sendText: async ({ cfg, to, text, accountId }: ChannelOutboundContext) => {
    // signal removed - not supported in current SDK

    // ── Bot WebSocket outbound 独立路径 ──
    // WS Bot 完全独立收发，不与 Agent 组成双模，失败时直接抛错而非 fallthrough
    const resolvedAccount = resolveWecomAccount({ cfg, accountId });
    const botAccount = resolvedAccount.bot;
    if (botAccount?.connectionMode === 'websocket' && botAccount.configured) {
      const wsClient = getWsClient(botAccount.accountId);
      const wsTarget = resolveWecomTarget(to);
      const chatid = wsTarget?.touser || wsTarget?.chatid;

      // 如果目标是 Agent 会话（wecom-agent:），跳过 WS Bot，走 Agent outbound
      const rawTo = typeof to === "string" ? to.trim().toLowerCase() : "";
      if (!rawTo.startsWith("wecom-agent:")) {
        if (!wsClient?.isConnected) {
          console.log(`[wecom-outbound] Bot WS 未连接，等待重连... (accountId=${botAccount.accountId})`);
          const reconnected = await waitForWsConnection(botAccount.accountId, 10_000);
          if (!reconnected) {
            throw new Error(`[wecom-outbound] Bot WS 等待重连超时，无法发送消息 (accountId=${botAccount.accountId})`);
          }
        }
        if (!chatid) {
          throw new Error(`[wecom-outbound] Bot WS 无法解析目标 chatid (to=${String(to)})`);
        }
        // 重连后重新获取 client（可能是新实例）
        const activeClient = getWsClient(botAccount.accountId);
        if (!activeClient?.isConnected) {
          throw new Error(`[wecom-outbound] Bot WS 重连后仍不可用 (accountId=${botAccount.accountId})`);
        }
        await activeClient.sendMessage(chatid, {
          msgtype: 'markdown',
          markdown: { content: text },
        });
        console.log(`[wecom-outbound] Sent text via Bot WS to chatid=${chatid} (len=${text.length})`);
        return { channel: "wecom", messageId: `ws-bot-${Date.now()}`, timestamp: Date.now() };
      }
    }

    // ── Agent outbound（Webhook Bot 双模 / Agent 独立）──
    const agent = resolveAgentConfigOrThrow({ cfg, accountId });
    const target = resolveWecomTarget(to);
    if (!target) {
      throw new Error("WeCom outbound requires a target (userid, partyid, tagid or chatid).");
    }

    // 体验优化：/new /reset 的“New session started”回执在 OpenClaw 核心里是英文固定文案，
    // 且通过 routeReply 走 wecom outbound（Agent 主动发送）。
    // 在 WeCom“双模式”场景下，这会造成：
    // - 用户在 Bot 会话发 /new，但却收到一条 Agent 私信回执（双重回复/错会话）。
    // 因此：
    // - Bot 会话目标：抑制该回执（Bot 会话里由 wecom 插件补中文回执）。
    // - Agent 会话目标（wecom-agent:）：允许发送，但改写成中文。
    let outgoingText = text;
    const trimmed = String(outgoingText ?? "").trim();
    const rawTo = typeof to === "string" ? to.trim().toLowerCase() : "";
    const isAgentSessionTarget = rawTo.startsWith("wecom-agent:");
    const looksLikeNewSessionAck =
      /new session started/i.test(trimmed) && /model:/i.test(trimmed);

    if (looksLikeNewSessionAck) {
      if (!isAgentSessionTarget) {
        console.log(`[wecom-outbound] Suppressed command ack to avoid Bot/Agent double-reply (len=${trimmed.length})`);
        return { channel: "wecom", messageId: `suppressed-${Date.now()}`, timestamp: Date.now() };
      }

      const modelLabel = (() => {
        const m = trimmed.match(/model:\s*([^\n()]+)\s*/i);
        return m?.[1]?.trim();
      })();
      const rewritten = modelLabel ? `✅ 已开启新会话（模型：${modelLabel}）` : "✅ 已开启新会话。";
      console.log(`[wecom-outbound] Rewrote command ack for agent session (len=${rewritten.length})`);
      outgoingText = rewritten;
    }

    const { touser, toparty, totag, chatid } = target;
    if (chatid) {
      throw new Error(
        `企业微信（WeCom）Agent 主动发送不支持向群 chatId 发送（chatId=${chatid}）。` +
          `该路径在实际环境中经常失败（例如 86008：无权限访问该会话/会话由其他应用创建）。` +
          `请改为发送给用户（userid / user:xxx），或由 Bot 模式在群内交付。`,
      );
    }
    console.log(`[wecom-outbound] Sending text to target=${JSON.stringify(target)} (len=${outgoingText.length})`);

    try {
      await sendAgentText({
        agent,
        toUser: touser,
        toParty: toparty,
        toTag: totag,
        chatId: chatid,
        text: outgoingText,
      });
      console.log(`[wecom-outbound] Successfully sent text to ${JSON.stringify(target)}`);
    } catch (err) {
      console.error(`[wecom-outbound] Failed to send text to ${JSON.stringify(target)}:`, err);
      throw err;
    }

    return {
      channel: "wecom",
      messageId: `agent-${Date.now()}`,
      timestamp: Date.now(),
    };
  },
  sendMedia: async ({ cfg, to, text, mediaUrl, accountId }: ChannelOutboundContext) => {
    if (!mediaUrl) {
      throw new Error("WeCom outbound requires mediaUrl.");
    }

    // ── Bot WebSocket 模式 ──
    const resolvedAccount = resolveWecomAccount({ cfg, accountId });
    const botAccount = resolvedAccount.bot;
    if (botAccount?.connectionMode === "websocket" && botAccount.configured) {
      const rawTo = typeof to === "string" ? to.trim().toLowerCase() : "";
      // 如果目标是 Agent 会话（wecom-agent:），跳过 WS Bot，走 Agent outbound
      if (!rawTo.startsWith("wecom-agent:")) {
        const wsTarget = resolveWecomTarget(to);
        const chatId = wsTarget?.touser || wsTarget?.chatid;
        if (!chatId) {
          throw new Error(`[wecom-outbound] Bot WS sendMedia 无法解析目标 chatId (to=${String(to)})`);
        }

        // 确保 WS 连接可用
        let wsClient = getWsClient(botAccount.accountId);
        if (!wsClient?.isConnected) {
          console.log(`[wecom-outbound] Bot WS 未连接，等待重连... (accountId=${botAccount.accountId})`);
          const reconnected = await waitForWsConnection(botAccount.accountId, 10_000);
          if (!reconnected) {
            throw new Error(`[wecom-outbound] Bot WS 等待重连超时，无法发送媒体 (accountId=${botAccount.accountId})`);
          }
          wsClient = getWsClient(botAccount.accountId);
        }
        if (!wsClient?.isConnected) {
          throw new Error(`[wecom-outbound] Bot WS 重连后仍不可用 (accountId=${botAccount.accountId})`);
        }

        // 加载媒体并通过 WSClient 上传发送
        const { buffer, contentType, filename } = await loadMediaBuffer(mediaUrl);
        console.log(`[wecom-outbound] Bot WS sendMedia: chatId=${chatId} filename=${filename} contentType=${contentType} size=${buffer.length}`);

        const result = await uploadAndSendMediaBuffer({
          wsClient,
          buffer,
          contentType,
          fileName: filename,
          chatId,
          log: (msg) => console.log(`[wecom-outbound] ${msg}`),
          errorLog: (msg) => console.error(`[wecom-outbound] ${msg}`),
        });

        if (result.rejected) {
          throw new Error(`WeCom Bot WS 媒体被拒绝: ${result.rejectReason}`);
        }
        if (!result.ok) {
          throw new Error(`WeCom Bot WS 媒体发送失败: ${result.error}`);
        }

        console.log(`[wecom-outbound] Bot WS sendMedia 成功: type=${result.finalType}${result.downgraded ? ` (降级: ${result.downgradeNote})` : ""}`);
        return {
          channel: "wecom",
          messageId: `ws-bot-media-${Date.now()}`,
          timestamp: Date.now(),
        };
      }
    }

    // ── Agent 模式 ──
    const agent = resolveAgentConfigOrThrow({ cfg, accountId });
    const target = resolveWecomTarget(to);
    if (!target) {
      throw new Error("WeCom outbound requires a target (userid, partyid, tagid or chatid).");
    }
    if (target.chatid) {
      throw new Error(
        `企业微信（WeCom）Agent 主动发送不支持向群 chatId 发送（chatId=${target.chatid}）。` +
          `该路径在实际环境中经常失败（例如 86008：无权限访问该会话/会话由其他应用创建）。` +
          `请改为发送给用户（userid / user:xxx），或由 Bot 模式在群内交付。`,
      );
    }

    const { buffer, contentType, filename } = await loadMediaBuffer(mediaUrl);

    let mediaType: "image" | "voice" | "video" | "file" = "file";
    if (contentType.startsWith("image/")) mediaType = "image";
    else if (contentType.startsWith("audio/")) mediaType = "voice";
    else if (contentType.startsWith("video/")) mediaType = "video";

    const mediaId = await uploadMedia({
      agent,
      type: mediaType,
      buffer,
      filename,
    });

    const { touser, toparty, totag, chatid } = target;
    console.log(`[wecom-outbound] Sending media (${mediaType}) to ${JSON.stringify(target)} (mediaId=${mediaId})`);

    try {
      await sendAgentMedia({
        agent,
        toUser: touser,
        toParty: toparty,
        toTag: totag,
        chatId: chatid,
        mediaId,
        mediaType,
        ...(mediaType === "video" && text?.trim()
          ? {
            title: text.trim().slice(0, 64),
            description: text.trim().slice(0, 512),
          }
          : {}),
      });
      console.log(`[wecom-outbound] Successfully sent media to ${JSON.stringify(target)}`);
    } catch (err) {
      console.error(`[wecom-outbound] Failed to send media to ${JSON.stringify(target)}:`, err);
      throw err;
    }

    return {
      channel: "wecom",
      messageId: `agent-media-${Date.now()}`,
      timestamp: Date.now(),
    };
  },
};
