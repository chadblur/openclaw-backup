// ADP OpenClaw channel plugin for OpenClaw
// Supports: API Token auth, multiple clients, multi-turn conversations

import {
  type ChannelPlugin,
  type ClawdbotConfig,
  DEFAULT_ACCOUNT_ID,
} from "openclaw/plugin-sdk";
import { adpOpenclawOnboardingAdapter } from "./onboarding.js";
import { getActiveWebSocket } from "./runtime.js";

// Default WebSocket URL for ADP OpenClaw
const DEFAULT_WS_URL = "wss://wss.lke.cloud.tencent.com/bot/gateway/conn";

// Channel-level config type (from channels["adp-openclaw"])
export type AdpOpenclawChannelConfig = {
  enabled?: boolean;
  wsUrl?: string; // WebSocket URL (optional, has default)
  clientToken?: string;
  signKey?: string; // HMAC key for signature generation
};

export type ResolvedAdpOpenclawAccount = {
  accountId: string;
  name: string;
  enabled: boolean;
  configured: boolean;
  wsUrl: string; // WebSocket URL
  clientToken: string;
  signKey: string; // HMAC key for signature generation
};

function resolveAdpOpenclawCredentials(channelCfg?: AdpOpenclawChannelConfig): {
  wsUrl: string;
  clientToken: string;
  signKey: string;
} | null {
  // Get wsUrl from config or env (has default value)
  let wsUrl = channelCfg?.wsUrl?.trim();
  if (!wsUrl) {
    wsUrl = process.env.ADP_OPENCLAW_WS_URL || DEFAULT_WS_URL;
  }

  // Get clientToken from config or env
  let clientToken = channelCfg?.clientToken?.trim();
  if (!clientToken) {
    clientToken = process.env.ADP_OPENCLAW_CLIENT_TOKEN || "";
  }

  // Get signKey from config or env (default: ADPOpenClaw)
  let signKey = channelCfg?.signKey?.trim();
  if (!signKey) {
    signKey = process.env.ADP_OPENCLAW_SIGN_KEY || "ADPOpenClaw";
  }

  // clientToken is required for configured status (wsUrl has default)
  if (!clientToken) {
    return null;
  }

  return { wsUrl, clientToken, signKey };
}

function resolveAccount(cfg: ClawdbotConfig, accountId?: string): ResolvedAdpOpenclawAccount {
  const channelCfg = cfg.channels?.["adp-openclaw"] as AdpOpenclawChannelConfig | undefined;
  const enabled = channelCfg?.enabled !== false;
  const creds = resolveAdpOpenclawCredentials(channelCfg);

  return {
    accountId: accountId?.trim() || DEFAULT_ACCOUNT_ID,
    name: "ADP OpenClaw",
    enabled,
    configured: Boolean(creds),
    wsUrl: creds?.wsUrl || DEFAULT_WS_URL,
    clientToken: creds?.clientToken || "",
    signKey: creds?.signKey || "ADPOpenClaw",
  };
}

export const adpOpenclawPlugin: ChannelPlugin<ResolvedAdpOpenclawAccount> = {
  id: "adp-openclaw",
  meta: {
    id: "adp-openclaw",
    label: "ADP OpenClaw",
    selectionLabel: "ADP OpenClaw",
    docsPath: "/channels/adp-openclaw",
    blurb: "ADP channel backed by a Go WebSocket server.",
    order: 999,
  },
  onboarding: adpOpenclawOnboardingAdapter,
  capabilities: {
    chatTypes: ["direct"],
    polls: false,
    reactions: false,
    threads: false,
    media: true, // 启用文件/媒体支持
    /**
     * blockStreaming: true 启用 SDK 的块流式功能
     * SDK 会通过 deliver 回调的 info.kind="block" 传递流式块
     */
    blockStreaming: true,
  },
  reload: { configPrefixes: ["channels.adp-openclaw"] },
  configSchema: {
    schema: {
      type: "object",
      additionalProperties: false,
      properties: {
        enabled: { type: "boolean" },
        wsUrl: { type: "string" }, // WebSocket URL (optional, default: wss://wss.lke.cloud.tencent.com/bot/gateway/conn)
        clientToken: { type: "string" },
        signKey: { type: "string" },
      },
    },
  },
  config: {
    listAccountIds: () => [DEFAULT_ACCOUNT_ID],
    resolveAccount: (cfg) => resolveAccount(cfg),
    defaultAccountId: () => DEFAULT_ACCOUNT_ID,
    setAccountEnabled: ({ cfg, enabled }) => ({
      ...cfg,
      channels: {
        ...cfg.channels,
        "adp-openclaw": {
          ...cfg.channels?.["adp-openclaw"],
          enabled,
        },
      },
    }),
    deleteAccount: ({ cfg }) => {
      const next = { ...cfg } as ClawdbotConfig;
      const nextChannels = { ...cfg.channels };
      delete (nextChannels as Record<string, unknown>)["adp-openclaw"];
      if (Object.keys(nextChannels).length > 0) {
        next.channels = nextChannels;
      } else {
        delete next.channels;
      }
      return next;
    },
    isConfigured: (_account, cfg) =>
      Boolean(resolveAdpOpenclawCredentials(cfg.channels?.["adp-openclaw"] as AdpOpenclawChannelConfig | undefined)),
    describeAccount: (account) => ({
      accountId: account.accountId,
      name: account.name,
      enabled: account.enabled,
      configured: account.configured,
      wsUrl: account.wsUrl,
    }),
    resolveAllowFrom: () => [],
    formatAllowFrom: ({ allowFrom }) => allowFrom,
  },
  setup: {
    resolveAccountId: () => DEFAULT_ACCOUNT_ID,
    applyAccountConfig: ({ cfg }) => ({
      ...cfg,
      channels: {
        ...cfg.channels,
        "adp-openclaw": {
          ...cfg.channels?.["adp-openclaw"],
          enabled: true,
        },
      },
    }),
  },
  status: {
    defaultRuntime: {
      accountId: DEFAULT_ACCOUNT_ID,
      running: false,
      lastStartAt: null,
      lastStopAt: null,
      lastError: null,
    },
    collectStatusIssues: () => [],
    buildChannelSummary: ({ snapshot }) => ({
      configured: snapshot.configured ?? false,
      wsUrl: snapshot.wsUrl ?? null,
      running: snapshot.running ?? false,
      lastStartAt: snapshot.lastStartAt ?? null,
      lastStopAt: snapshot.lastStopAt ?? null,
      lastError: snapshot.lastError ?? null,
    }),
    probeAccount: async ({ cfg }) => {
      // For WebSocket-only architecture, we just check if config is valid
      const account = resolveAccount(cfg);
      const start = Date.now();
      return {
        ok: account.configured && Boolean(account.clientToken),
        elapsedMs: Date.now() - start,
      };
    },
    buildAccountSnapshot: ({ account, runtime, probe }) => ({
      accountId: account.accountId,
      name: account.name,
      enabled: account.enabled,
      configured: account.configured,
      wsUrl: account.wsUrl,
      running: runtime?.running ?? false,
      lastStartAt: runtime?.lastStartAt ?? null,
      lastStopAt: runtime?.lastStopAt ?? null,
      lastError: runtime?.lastError ?? null,
      probe,
    }),
  },
  gateway: {
    startAccount: async (ctx) => {
      const account = ctx.account;
      ctx.setStatus({ accountId: account.accountId, wsUrl: account.wsUrl });
      ctx.log?.info(`[adp-openclaw] starting WebSocket connection → ${account.wsUrl}`);

      const { monitorAdpOpenclaw } = await import("./monitor.js");
      return monitorAdpOpenclaw({
        wsUrl: account.wsUrl,
        clientToken: account.clientToken,
        signKey: account.signKey,
        abortSignal: ctx.abortSignal,
        log: ctx.log,
        cfg: ctx.cfg,
        setStatus: ctx.setStatus,
      });
    },
  },
  // Outbound message support for the "message" tool and cron tasks
  outbound: {
    deliveryMode: "direct",
    
    // Send text message
    sendText: async (ctx) => {
      const { to, text } = ctx;
      const ws = getActiveWebSocket();
      
      if (!ws) {
        console.error("[adp-openclaw] No active WebSocket connection for outbound message");
        throw new Error("No active WebSocket connection");
      }
      
      // Parse target: expected format is "adp-openclaw:{userId}" or just "{userId}"
      const targetParts = to.split(":");
      const targetUserId = targetParts.length > 1 ? targetParts.slice(1).join(":") : to;
      
      console.log(`[adp-openclaw] Sending outbound text to ${targetUserId}: ${text.slice(0, 50)}...`);
      
      // Generate unique request ID
      const requestId = `outbound-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      
      const outMsg = {
        type: "outbound",
        requestId,
        payload: {
          to: targetUserId,
          text: text,
        },
        timestamp: Date.now(),
      };
      
      ws.send(JSON.stringify(outMsg));
      return { channel: "adp-openclaw", messageId: requestId };
    },
    
    // Send media message (text with optional media URL)
    sendMedia: async (ctx) => {
      const { to, text, mediaUrl } = ctx;
      const ws = getActiveWebSocket();
      
      if (!ws) {
        console.error("[adp-openclaw] No active WebSocket connection for outbound media");
        throw new Error("No active WebSocket connection");
      }
      
      // Parse target
      const targetParts = to.split(":");
      const targetUserId = targetParts.length > 1 ? targetParts.slice(1).join(":") : to;
      
      console.log(`[adp-openclaw] Sending outbound media to ${targetUserId}: ${text.slice(0, 50)}... (media: ${mediaUrl || "none"})`);
      
      // Generate unique request ID
      const requestId = `outbound-media-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      
      // Combine text and media URL if present
      const finalText = mediaUrl ? `${text}\n\n📎 ${mediaUrl}` : text;
      
      const outMsg = {
        type: "outbound",
        requestId,
        payload: {
          to: targetUserId,
          text: finalText,
          mediaUrl: mediaUrl,
        },
        timestamp: Date.now(),
      };
      
      ws.send(JSON.stringify(outMsg));
      return { channel: "adp-openclaw", messageId: requestId };
    },
  },
};
