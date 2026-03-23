import type {
  ChannelAccountSnapshot,
  ChannelPlugin,
  OpenClawConfig,
} from "openclaw/plugin-sdk";
import {
  deleteAccountFromConfigSection,
  setAccountEnabledInConfigSection,
} from "openclaw/plugin-sdk";

import {
  DEFAULT_ACCOUNT_ID,
  listWecomAccountIds,
  resolveDefaultWecomAccountId,
  resolveWecomAccount,
  resolveWecomAccountConflict,
} from "./config/index.js";
import type { ResolvedWecomAccount, WecomBotConfig } from "./types/index.js";
import { monitorWecomProvider } from "./gateway-monitor.js";
import { setWecomBotConfig, wecomOnboardingAdapter } from "./onboarding.js";
import { wecomOutbound } from "./outbound.js";
import { WEBHOOK_PATHS } from "./types/constants.js";

const meta = {
  id: "wecom",
  label: "WeCom",
  selectionLabel: "WeCom (plugin)",
  docsPath: "/channels/wecom",
  docsLabel: "wecom",
  blurb: "Enterprise WeCom intelligent bot (API mode) via encrypted webhooks + passive replies.",
  aliases: ["wechatwork", "wework", "qywx", "企微", "企业微信"],
  order: 85,
  quickstartAllowFrom: true,
};

function normalizeWecomMessagingTarget(raw: string): string | undefined {
  const trimmed = raw.trim();
  if (!trimmed) return undefined;
  return trimmed.replace(/^(wecom-agent|wecom|wechatwork|wework|qywx):/i, "").trim() || undefined;
}

export const wecomPlugin: ChannelPlugin<ResolvedWecomAccount> = {
  id: "wecom",
  meta,
  onboarding: wecomOnboardingAdapter,
  setup: {
    resolveAccountId: ({ cfg, accountId }) => {
      return accountId?.trim() || resolveDefaultWecomAccountId(cfg as OpenClawConfig) || DEFAULT_ACCOUNT_ID;
    },
    applyAccountConfig: ({ cfg, accountId, input }) => {
      const isWsMode = input.url === "ws" || input.url === "websocket";

      if (isWsMode) {
        // websocket 模式: --bot-token → botId, --token → secret
        const botConfig: WecomBotConfig = {
          connectionMode: "websocket",
          botId: input.botToken?.trim() || undefined,
          secret: input.token?.trim() || undefined,
        };
        return setWecomBotConfig(cfg as OpenClawConfig, botConfig, accountId);
      }

      // webhook 模式: --token → token, --access-token → encodingAESKey
      const botConfig: WecomBotConfig = {
        connectionMode: "webhook",
        token: input.token?.trim() ?? "",
        encodingAESKey: input.accessToken?.trim() ?? "",
      };
      return setWecomBotConfig(cfg as OpenClawConfig, botConfig, accountId);
    },
    validateInput: ({ input }) => {
      const isWsMode = input.url === "ws" || input.url === "websocket";
      if (isWsMode) {
        if (!input.botToken?.trim()) return "websocket 模式需要 --bot-token <BotID>";
        if (!input.token?.trim()) return "websocket 模式需要 --token <Secret>";
      } else {
        if (!input.token?.trim()) return "webhook 模式需要 --token <Token>";
      }
      return null;
    },
  },
  capabilities: {
    chatTypes: ["direct", "group"],
    media: true,
    reactions: false,
    threads: false,
    polls: false,
    nativeCommands: false,
    blockStreaming: true,
  },
  reload: { configPrefixes: ["channels.wecom"] },
  // NOTE: We intentionally avoid Zod -> JSON Schema conversion at plugin-load time.
  // Some OpenClaw runtime environments load plugin modules via jiti in a way that can
  // surface zod `toJSONSchema()` binding issues (e.g. `this` undefined leading to `_zod` errors).
  // A permissive schema keeps config UX working while preventing startup failures.
  configSchema: {
    schema: {
      type: "object",
      additionalProperties: true,
      properties: {},
    },
  },
  config: {
    listAccountIds: (cfg) => listWecomAccountIds(cfg as OpenClawConfig),
    resolveAccount: (cfg, accountId) => resolveWecomAccount({ cfg: cfg as OpenClawConfig, accountId }),
    defaultAccountId: (cfg) => resolveDefaultWecomAccountId(cfg as OpenClawConfig),
    setAccountEnabled: ({ cfg, accountId, enabled }) =>
      setAccountEnabledInConfigSection({
        cfg: cfg as OpenClawConfig,
        sectionKey: "wecom",
        accountId,
        enabled,
        allowTopLevel: true,
      }),
    deleteAccount: ({ cfg, accountId }) =>
      deleteAccountFromConfigSection({
        cfg: cfg as OpenClawConfig,
        sectionKey: "wecom",
        accountId,
        clearBaseFields: ["bot", "agent"],
      }),
    isConfigured: (account, cfg) => {
      if (!account.configured) {
        return false;
      }
      return !resolveWecomAccountConflict({
        cfg: cfg as OpenClawConfig,
        accountId: account.accountId,
      });
    },
    unconfiguredReason: (account, cfg) =>
      resolveWecomAccountConflict({
        cfg: cfg as OpenClawConfig,
        accountId: account.accountId,
      })?.message ?? "not configured",
    describeAccount: (account, cfg): ChannelAccountSnapshot => {
      const matrixMode = account.accountId !== DEFAULT_ACCOUNT_ID;
      const conflict = resolveWecomAccountConflict({
        cfg: cfg as OpenClawConfig,
        accountId: account.accountId,
      });
      return {
        accountId: account.accountId,
        name: account.name,
        enabled: account.enabled,
        configured: account.configured && !conflict,
        webhookPath: account.bot?.config
          ? (matrixMode ? `${WEBHOOK_PATHS.BOT_PLUGIN}/${account.accountId}` : WEBHOOK_PATHS.BOT_PLUGIN)
          : account.agent?.config
            ? (matrixMode ? `${WEBHOOK_PATHS.AGENT_PLUGIN}/${account.accountId}` : WEBHOOK_PATHS.AGENT_PLUGIN)
            : WEBHOOK_PATHS.BOT_PLUGIN,
      };
    },
    resolveAllowFrom: ({ cfg, accountId }) => {
      const account = resolveWecomAccount({ cfg: cfg as OpenClawConfig, accountId });
      // 与其他渠道保持一致：直接返回 allowFrom，空则允许所有人
      const allowFrom = account.agent?.config.dm?.allowFrom ?? account.bot?.config.dm?.allowFrom ?? [];
      return allowFrom.map((entry) => String(entry));
    },
    formatAllowFrom: ({ allowFrom }) =>
      allowFrom
        .map((entry) => String(entry).trim())
        .filter(Boolean)
        .map((entry) => entry.toLowerCase()),
  },
  // security 配置在 WeCom 中不需要，框架会通过 resolveAllowFrom 自动判断
  groups: {
    // WeCom bots are usually mention-gated by the platform in groups already.
    resolveRequireMention: () => true,
  },
  threading: {
    resolveReplyToMode: () => "off",
  },
  messaging: {
    normalizeTarget: normalizeWecomMessagingTarget,
    targetResolver: {
      looksLikeId: (raw) => Boolean(raw.trim()),
      hint: "<userid|chatid>",
    },
  },
  outbound: {
    ...wecomOutbound,
  },
  status: {
    defaultRuntime: {
      accountId: DEFAULT_ACCOUNT_ID,
      running: false,
      lastStartAt: null,
      lastStopAt: null,
      lastError: null,
    },
    buildChannelSummary: ({ snapshot }) => ({
      configured: snapshot.configured ?? false,
      running: snapshot.running ?? false,
      webhookPath: snapshot.webhookPath ?? null,
      lastStartAt: snapshot.lastStartAt ?? null,
      lastStopAt: snapshot.lastStopAt ?? null,
      lastError: snapshot.lastError ?? null,
      lastInboundAt: snapshot.lastInboundAt ?? null,
      lastOutboundAt: snapshot.lastOutboundAt ?? null,
      probe: snapshot.probe,
      lastProbeAt: snapshot.lastProbeAt ?? null,
    }),
    probeAccount: async () => ({ ok: true }),
    buildAccountSnapshot: ({ account, runtime, cfg }) => {
      const conflict = resolveWecomAccountConflict({
        cfg: cfg as OpenClawConfig,
        accountId: account.accountId,
      });
      return {
        accountId: account.accountId,
        name: account.name,
        enabled: account.enabled,
        configured: account.configured && !conflict,
        webhookPath: account.bot?.config
          ? (account.accountId === DEFAULT_ACCOUNT_ID
              ? WEBHOOK_PATHS.BOT_PLUGIN
              : `${WEBHOOK_PATHS.BOT_PLUGIN}/${account.accountId}`)
          : account.agent?.config
            ? (account.accountId === DEFAULT_ACCOUNT_ID
                ? WEBHOOK_PATHS.AGENT_PLUGIN
                : `${WEBHOOK_PATHS.AGENT_PLUGIN}/${account.accountId}`)
            : WEBHOOK_PATHS.BOT_PLUGIN,
        running: runtime?.running ?? false,
        lastStartAt: runtime?.lastStartAt ?? null,
        lastStopAt: runtime?.lastStopAt ?? null,
        lastError: runtime?.lastError ?? conflict?.message ?? null,
        lastInboundAt: runtime?.lastInboundAt ?? null,
        lastOutboundAt: runtime?.lastOutboundAt ?? null,
        dmPolicy: account.bot?.config.dm?.policy ?? "pairing",
      };
    },
  },
  gateway: {
    /**
     * **startAccount (启动账号)**
     *
     * WeCom lifecycle is long-running: keep webhook targets active until
     * gateway stop/reload aborts the account.
     */
    startAccount: monitorWecomProvider,
    stopAccount: async (ctx) => {
      ctx.setStatus({
        accountId: ctx.account.accountId,
        running: false,
        lastStopAt: Date.now(),
      });
    },
  },
};
