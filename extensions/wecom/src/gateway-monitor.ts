import type {
  ChannelGatewayContext,
  OpenClawConfig,
  PluginRuntime,
} from "openclaw/plugin-sdk";

import {
  detectMode,
  listWecomAccountIds,
  resolveWecomAccount,
  resolveWecomAccountConflict,
} from "./config/index.js";
import { registerAgentWebhookTarget, registerWecomWebhookTarget } from "./monitor.js";
import { startWsClient } from "./ws-adapter.js";
import type { ResolvedWecomAccount, WecomConfig } from "./types/index.js";
import { WEBHOOK_PATHS } from "./types/constants.js";

type AccountRouteRegistryItem = {
  botPaths: string[];
  agentPaths: string[];
};

const accountRouteRegistry = new Map<string, AccountRouteRegistryItem>();

function logRegisteredRouteSummary(
  ctx: ChannelGatewayContext<ResolvedWecomAccount>,
  preferredOrder: string[],
): void {
  const seen = new Set<string>();
  const orderedAccountIds = [
    ...preferredOrder.filter((accountId) => accountRouteRegistry.has(accountId)),
    ...Array.from(accountRouteRegistry.keys())
      .filter((accountId) => !seen.has(accountId))
      .sort((a, b) => a.localeCompare(b)),
  ].filter((accountId) => {
    if (seen.has(accountId)) return false;
    seen.add(accountId);
    return true;
  });

  const entries = orderedAccountIds
    .map((accountId) => {
      const routes = accountRouteRegistry.get(accountId);
      if (!routes) return undefined;
      const botText = routes.botPaths.length > 0 ? routes.botPaths.join(", ") : "未启用";
      const agentText = routes.agentPaths.length > 0 ? routes.agentPaths.join(", ") : "未启用";
      return `accountId=${accountId}（Bot: ${botText}；Agent: ${agentText}）`;
    })
    .filter((entry): entry is string => Boolean(entry));
  const summary = entries.length > 0 ? entries.join("； ") : "无";
  ctx.log?.info(`[${ctx.account.accountId}] 已注册账号路由汇总：${summary}`);
}

function resolveExpectedRouteSummaryAccountIds(cfg: OpenClawConfig): string[] {
  return listWecomAccountIds(cfg)
    .filter((accountId) => {
      const conflict = resolveWecomAccountConflict({ cfg, accountId });
      if (conflict) return false;
      const account = resolveWecomAccount({ cfg, accountId });
      if (!account.enabled || !account.configured) return false;
      return Boolean(account.bot?.configured || account.agent?.configured);
    })
    .sort((a, b) => a.localeCompare(b));
}

function waitForAbortSignal(abortSignal: AbortSignal): Promise<void> {
  if (abortSignal.aborted) {
    return Promise.resolve();
  }
  return new Promise<void>((resolve) => {
    const onAbort = () => {
      abortSignal.removeEventListener("abort", onAbort);
      resolve();
    };
    abortSignal.addEventListener("abort", onAbort, { once: true });
  });
}

function uniquePaths(paths: string[]): string[] {
  return Array.from(new Set(paths.map((path) => path.trim()).filter(Boolean)));
}

function resolveBotRegistrationPaths(params: { accountId: string; matrixMode: boolean }): string[] {
  if (params.matrixMode) {
    return uniquePaths([
      `${WEBHOOK_PATHS.BOT_PLUGIN}/${params.accountId}`,
      `${WEBHOOK_PATHS.BOT_ALT}/${params.accountId}`,
      // 兼容老路径：不带 accountId 后缀，签名验证会自动匹配到正确账号
      WEBHOOK_PATHS.BOT_PLUGIN,
      WEBHOOK_PATHS.BOT,
      WEBHOOK_PATHS.BOT_ALT,
    ]);
  }
  return uniquePaths([WEBHOOK_PATHS.BOT_PLUGIN, WEBHOOK_PATHS.BOT, WEBHOOK_PATHS.BOT_ALT]);
}

function resolveAgentRegistrationPaths(params: { accountId: string; matrixMode: boolean }): string[] {
  if (params.matrixMode) {
    return uniquePaths([
      `${WEBHOOK_PATHS.AGENT_PLUGIN}/${params.accountId}`,
      `${WEBHOOK_PATHS.AGENT}/${params.accountId}`,
      // 兼容老路径
      WEBHOOK_PATHS.AGENT_PLUGIN,
      WEBHOOK_PATHS.AGENT,
    ]);
  }
  return uniquePaths([WEBHOOK_PATHS.AGENT_PLUGIN, WEBHOOK_PATHS.AGENT]);
}

/**
 * Keeps WeCom webhook targets registered for the account lifecycle.
 * The promise only settles after gateway abort/reload signals shutdown.
 */
export async function monitorWecomProvider(
  ctx: ChannelGatewayContext<ResolvedWecomAccount>,
): Promise<void> {
  const account = ctx.account;
  const cfg = ctx.cfg as OpenClawConfig;
  const expectedRouteSummaryAccountIds = resolveExpectedRouteSummaryAccountIds(cfg);
  const conflict = resolveWecomAccountConflict({
    cfg,
    accountId: account.accountId,
  });
  if (conflict) {
    ctx.setStatus({
      accountId: account.accountId,
      running: false,
      configured: false,
      lastError: conflict.message,
    });
    throw new Error(conflict.message);
  }
  const mode = detectMode(cfg.channels?.wecom as WecomConfig | undefined);
  const matrixMode = mode === "matrix";
  const bot = account.bot;
  const agent = account.agent;
  const botConfigured = Boolean(bot?.configured);
  const agentConfigured = Boolean(agent?.configured);

  if (mode === "legacy" && (botConfigured || agentConfigured)) {
    if (agentConfigured && !botConfigured) {
      ctx.log?.warn(
        `[${account.accountId}] 检测到仍在使用单 Agent 兼容模式。建议尽快升级为多账号模式：` +
        `将 channels.wecom.agent 迁移到 channels.wecom.accounts.<accountId>.agent，` +
        `并设置 channels.wecom.defaultAccount。`,
      );
    } else {
      ctx.log?.warn(
        `[${account.accountId}] 检测到仍在使用单账号兼容模式。建议尽快升级为多账号模式：` +
        `将 channels.wecom.bot/agent 迁移到 channels.wecom.accounts.<accountId>.bot/agent，` +
        `并设置 channels.wecom.defaultAccount。`,
      );
    }
  }

  if (!botConfigured && !agentConfigured) {
    ctx.log?.warn(`[${account.accountId}] wecom not configured; channel is idle`);
    ctx.setStatus({ accountId: account.accountId, running: false, configured: false });
    await waitForAbortSignal(ctx.abortSignal);
    return;
  }

  const unregisters: Array<() => void> = [];
  const botPaths: string[] = [];
  const agentPaths: string[] = [];
  try {
    if (bot && botConfigured) {
      const connectionMode = bot.connectionMode ?? 'webhook';

      if (connectionMode === 'websocket') {
        // 长链接模式：启动 WSClient
        unregisters.push(
          startWsClient({
            accountId: account.accountId,
            botId: bot.botId!,
            secret: bot.secret!,
            account: bot,
            config: cfg,
            runtime: ctx.runtime,
            core: {} as PluginRuntime,
            statusSink: (patch) => ctx.setStatus({ accountId: ctx.accountId, ...patch }),
            welcomeText: bot.config.welcomeText,
            network: bot.network,
          }),
        );
        botPaths.push(`ws://${account.accountId}`);
        ctx.log?.info(`[${account.accountId}] wecom bot websocket client started (botId=${bot.botId})`);
      } else {
        // Webhook 模式：注册 HTTP 路径（现有逻辑不变）
        const paths = resolveBotRegistrationPaths({
          accountId: account.accountId,
          matrixMode,
        });
        for (const path of paths) {
          unregisters.push(
            registerWecomWebhookTarget({
              account: bot,
              config: cfg,
              runtime: ctx.runtime,
              core: {} as PluginRuntime,
              path,
              statusSink: (patch) => ctx.setStatus({ accountId: ctx.accountId, ...patch }),
            }),
          );
        }
        botPaths.push(...paths);
        ctx.log?.info(`[${account.accountId}] wecom bot webhook registered at ${paths.join(", ")}`);
      }
    }

    if (agent && agentConfigured) {
      const paths = resolveAgentRegistrationPaths({
        accountId: account.accountId,
        matrixMode,
      });
      for (const path of paths) {
        unregisters.push(
          registerAgentWebhookTarget({
            agent,
            config: cfg,
            runtime: ctx.runtime,
            path,
          }),
        );
      }
      agentPaths.push(...paths);
      ctx.log?.info(`[${account.accountId}] wecom agent webhook registered at ${paths.join(", ")}`);
    }

    accountRouteRegistry.set(account.accountId, { botPaths, agentPaths });
    const shouldLogSummary =
      expectedRouteSummaryAccountIds.length <= 1 ||
      expectedRouteSummaryAccountIds.every((accountId) => accountRouteRegistry.has(accountId));
    if (shouldLogSummary) {
      logRegisteredRouteSummary(ctx, expectedRouteSummaryAccountIds);
    }

    ctx.setStatus({
      accountId: account.accountId,
      running: true,
      configured: true,
      webhookPath: botConfigured
        ? (botPaths[0] ?? WEBHOOK_PATHS.BOT_PLUGIN)
        : (agentPaths[0] ?? WEBHOOK_PATHS.AGENT_PLUGIN),
      lastStartAt: Date.now(),
    });

    await waitForAbortSignal(ctx.abortSignal);
  } finally {
    for (const unregister of unregisters) {
      unregister();
    }
    accountRouteRegistry.delete(account.accountId);
    ctx.setStatus({
      accountId: account.accountId,
      running: false,
      lastStopAt: Date.now(),
    });
  }
}
