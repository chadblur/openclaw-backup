/**
 * WeCom 配置向导 (Onboarding)
 * 支持 Bot、Agent 和双模式同时启动的交互式配置流程
 */

import type {
    ChannelOnboardingAdapter,
    ChannelOnboardingDmPolicy,
    OpenClawConfig,
    WizardPrompter,
} from "openclaw/plugin-sdk";
import { DEFAULT_ACCOUNT_ID, promptAccountId } from "openclaw/plugin-sdk";
import { listWecomAccountIds, resolveDefaultWecomAccountId, resolveWecomAccount, resolveWecomAccounts } from "./config/index.js";
import type { WecomConfig, WecomBotConfig, WecomAgentConfig, WecomDmConfig, WecomAccountConfig } from "./types/index.js";

const channel = "wecom" as const;

type WecomMode = "bot" | "agent" | "both";

// ============================================================
// 辅助函数
// ============================================================

function getWecomConfig(cfg: OpenClawConfig): WecomConfig | undefined {
    return cfg.channels?.wecom as WecomConfig | undefined;
}

function setWecomEnabled(cfg: OpenClawConfig, enabled: boolean): OpenClawConfig {
    return {
        ...cfg,
        channels: {
            ...cfg.channels,
            wecom: {
                ...(cfg.channels?.wecom ?? {}),
                enabled,
            },
        },
    } as OpenClawConfig;
}

/**
 * 确保 cfg.bindings 中存在一条 wecom 账号到默认 agent 的路由。
 *
 * `openclaw channels add` 流程会在插件 configure() 返回后单独提示用户绑定 agent，
 * 但 `openclaw onboard` 的 quickstart 路径会跳过这一步，导致消息路由缺失。
 * 在插件层面主动补全 binding 可以让两种流程都能正常工作。
 *
 * 如果 bindings 中已存在匹配 channel+accountId 的条目，则不会重复添加。
 */
function ensureWecomBinding(cfg: OpenClawConfig, accountId: string): OpenClawConfig {
    const existing = cfg.bindings ?? [];
    const alreadyBound = existing.some(
        (b) => b.match.channel === channel && (b.match.accountId === accountId || (!b.match.accountId && accountId === DEFAULT_ACCOUNT_ID)),
    );
    if (alreadyBound) return cfg;

    // 默认路由到 main agent（OpenClaw 约定 defaultAgentId 为 "main"）
    const defaultAgentId = "main";
    return {
        ...cfg,
        bindings: [
            ...existing,
            {
                agentId: defaultAgentId,
                match: {
                    channel,
                    accountId,
                },
            },
        ],
    };
}

function setGatewayBindLan(cfg: OpenClawConfig): OpenClawConfig {
    return {
        ...cfg,
        gateway: {
            ...(cfg.gateway ?? {}),
            bind: "lan",
        },
    } as OpenClawConfig;
}

function setWecomDefaultAccount(cfg: OpenClawConfig, accountId: string): OpenClawConfig {
    const wecom = getWecomConfig(cfg) ?? {};
    return {
        ...cfg,
        channels: {
            ...cfg.channels,
            wecom: {
                ...wecom,
                defaultAccount: accountId,
            },
        },
    } as OpenClawConfig;
}

function shouldUseAccountScopedConfig(wecom: WecomConfig | undefined, accountId: string): boolean {
    void wecom;
    void accountId;
    return true;
}

function ensureMatrixAccounts(wecom: WecomConfig): WecomConfig {
    const accounts = wecom.accounts ?? {};
    if (Object.keys(accounts).length > 0) {
        return wecom;
    }

    if (!wecom.bot && !wecom.agent) {
        return wecom;
    }

    const { bot: legacyBot, agent: legacyAgent, ...rest } = wecom;
    const defaultAccount: WecomAccountConfig = {
        enabled: true,
        ...(legacyBot ? { bot: legacyBot } : {}),
        ...(legacyAgent ? { agent: legacyAgent } : {}),
    };

    return {
        ...rest,
        defaultAccount: rest.defaultAccount?.trim() || DEFAULT_ACCOUNT_ID,
        accounts: {
            [DEFAULT_ACCOUNT_ID]: defaultAccount,
        },
    };
}

function accountWebhookPath(kind: "bot" | "agent", accountId: string): string {
    const recommendedBase = kind === "bot" ? "/plugins/wecom/bot" : "/plugins/wecom/agent";
    return `${recommendedBase}/${accountId}`;
}

export function setWecomBotConfig(cfg: OpenClawConfig, bot: WecomBotConfig, accountId: string): OpenClawConfig {
    const wecom = getWecomConfig(cfg) ?? {};
    if (!shouldUseAccountScopedConfig(wecom, accountId)) {
        return {
            ...cfg,
            channels: {
                ...cfg.channels,
                wecom: {
                    ...wecom,
                    enabled: true,
                    bot,
                },
            },
        } as OpenClawConfig;
    }

    const matrixWecom = ensureMatrixAccounts(wecom);
    const accounts = matrixWecom.accounts ?? {};
    const existingAccount = accounts[accountId] ?? {};
    return {
        ...cfg,
        channels: {
            ...cfg.channels,
            wecom: {
                ...matrixWecom,
                enabled: true,
                defaultAccount: matrixWecom.defaultAccount?.trim() || DEFAULT_ACCOUNT_ID,
                accounts: {
                    ...accounts,
                    [accountId]: {
                        ...existingAccount,
                        enabled: existingAccount.enabled ?? true,
                        bot,
                    },
                },
            },
        },
    } as OpenClawConfig;
}

function setWecomAgentConfig(cfg: OpenClawConfig, agent: WecomAgentConfig, accountId: string): OpenClawConfig {
    const wecom = getWecomConfig(cfg) ?? {};
    if (!shouldUseAccountScopedConfig(wecom, accountId)) {
        return {
            ...cfg,
            channels: {
                ...cfg.channels,
                wecom: {
                    ...wecom,
                    enabled: true,
                    agent,
                },
            },
        } as OpenClawConfig;
    }

    const matrixWecom = ensureMatrixAccounts(wecom);
    const accounts = matrixWecom.accounts ?? {};
    const existingAccount = accounts[accountId] ?? {};
    return {
        ...cfg,
        channels: {
            ...cfg.channels,
            wecom: {
                ...matrixWecom,
                enabled: true,
                defaultAccount: matrixWecom.defaultAccount?.trim() || DEFAULT_ACCOUNT_ID,
                accounts: {
                    ...accounts,
                    [accountId]: {
                        ...existingAccount,
                        enabled: existingAccount.enabled ?? true,
                        agent,
                    },
                },
            },
        },
    } as OpenClawConfig;
}

function setWecomDmPolicy(
    cfg: OpenClawConfig,
    mode: "bot" | "agent",
    dm: WecomDmConfig,
    accountId: string,
): OpenClawConfig {
    const wecom = getWecomConfig(cfg) ?? {};
    if (shouldUseAccountScopedConfig(wecom, accountId)) {
        const matrixWecom = ensureMatrixAccounts(wecom);
        const accounts = matrixWecom.accounts ?? {};
        const existingAccount = accounts[accountId] ?? {};
        const nextAccount: WecomAccountConfig =
            mode === "bot"
                ? {
                    ...existingAccount,
                    bot: {
                        ...existingAccount.bot,
                        dm,
                    },
                }
                : {
                    ...existingAccount,
                    agent: {
                        ...existingAccount.agent,
                        dm,
                    } as WecomAgentConfig,
                };
        return {
            ...cfg,
            channels: {
                ...cfg.channels,
                wecom: {
                    ...matrixWecom,
                    enabled: true,
                    defaultAccount: matrixWecom.defaultAccount?.trim() || DEFAULT_ACCOUNT_ID,
                    accounts: {
                        ...accounts,
                        [accountId]: {
                            ...nextAccount,
                            enabled: nextAccount.enabled ?? true,
                        },
                    },
                },
            },
        } as OpenClawConfig;
    }

    if (mode === "bot") {
        return {
            ...cfg,
            channels: {
                ...cfg.channels,
                wecom: {
                    ...wecom,
                    bot: {
                        ...wecom.bot,
                        dm,
                    },
                },
            },
        } as OpenClawConfig;
    }
    return {
        ...cfg,
        channels: {
            ...cfg.channels,
            wecom: {
                ...wecom,
                agent: {
                    ...wecom.agent,
                    dm,
                },
            },
        },
    } as OpenClawConfig;
}

async function resolveOnboardingAccountId(params: {
    cfg: OpenClawConfig;
    prompter: WizardPrompter;
    accountOverride?: string;
    shouldPromptAccountIds: boolean;
}): Promise<string> {
    const defaultAccountId = resolveDefaultWecomAccountId(params.cfg);
    const override = params.accountOverride?.trim();
    let accountId = override || defaultAccountId;
    if (!override && params.shouldPromptAccountIds) {
        accountId = await promptAccountId({
            cfg: params.cfg,
            prompter: params.prompter,
            label: "WeCom",
            currentId: accountId,
            listAccountIds: (cfg) => listWecomAccountIds(cfg),
            defaultAccountId,
        });
    }
    return accountId.trim() || DEFAULT_ACCOUNT_ID;
}

// ============================================================
// 欢迎与引导
// ============================================================

async function showWelcome(prompter: WizardPrompter): Promise<void> {
    await prompter.note(
        [
            "🚀 欢迎使用企业微信（WeCom）接入向导",
            "本插件支持「智能体 Bot」与「自建应用 Agent」双模式并行。",
        ].join("\n"),
        "WeCom 配置向导",
    );
}

// ============================================================
// 模式选择
// ============================================================

async function promptMode(prompter: WizardPrompter): Promise<WecomMode> {
    const choice = await prompter.select({
        message: "请选择您要配置的接入模式:",
        options: [
            {
                value: "bot",
                label: "Bot 模式 (智能机器人)",
                hint: "回调速度快，支持流式占位符，适合日常对话",
            },
            {
                value: "agent",
                label: "Agent 模式 (自建应用)",
                hint: "功能最全，支持 API 主动推送、发送文件/视频、交互卡片",
            },
            {
                value: "both",
                label: "双模式 (Bot + Agent 同时启用)",
                hint: "推荐：Bot 用于快速对话，Agent 用于主动推送和媒体发送",
            },
        ],
        initialValue: "both",
    });
    return choice as WecomMode;
}

// ============================================================
// Bot 模式配置
// ============================================================

async function configureBotMode(
    cfg: OpenClawConfig,
    prompter: WizardPrompter,
    accountId: string,
): Promise<OpenClawConfig> {
    // 选择接入方式
    const connectionMode = (await prompter.select({
        message: "请选择 Bot 接入方式:",
        options: [
            {
                value: "websocket",
                label: "WebSocket 长链接模式",
                hint: "无需公网 IP，SDK 主动连接企微服务器，适合内网环境",
            },
            {
                value: "webhook",
                label: "Webhook 回调模式",
                hint: "需要公网 IP + 回调 URL，适合有公网服务器的环境",
            },
        ],
        initialValue: "websocket",
    })) as "webhook" | "websocket";

    if (connectionMode === "websocket") {
        return configureBotWebsocket(cfg, prompter, accountId);
    }
    return configureBotWebhook(cfg, prompter, accountId);
}

async function configureBotWebhook(
    cfg: OpenClawConfig,
    prompter: WizardPrompter,
    accountId: string,
): Promise<OpenClawConfig> {
    const recommendedPath = accountWebhookPath("bot", accountId);
    await prompter.note(
        [
            "正在配置 Bot 模式（Webhook 回调）...",
            "",
            "💡 操作指南: 请在企微后台【管理工具 -> 智能机器人】开启 API 模式。",
            `🔗 回调 URL (推荐): https://您的域名${recommendedPath}`,
            "",
            "请先在后台填入回调 URL，然后获取以下信息。",
        ].join("\n"),
        "Bot 模式配置 — Webhook",
    );

    const token = String(
        await prompter.text({
            message: "请输入 Token:",
            validate: (value: string | undefined) => (value?.trim() ? undefined : "Token 不能为空"),
        }),
    ).trim();

    const encodingAESKey = String(
        await prompter.text({
            message: "请输入 EncodingAESKey:",
            validate: (value: string | undefined) => {
                const v = value?.trim() ?? "";
                if (!v) return "EncodingAESKey 不能为空";
                if (v.length !== 43) return "EncodingAESKey 应为 43 个字符";
                return undefined;
            },
        }),
    ).trim();

    const streamPlaceholder = await prompter.text({
        message: "流式占位符 (可选):",
        placeholder: "正在思考...",
        initialValue: "正在思考...",
    });

    const welcomeText = await prompter.text({
        message: "欢迎语 (可选):",
        placeholder: "你好！我是 AI 助手",
        initialValue: "你好！我是 AI 助手",
    });

    const botConfig: WecomBotConfig = {
        connectionMode: "webhook",
        token,
        encodingAESKey,
        streamPlaceholderContent: streamPlaceholder?.trim() || undefined,
        welcomeText: welcomeText?.trim() || undefined,
    };

    return setWecomBotConfig(cfg, botConfig, accountId);
}

async function configureBotWebsocket(
    cfg: OpenClawConfig,
    prompter: WizardPrompter,
    accountId: string,
): Promise<OpenClawConfig> {
    await prompter.note(
        [
            "正在配置 Bot 模式（WebSocket 长链接）...",
            "",
            "💡 操作指南: 请在企微后台【管理工具 -> 智能机器人】获取 BotID 和 Secret。",
            "",
            "长链接模式无需公网 IP 和回调 URL，适合内网环境。",
        ].join("\n"),
        "Bot 模式配置 — WebSocket",
    );

    const botId = String(
        await prompter.text({
            message: "请输入 BotID (机器人ID):",
            validate: (value: string | undefined) => (value?.trim() ? undefined : "BotID 不能为空"),
        }),
    ).trim();

    const secret = String(
        await prompter.text({
            message: "请输入 Secret (机器人密钥):",
            validate: (value: string | undefined) => (value?.trim() ? undefined : "Secret 不能为空"),
        }),
    ).trim();

    const streamPlaceholder = await prompter.text({
        message: "流式占位符 (可选):",
        placeholder: "正在思考...",
        initialValue: "正在思考...",
    });

    const welcomeText = await prompter.text({
        message: "欢迎语 (可选):",
        placeholder: "你好！我是 AI 助手",
        initialValue: "你好！我是 AI 助手",
    });

    const botConfig: WecomBotConfig = {
        connectionMode: "websocket",
        botId,
        secret,
        streamPlaceholderContent: streamPlaceholder?.trim() || undefined,
        welcomeText: welcomeText?.trim() || undefined,
    };

    return setWecomBotConfig(cfg, botConfig, accountId);
}

// ============================================================
// Agent 模式配置
// ============================================================

async function configureAgentMode(
    cfg: OpenClawConfig,
    prompter: WizardPrompter,
    accountId: string,
): Promise<OpenClawConfig> {
    const recommendedPath = accountWebhookPath("agent", accountId);
    await prompter.note(
        [
            "正在配置 Agent 模式...",
            "",
            "💡 操作指南: 请在企微后台【应用管理 -> 自建应用】创建应用。",
        ].join("\n"),
        "Agent 模式配置",
    );

    const corpId = String(
        await prompter.text({
            message: "请输入 CorpID (企业ID):",
            validate: (value: string | undefined) => (value?.trim() ? undefined : "CorpID 不能为空"),
        }),
    ).trim();

    const agentIdStr = String(
        await prompter.text({
            message: "请输入 AgentID (应用ID):",
            validate: (value: string | undefined) => {
                const v = value?.trim() ?? "";
                if (!v) return "AgentID 不能为空";
                if (!/^\d+$/.test(v)) return "AgentID 应为数字";
                return undefined;
            },
        }),
    ).trim();
    const agentId = Number(agentIdStr);

    const corpSecret = String(
        await prompter.text({
            message: "请输入 Secret (应用密钥):",
            validate: (value: string | undefined) => (value?.trim() ? undefined : "Secret 不能为空"),
        }),
    ).trim();

    await prompter.note(
        [
            "💡 操作指南: 请在自建应用详情页进入【接收消息 -> 设置API接收】。",
            `🔗 回调 URL (推荐): https://您的域名${recommendedPath}`,
            "",
            "请先在后台填入回调 URL，然后获取以下信息。",
        ].join("\n"),
        "回调配置",
    );

    const token = String(
        await prompter.text({
            message: "请输入 Token (回调令牌):",
            validate: (value: string | undefined) => (value?.trim() ? undefined : "Token 不能为空"),
        }),
    ).trim();

    const encodingAESKey = String(
        await prompter.text({
            message: "请输入 EncodingAESKey (回调加密密钥):",
            validate: (value: string | undefined) => {
                const v = value?.trim() ?? "";
                if (!v) return "EncodingAESKey 不能为空";
                if (v.length !== 43) return "EncodingAESKey 应为 43 个字符";
                return undefined;
            },
        }),
    ).trim();

    const welcomeText = await prompter.text({
        message: "欢迎语 (可选):",
        placeholder: "欢迎使用智能助手",
        initialValue: "欢迎使用智能助手",
    });

    const agentConfig: WecomAgentConfig = {
        corpId,
        corpSecret,
        agentId,
        token,
        encodingAESKey,
        welcomeText: welcomeText?.trim() || undefined,
    };

    return setWecomAgentConfig(cfg, agentConfig, accountId);
}

// ============================================================
// DM 策略配置
// ============================================================

async function promptDmPolicy(
    cfg: OpenClawConfig,
    prompter: WizardPrompter,
    modes: ("bot" | "agent")[],
    accountId: string,
): Promise<OpenClawConfig> {
    const policyChoice = await prompter.select({
        message: "请选择私聊 (DM) 访问策略:",
        options: [
            { value: "pairing", label: "配对模式", hint: "推荐：安全，未知用户需授权" },
            { value: "allowlist", label: "白名单模式", hint: "仅允许特定 UserID" },
            { value: "open", label: "开放模式", hint: "任何人可发起" },
            { value: "disabled", label: "禁用私聊", hint: "不接受私聊消息" },
        ],
        initialValue: "pairing",
    });

    const policy = policyChoice as "pairing" | "allowlist" | "open" | "disabled";
    let allowFrom: string[] | undefined;

    if (policy === "allowlist") {
        const allowFromStr = String(
            await prompter.text({
                message: "请输入白名单 UserID (多个用逗号分隔):",
                placeholder: "user1,user2",
                validate: (value: string | undefined) => (value?.trim() ? undefined : "请输入至少一个 UserID"),
            }),
        ).trim();
        allowFrom = allowFromStr.split(",").map((s) => s.trim()).filter(Boolean);
    }

    const dm: WecomDmConfig = { policy, allowFrom };

    let result = cfg;
    for (const mode of modes) {
        result = setWecomDmPolicy(result, mode, dm, accountId);
    }
    return result;
}

// ============================================================
// 配置汇总
// ============================================================

async function showSummary(cfg: OpenClawConfig, prompter: WizardPrompter, accountId: string): Promise<void> {
    const account = resolveWecomAccount({ cfg, accountId });
    const lines: string[] = ["✅ 配置已保存！", ""];

    if (account.bot?.configured) {
        if (account.bot.connectionMode === "websocket") {
            lines.push("📱 Bot 模式: 已配置 (WebSocket 长链接)");
            lines.push("   无需配置回调 URL，SDK 将主动连接企微服务器");
        } else {
            lines.push("📱 Bot 模式: 已配置 (Webhook 回调)");
            lines.push(`   回调 URL: https://您的域名${accountWebhookPath("bot", accountId)}`);
        }
    }

    if (account.agent?.configured) {
        lines.push("🏢 Agent 模式: 已配置");
        lines.push(`   回调 URL: https://您的域名${accountWebhookPath("agent", accountId)}`);
    }

    lines.push(`   账号 ID: ${accountId}`);

    const hasWebhook =
        (account.bot?.configured && account.bot.connectionMode !== "websocket") ||
        account.agent?.configured;

    lines.push("");
    if (hasWebhook) {
        lines.push("⚠️ 请确保您已在企微后台填写了正确的回调 URL，");
        lines.push("   并点击了后台的『保存』按钮完成验证。");
    } else {
        lines.push("💡 WebSocket 模式将在服务启动时自动连接企微服务器。");
    }

    await prompter.note(lines.join("\n"), "配置完成");
}

// ============================================================
// DM Policy Adapter
// ============================================================

const dmPolicy: ChannelOnboardingDmPolicy = {
    label: "WeCom",
    channel,
    policyKey: "channels.wecom.bot.dm.policy",
    allowFromKey: "channels.wecom.bot.dm.allowFrom",
    getCurrent: (cfg: OpenClawConfig) => {
        const account = resolveWecomAccount({ cfg });
        return (account.bot?.config.dm?.policy ?? "pairing") as "pairing";
    },
    setPolicy: (cfg: OpenClawConfig, policy: "pairing" | "allowlist" | "open" | "disabled") => {
        const accountId = resolveDefaultWecomAccountId(cfg);
        return setWecomDmPolicy(cfg, "bot", { policy }, accountId);
    },
    promptAllowFrom: async ({ cfg, prompter }: { cfg: OpenClawConfig; prompter: WizardPrompter }) => {
        const allowFromStr = String(
            await prompter.text({
                message: "请输入白名单 UserID:",
                validate: (value: string | undefined) => (value?.trim() ? undefined : "请输入 UserID"),
            }),
        ).trim();
        const allowFrom = allowFromStr.split(",").map((s) => s.trim()).filter(Boolean);
        const accountId = resolveDefaultWecomAccountId(cfg);
        return setWecomDmPolicy(cfg, "bot", { policy: "allowlist", allowFrom }, accountId);
    },
};

// ============================================================
// Onboarding Adapter
// ============================================================

export const wecomOnboardingAdapter: ChannelOnboardingAdapter = {
    channel,
    dmPolicy,
    getStatus: async ({ cfg }: { cfg: OpenClawConfig }) => {
        const resolved = resolveWecomAccounts(cfg);
        const accounts = Object.values(resolved.accounts).filter((account) => account.enabled !== false);
        const botConfigured = accounts.some((account) => Boolean(account.bot?.configured));
        const agentConfigured = accounts.some((account) => Boolean(account.agent?.configured));
        const configured = accounts.some((account) => account.configured);

        const statusParts: string[] = [];
        if (botConfigured) statusParts.push("Bot ✓");
        if (agentConfigured) statusParts.push("Agent ✓");
        const accountSuffix = accounts.length > 1 ? ` · ${accounts.length} accounts` : "";
        const statusSummary = statusParts.length > 0 ? statusParts.join(" + ") : "已配置";

        return {
            channel,
            configured,
            statusLines: [
                `WeCom: ${configured ? `${statusSummary}${accountSuffix}` : "需要配置"}`,
            ],
            selectionHint: configured
                ? `configured · ${statusSummary}${accountSuffix}`
                : "enterprise-ready · dual-mode",
            quickstartScore: configured ? 1 : 8,
        };
    },
    configure: async ({
        cfg,
        prompter,
        accountOverrides,
        shouldPromptAccountIds,
    }) => {
        // 1. 欢迎
        await showWelcome(prompter);

        // 2. 账号选择
        const accountId = await resolveOnboardingAccountId({
            cfg,
            prompter,
            accountOverride: accountOverrides.wecom,
            shouldPromptAccountIds,
        });

        // 3. 模式选择
        const mode = await promptMode(prompter);

        let next = cfg;
        const configuredModes: ("bot" | "agent")[] = [];

        // 4. 配置 Bot
        if (mode === "bot" || mode === "both") {
            next = await configureBotMode(next, prompter, accountId);
            configuredModes.push("bot");
        }

        // 5. 配置 Agent
        if (mode === "agent" || mode === "both") {
            next = await configureAgentMode(next, prompter, accountId);
            configuredModes.push("agent");
        }

        // 6. DM 策略
        next = await promptDmPolicy(next, prompter, configuredModes, accountId);

        // 7. 设置 defaultAccount
        next = setWecomDefaultAccount(next, accountId);

        // 8. 启用通道
        next = setWecomEnabled(next, true);

        // 9. 设置 gateway.bind 为 lan（允许外部访问回调）
        next = setGatewayBindLan(next);

        // 10. 确保 bindings 中有默认路由（onboard quickstart 不会提示绑定）
        next = ensureWecomBinding(next, accountId);

        // 11. 汇总
        await showSummary(next, prompter, accountId);

        return { cfg: next, accountId };
    },
};
