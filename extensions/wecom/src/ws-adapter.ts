/**
 * WeCom WebSocket 长链接模式适配器
 *
 * 职责：管理 WSClient 生命周期，将 SDK 事件桥接到现有 monitor.ts 消息管线。
 *
 * SDK WsFrame 事件
 *   ↓
 * ws-adapter 转换为 WecomBotInboundMessage 格式
 *   ↓
 * 复用 monitor.ts 中的 shouldProcessBotInboundMessage → buildInboundBody
 *   → streamStore.addPendingMessage → flushPending 管线
 */

import type { OpenClawConfig, PluginRuntime } from "openclaw/plugin-sdk";
import { WSClient } from "@wecom/aibot-node-sdk";
import type {
    WsFrame,
    BaseMessage,
    TextMessage,
    ImageMessage,
    MixedMessage,
    VoiceMessage,
    FileMessage,
    EventMessage,
    EventMessageWith,
    ReplyMsgItem,
} from "@wecom/aibot-node-sdk";
import type { EnterChatEvent, TemplateCardEventData } from "@wecom/aibot-node-sdk";

import type { ResolvedBotAccount, WecomNetworkConfig, WecomBotInboundMessage } from "./types/index.js";
import type { WecomRuntimeEnv, WecomWebhookTarget, StreamState } from "./monitor/types.js";
import { shouldProcessBotInboundMessage, buildInboundBody } from "./monitor.js";
import { monitorState } from "./monitor/state.js";
import { getWecomRuntime } from "./runtime.js";
import { fetchAndSaveMcpConfig } from "./mcp-config.js";

// ─── Constants ─────────────────────────────────────────────────────────

/** "思考中" 占位消息，让用户立即看到机器人正在响应 */
const THINKING_MESSAGE = "<think></think>";

// ─── WSClient Instance Registry ────────────────────────────────────────

const wsClients = new Map<string, WSClient>();

/**
 * 获取指定账号的 WSClient 实例
 */
export function getWsClient(accountId: string): WSClient | undefined {
    return wsClients.get(accountId);
}

/**
 * 等待 WSClient 连接就绪，最多等待 timeoutMs 毫秒（默认 30 秒）。
 * 如果已连接则立即返回；如果 client 尚未创建，会轮询等待创建后再监听连接事件。
 */
export async function waitForWsConnection(accountId: string, timeoutMs = 30_000): Promise<boolean> {
    const deadline = Date.now() + timeoutMs;

    // 等待 client 实例出现（gateway 重启时 client 可能还没注册）
    while (!wsClients.has(accountId)) {
        if (Date.now() >= deadline) return false;
        await new Promise((r) => setTimeout(r, 500));
    }

    const client = wsClients.get(accountId)!;
    if (client.isConnected) return true;

    const remaining = deadline - Date.now();
    if (remaining <= 0) return false;

    return new Promise<boolean>((resolve) => {
        const timer = setTimeout(() => {
            cleanup();
            resolve(false);
        }, remaining);

        const onConnected = () => {
            cleanup();
            resolve(true);
        };

        const cleanup = () => {
            clearTimeout(timer);
            client.off("connected", onConnected);
        };

        client.on("connected", onConnected);
        // 再检查一次，防止在注册监听器之前已连上
        if (client.isConnected) {
            cleanup();
            resolve(true);
        }
    });
}

// ─── Stream Reply Watcher ──────────────────────────────────────────────

/**
 * 流式回复监听器：轮询 StreamState 变化并通过 WSClient 推送回复
 */
function watchStreamReply(params: {
    wsClient: WSClient;
    frame: WsFrame;
    streamId: string;
    log?: (msg: string) => void;
    error?: (msg: string) => void;
}): void {
    const { wsClient, frame, streamId, log, error } = params;
    const streamStore = monitorState.streamStore;
    let lastSentContent = "";
    let finished = false;
    const POLL_INTERVAL_MS = 200;

    const tick = async () => {
        if (finished) return;

        const state = streamStore.getStream(streamId);
        if (!state) {
            finished = true;
            return;
        }

        const content = state.content ?? "";
        const isFinished = state.finished ?? false;

        // 有新内容或流结束时发送
        if (content !== lastSentContent || isFinished) {
            try {
                // 构建图片附件（仅在结束时）
                let msgItems: ReplyMsgItem[] | undefined;
                if (isFinished && state.images?.length) {
                    msgItems = state.images.map((img) => ({
                        msgtype: "image" as const,
                        image: { base64: img.base64, md5: img.md5 },
                    }));
                }

                await wsClient.replyStream(
                    frame,
                    streamId,
                    content,
                    isFinished,
                    msgItems,
                );
                lastSentContent = content;
                log?.(`ws-reply: streamId=${streamId} len=${content.length} finish=${isFinished}`);
            } catch (err) {
                error?.(`ws-reply: replyStream failed streamId=${streamId}: ${String(err)}`);
            }
        }

        if (isFinished) {
            finished = true;
            return;
        }

        setTimeout(tick, POLL_INTERVAL_MS);
    };

    // 初次延迟启动，等待 agent 开始生产内容
    setTimeout(tick, POLL_INTERVAL_MS);
}

// ─── SDK Message → WecomBotInboundMessage Conversion ───────────────────

/**
 * 将 SDK 的 WsFrame<BaseMessage> 转换为现有的 WecomBotInboundMessage 格式
 */
function convertSdkMessageToInbound(body: BaseMessage): WecomBotInboundMessage {
    const base: WecomBotInboundMessage = {
        msgid: body.msgid,
        aibotid: body.aibotid,
        chattype: body.chattype,
        chatid: body.chatid,
        response_url: body.response_url,
        from: body.from ? { userid: body.from.userid } : undefined,
        msgtype: body.msgtype as string,
    };

    const msgtype = String(body.msgtype ?? "").toLowerCase();

    if (msgtype === "text") {
        const textBody = body as TextMessage;
        return { ...base, msgtype: "text", text: textBody.text, quote: textBody.quote as any };
    }
    if (msgtype === "voice") {
        const voiceBody = body as VoiceMessage;
        return { ...base, msgtype: "voice", voice: voiceBody.voice, quote: voiceBody.quote as any };
    }
    if (msgtype === "image") {
        const imageBody = body as ImageMessage;
        return { ...base, msgtype: "image" as any, image: imageBody.image, quote: imageBody.quote as any } as any;
    }
    if (msgtype === "file") {
        const fileBody = body as FileMessage;
        return { ...base, msgtype: "file" as any, file: fileBody.file, quote: fileBody.quote as any } as any;
    }
    if (msgtype === "video") {
        // SDK 没有导出 VideoMessage 类型，直接从 BaseMessage 取 video 字段
        return { ...base, msgtype: "video" as any, video: (body as any).video, quote: (body as any).quote } as any;
    }
    if (msgtype === "mixed") {
        const mixedBody = body as MixedMessage;
        return { ...base, msgtype: "mixed" as any, mixed: mixedBody.mixed, quote: mixedBody.quote as any } as any;
    }

    // Fallback: pass through as-is
    return { ...base, ...body };
}

// ─── WS Event Handlers ────────────────────────────────────────────────

function setupMessageHandler(params: {
    wsClient: WSClient;
    accountId: string;
    target: WecomWebhookTarget;
}) {
    const { wsClient, accountId, target } = params;
    const streamStore = monitorState.streamStore;

    // 监听所有消息类型
    wsClient.on("message", (frame: WsFrame<BaseMessage>) => {
        const body = frame.body;
        if (!body) return;

        const msgtype = String(body.msgtype ?? "").toLowerCase();

        // event 类型由专门的 event handler 处理
        if (msgtype === "event") return;

        const msg = convertSdkMessageToInbound(body);
        const decision = shouldProcessBotInboundMessage(msg);
        if (!decision.shouldProcess) {
            target.runtime.log?.(
                `[${accountId}] ws-inbound: skipped msgtype=${msgtype} reason=${decision.reason}`,
            );
            return;
        }

        const userid = decision.senderUserId!;
        const chatId = decision.chatId ?? userid;
        const conversationKey = `wecom:${accountId}:${userid}:${chatId}`;
        const msgContent = buildInboundBody(msg);

        target.runtime.log?.(
            `[${accountId}] ws-inbound: msgtype=${msgtype} chattype=${String(msg.chattype ?? "")} ` +
            `from=${userid} msgid=${String(msg.msgid ?? "")}`,
        );

        // 消息去重
        if (msg.msgid) {
            const existingStreamId = streamStore.getStreamByMsgId(String(msg.msgid));
            if (existingStreamId) {
                target.runtime.log?.(
                    `[${accountId}] ws-inbound: duplicate msgid=${msg.msgid}, skipping`,
                );
                return;
            }
        }

        // 加入 Pending 队列（复用现有防抖/聚合逻辑）
        const { streamId } = streamStore.addPendingMessage({
            conversationKey,
            target,
            msg,
            msgContent,
            nonce: "",
            timestamp: String(Date.now()),
            debounceMs: (target.account.config as any).debounceMs,
        });

        // 标记 wsMode
        streamStore.updateStream(streamId, (s: StreamState) => {
            s.wsMode = true;
        });

        // 立即发送"思考中"占位消息，让用户看到即时反馈
        const sendThinking = (target.account.config as any).sendThinkingMessage ?? true;
        if (sendThinking) {
            wsClient.replyStream(frame, streamId, THINKING_MESSAGE, false).catch((err) => {
                target.runtime.error?.(
                    `[${accountId}] ws-thinking: failed to send thinking message: ${String(err)}`,
                );
            });
        }

        // 注册流式回复监听器
        watchStreamReply({
            wsClient,
            frame,
            streamId,
            log: (msg) => target.runtime.log?.(`[${accountId}] ${msg}`),
            error: (msg) => target.runtime.error?.(`[${accountId}] ${msg}`),
        });

        target.statusSink?.({ lastInboundAt: Date.now() });
    });
}

function setupEventHandler(params: {
    wsClient: WSClient;
    accountId: string;
    target: WecomWebhookTarget;
    welcomeText?: string;
}) {
    const { wsClient, accountId, target, welcomeText } = params;
    const streamStore = monitorState.streamStore;

    // 进入会话事件 → 欢迎语
    wsClient.on("event.enter_chat", async (frame: WsFrame<EventMessageWith<EnterChatEvent>>) => {
        const text = welcomeText?.trim();
        if (!text) return;

        try {
            await wsClient.replyWelcome(frame, {
                msgtype: "text",
                text: { content: text },
            });
            target.runtime.log?.(`[${accountId}] ws-event: sent welcome text`);
        } catch (err) {
            target.runtime.error?.(`[${accountId}] ws-event: replyWelcome failed: ${String(err)}`);
        }
    });

    // 模板卡片交互事件 → 转换为文本消息注入管线
    wsClient.on("event.template_card_event", (frame: WsFrame<EventMessageWith<TemplateCardEventData>>) => {
        const body = frame.body;
        if (!body) return;

        const eventData = body.event;
        let interactionDesc = `[卡片交互] 按钮: ${eventData?.event_key || "unknown"}`;
        if (eventData?.task_id) interactionDesc += ` (任务ID: ${eventData.task_id})`;

        const msgid = body.msgid ? String(body.msgid) : undefined;

        // 去重
        if (msgid && streamStore.getStreamByMsgId(msgid)) {
            target.runtime.log?.(`[${accountId}] ws-event: template_card_event already processed msgid=${msgid}`);
            return;
        }

        const streamId = streamStore.createStream({ msgid });
        streamStore.markStarted(streamId);
        streamStore.updateStream(streamId, (s: StreamState) => {
            s.wsMode = true;
        });

        const syntheticMsg: WecomBotInboundMessage = {
            msgid,
            aibotid: body.aibotid,
            chattype: body.chattype,
            chatid: body.chatid,
            from: body.from ? { userid: body.from.userid } : undefined,
            msgtype: "text",
            text: { content: interactionDesc },
        };

        let core: PluginRuntime;
        try {
            core = getWecomRuntime();
        } catch {
            target.runtime.error?.(`[${accountId}] ws-event: runtime not ready for template_card_event`);
            streamStore.markFinished(streamId);
            return;
        }

        // 由于卡片事件没有经过防抖队列，直接触发 flushPending 的等效操作
        // 需要通过 addPendingMessage 注入，让现有管线处理
        const userid = body.from?.userid ?? "unknown";
        const chatId = body.chatid ?? userid;
        const conversationKey = `wecom:${accountId}:${userid}:${chatId}`;

        // 先清除之前创建的 stream（addPendingMessage 会创建新的）
        // 直接用 addPendingMessage 复用完整管线
        const enrichedTarget: WecomWebhookTarget = { ...target, core };
        const { streamId: actualStreamId } = streamStore.addPendingMessage({
            conversationKey,
            target: enrichedTarget,
            msg: syntheticMsg,
            msgContent: interactionDesc,
            nonce: "",
            timestamp: String(Date.now()),
            debounceMs: 0, // 卡片事件不防抖
        });

        streamStore.updateStream(actualStreamId, (s: StreamState) => {
            s.wsMode = true;
        });

        watchStreamReply({
            wsClient,
            frame,
            streamId: actualStreamId,
            log: (msg) => target.runtime.log?.(`[${accountId}] ${msg}`),
            error: (msg) => target.runtime.error?.(`[${accountId}] ${msg}`),
        });
    });

    // 反馈事件 → 仅记录日志
    wsClient.on("event.feedback_event", (frame) => {
        target.runtime.log?.(
            `[${accountId}] ws-event: feedback_event received (logged only)`,
        );
    });
}

// ─── WSClient Lifecycle ────────────────────────────────────────────────

export type StartWsClientParams = {
    accountId: string;
    botId: string;
    secret: string;
    account: ResolvedBotAccount;
    config: OpenClawConfig;
    runtime: WecomRuntimeEnv;
    core: PluginRuntime;
    statusSink?: (patch: { lastInboundAt?: number; lastOutboundAt?: number }) => void;
    welcomeText?: string;
    network?: WecomNetworkConfig;
};

/**
 * 启动 WebSocket 长链接客户端
 * @returns cleanup 函数（用于注销）
 */
export function startWsClient(params: StartWsClientParams): () => void {
    const {
        accountId, botId, secret,
        account, config, runtime, core,
        statusSink, welcomeText,
    } = params;

    // 如果已有实例，先停止
    stopWsClient(accountId);

    const wsClient = new WSClient({
        botId,
        secret,
        maxReconnectAttempts: -1, // 无限重连
        logger: {
            debug: (msg: string) => runtime.log?.(`[${accountId}][ws-sdk] ${msg}`),
            info: (msg: string) => runtime.log?.(`[${accountId}][ws-sdk] ${msg}`),
            warn: (msg: string) => runtime.log?.(`[${accountId}][ws-sdk] WARN: ${msg}`),
            error: (msg: string) => runtime.error?.(`[${accountId}][ws-sdk] ERROR: ${msg}`),
        },
    });

    wsClients.set(accountId, wsClient);

    // 构建 WecomWebhookTarget 以复用 monitor 管线
    const target: WecomWebhookTarget = {
        account,
        config,
        runtime,
        core,
        path: `ws://${accountId}`,
        statusSink,
    };

    // 设置消息和事件处理
    setupMessageHandler({ wsClient, accountId, target });
    setupEventHandler({ wsClient, accountId, target, welcomeText });

    // 连接状态日志
    wsClient.on("connected", () => {
        runtime.log?.(`[${accountId}] ws: connected`);
    });
    wsClient.on("authenticated", () => {
        runtime.log?.(`[${accountId}] ws: authenticated successfully`);
        // 认证成功后拉取 MCP 配置（非阻塞，失败仅记日志）
        void fetchAndSaveMcpConfig(wsClient, accountId, runtime);
    });
    wsClient.on("disconnected", (reason: string) => {
        runtime.log?.(`[${accountId}] ws: disconnected - ${reason}`);
    });
    wsClient.on("reconnecting", (attempt: number) => {
        runtime.log?.(`[${accountId}] ws: reconnecting attempt=${attempt}`);
    });
    wsClient.on("error", (err: Error) => {
        runtime.error?.(`[${accountId}] ws: error - ${err.message}`);
    });

    // 建立连接
    wsClient.connect();
    runtime.log?.(`[${accountId}] ws: starting connection (botId=${botId})`);

    // 返回清理函数
    return () => {
        stopWsClient(accountId);
    };
}

/**
 * 停止指定账号的 WSClient
 */
export function stopWsClient(accountId: string): void {
    const existing = wsClients.get(accountId);
    if (existing) {
        existing.disconnect();
        wsClients.delete(accountId);
    }
}
