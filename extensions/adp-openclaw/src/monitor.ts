// Monitor: WebSocket connection to Go server for real-time message handling
// Supports: API Token auth, conversation tracking for multi-turn dialogues

import type { PluginLogger, ClawdbotConfig } from "openclaw/plugin-sdk";
import { getAdpOpenclawRuntime, setActiveWebSocket } from "./runtime.js";
import {
  getChatHistory,
  getMergedChatHistory,
  listSessions,
  resolveSessionKey,
  type ChatHistoryResponse,
  type SessionsListResponse,
} from "./session-history.js";
import {
  ADP_UPLOAD_TOOL_NAME,
  ADP_UPLOAD_TOOL_SCHEMA,
  executeAdpUploadTool,
  uploadResultEmitter,
  UPLOAD_RESULT_EVENT,
  type AdpUploadToolResult,
} from "./adp-upload-tool.js";
import {
  formatUploadResultForUser,
  formatUploadResultAsMarkdown,
} from "./tool-result-message-blocks.js";
import crypto from "crypto";
import fs from "fs";
// @ts-ignore - import JSON file
import packageJson from "../package.json" with { type: "json" };

// Plugin version from package.json
const PLUGIN_VERSION = packageJson.version;

// WebSocket reconnect delay (fixed at 1 second)
const RECONNECT_DELAY_MS = 1000;

export type MonitorParams = {
  wsUrl: string; // WebSocket URL (direct, no conversion needed)
  clientToken: string;
  signKey?: string; // HMAC key for signature generation
  abortSignal?: AbortSignal;
  log?: PluginLogger;
  cfg?: ClawdbotConfig; // OpenClaw config for model settings
  setStatus?: (next: Record<string, unknown>) => void; // SDK health-monitor status reporter
};

// WebSocket message types
const MsgType = {
  Auth: "auth",
  AuthResult: "auth_result",
  Ping: "ping",
  Pong: "pong",
  Inbound: "inbound",
  Outbound: "outbound",
  OutboundChunk: "outbound_chunk",
  OutboundEnd: "outbound_end",
  Ack: "ack",
  Error: "error",
  // OpenClaw session history: GoServer triggers client to fetch from OpenClaw Gateway
  ConvHistory: "conv_history",           // Request OpenClaw chat history
  ConvResponse: "conv_response",         // OpenClaw chat history response
  // OpenClaw sessions list
  FetchOpenClawSessions: "fetch_openclaw_sessions",
  OpenClawSessionsResponse: "openclaw_sessions_response",
  // Cancel generation
  Cancel: "cancel",                        // Server → Client: cancel ongoing generation
} as const;

type WSMessage = {
  type: string;
  requestId?: string;
  payload?: unknown;
  timestamp: number;
};

// UserInfo represents full user identity (matching Go server's UserInfo)
type UserInfo = {
  userId: string;
  username?: string;
  avatar?: string;
  email?: string;
  tenantId?: string;
  source?: string;
  extra?: Record<string, string>;
};

type InboundMessage = {
  id: string;
  conversationId: string;
  recordId?: string; // Record ID from server for tracking message pairs
  clientId: string;
  from: string;
  to: string;
  text: string;
  timestamp: number;
  user?: UserInfo; // Full user identity information
};

type AuthResultPayload = {
  success: boolean;
  clientId?: string;
  message?: string;
};

// Generate HMAC-SHA256 signature for authentication (includes timestamp for anti-replay)
// Uses signKey as the HMAC key, and "token:nonce:timestamp" as the message
function generateSignature(signKey: string, token: string, nonce: string, timestamp: number): string {
  // Use HMAC-SHA256 with signKey as the key, and "token:nonce:timestamp" as the message
  return crypto.createHmac("sha256", signKey).update(`${token}:${nonce}:${timestamp}`).digest("hex");
}

// Generate random nonce
function generateNonce(): string {
  return crypto.randomBytes(16).toString("hex");
}

// Generate unique request ID
function generateRequestId(): string {
  return `req-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

/**
 * Mark a session's abortedLastRun flag in the sessions store.
 * This tells the SDK to inject an "abort hint" on the next message,
 * preventing the AI from resuming the cancelled task.
 */
async function markSessionAborted(params: {
  sessionKey: string;
  runtime: ReturnType<typeof getAdpOpenclawRuntime>;
  cfg?: ClawdbotConfig;
  log?: PluginLogger;
}): Promise<void> {
  const { sessionKey, runtime, cfg, log } = params;
  try {
    // Use SDK's resolveStorePath to find the sessions.json location
    const storePath = runtime.channel.session.resolveStorePath(cfg?.session?.store);
    if (!storePath || !fs.existsSync(storePath)) {
      log?.warn?.(`[adp-openclaw] Cannot mark session aborted: store not found at ${storePath}`);
      return;
    }

    const raw = fs.readFileSync(storePath, "utf-8");
    const store = JSON.parse(raw) as Record<string, { abortedLastRun?: boolean; updatedAt?: number; [key: string]: unknown }>;

    // Try both the raw sessionKey and the "agent:main:{sessionKey}" variant
    const candidates = [sessionKey, `agent:main:${sessionKey}`];
    let matchedKey: string | undefined;
    for (const key of candidates) {
      if (store[key]) {
        matchedKey = key;
        break;
      }
    }

    if (!matchedKey) {
      log?.info?.(`[adp-openclaw] Session key not found in store for abort marking: ${sessionKey}`);
      return;
    }

    store[matchedKey].abortedLastRun = true;
    store[matchedKey].updatedAt = Date.now();
    fs.writeFileSync(storePath, JSON.stringify(store, null, 2), "utf-8");
    log?.info?.(`[adp-openclaw] Marked session ${matchedKey} as abortedLastRun=true`);
  } catch (err) {
    log?.error?.(`[adp-openclaw] Failed to mark session aborted: ${err}`);
  }
}

export async function monitorAdpOpenclaw(params: MonitorParams): Promise<void> {
  const { wsUrl, clientToken, signKey, abortSignal, log, cfg, setStatus } = params;
  const runtime = getAdpOpenclawRuntime();

  log?.info(`[adp-openclaw] WebSocket monitor started, connecting to ${wsUrl}`);

  while (!abortSignal?.aborted) {
    try {
      await connectAndHandle({
        wsUrl,
        clientToken,
        signKey,
        abortSignal,
        log,
        runtime,
        cfg,
        setStatus,
      });
    } catch (err) {
      if (abortSignal?.aborted) break;
      log?.error(`[adp-openclaw] WebSocket error: ${err}`);
    }

    // Wait before reconnecting
    if (!abortSignal?.aborted) {
      log?.info(`[adp-openclaw] Reconnecting in ${RECONNECT_DELAY_MS}ms...`);
      await sleep(RECONNECT_DELAY_MS, abortSignal);
    }
  }

  log?.info(`[adp-openclaw] WebSocket monitor stopped`);
}

type ConnectParams = {
  wsUrl: string;
  clientToken: string;
  signKey?: string;
  abortSignal?: AbortSignal;
  log?: PluginLogger;
  runtime: ReturnType<typeof getAdpOpenclawRuntime>;
  cfg?: ClawdbotConfig;
  setStatus?: (next: Record<string, unknown>) => void;
};

async function connectAndHandle(params: ConnectParams): Promise<void> {
  const { wsUrl, clientToken, signKey, abortSignal, log, runtime, cfg, setStatus } = params;

  // Dynamic import for WebSocket (works in both Node.js and browser)
  const WebSocket = (await import("ws")).default;

  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    let authenticated = false;
    let pingInterval: NodeJS.Timeout | null = null;

    // Track active generations for cancel support (keyed by conversationId)
    const activeGenerations = new Map<string, AbortController>();

    // Handle abort signal
    const abortHandler = () => {
      ws.close();
      resolve();
    };
    abortSignal?.addEventListener("abort", abortHandler);

    ws.on("open", () => {
      log?.info(`[adp-openclaw] WebSocket connected, authenticating...`);
      
      // 设置 TCP keepalive（作为额外保障）
      const socket = (ws as any)._socket;
      if (socket && typeof socket.setKeepAlive === 'function') {
        socket.setKeepAlive(true, 30000);  // 30秒
        log?.info(`[adp-openclaw] TCP keepalive enabled`);
      }
      
      // Save active WebSocket for outbound messaging
      setActiveWebSocket(ws);

      // Send authentication message with signature (includes timestamp for anti-replay)
      const nonce = generateNonce();
      const timestamp = Date.now();
      // Generate signature only if signKey is provided
      const signature = signKey ? generateSignature(signKey, clientToken, nonce, timestamp) : "";

      const authMsg: WSMessage = {
        type: MsgType.Auth,
        requestId: generateRequestId(),
        payload: {
          token: clientToken,
          nonce: signKey ? nonce : undefined,
          signature: signKey ? signature : undefined,
          timestamp: signKey ? timestamp : undefined, // Include timestamp in payload for server verification
        },
        timestamp: Date.now(),
      };

      ws.send(JSON.stringify(authMsg));
    });

    ws.on("message", async (data: Buffer) => {
      try {
        const msg: WSMessage = JSON.parse(data.toString());

        switch (msg.type) {
          case MsgType.AuthResult: {
            const result = msg.payload as AuthResultPayload;
            if (result.success) {
              authenticated = true;
              log?.info(`[adp-openclaw] Plugin v${PLUGIN_VERSION} authenticated as client ${result.clientId}`);

              // Report connected status to SDK health-monitor
              const now = Date.now();
              setStatus?.({ connected: true, lastConnectedAt: now, lastEventAt: now });

              // Start ping interval
              pingInterval = setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) {
                  ws.send(JSON.stringify({
                    type: MsgType.Ping,
                    requestId: generateRequestId(),
                    timestamp: Date.now(),
                  }));
                }
              }, 25000);
            } else {
              log?.error(`[adp-openclaw] Authentication failed: ${result.message}`);
              ws.close();
            }
            break;
          }

          case MsgType.Pong:
            // Heartbeat response — update lastEventAt so health-monitor knows connection is alive
            setStatus?.({ lastEventAt: Date.now() });
            break;

          case MsgType.Inbound: {
            if (!authenticated) break;

            // Report inbound event to SDK health-monitor
            const inboundAt = Date.now();
            setStatus?.({ lastEventAt: inboundAt, lastInboundAt: inboundAt });

            // Debug: log raw payload to verify recordId is received
            log?.info(`[adp-openclaw] Raw payload: ${JSON.stringify(msg.payload)}`);
            
            const inMsg = msg.payload as InboundMessage;
            log?.info(`[adp-openclaw] Received: ${inMsg.from}: ${inMsg.text} (conv=${inMsg.conversationId}, rec=${inMsg.recordId || 'none'}, user=${JSON.stringify(inMsg.user || {})})`);

            // Process the message with full user identity
            const convIdForCleanup = inMsg.conversationId || `fallback-${Date.now()}`;
            try {
              // Build user identity string for From field (like Feishu: "feishu:user_id")
              const userIdentifier = inMsg.user?.userId || inMsg.from;
              const tenantPrefix = inMsg.user?.tenantId ? `${inMsg.user.tenantId}:` : "";
              
              // Build metadata for user context (passed through to openclaw)
              const userMetadata: Record<string, string> = {};
              if (inMsg.user) {
                if (inMsg.user.username) userMetadata.username = inMsg.user.username;
                if (inMsg.user.email) userMetadata.email = inMsg.user.email;
                if (inMsg.user.avatar) userMetadata.avatar = inMsg.user.avatar;
                if (inMsg.user.tenantId) userMetadata.tenantId = inMsg.user.tenantId;
                if (inMsg.user.source) userMetadata.source = inMsg.user.source;
                if (inMsg.user.extra) {
                  Object.entries(inMsg.user.extra).forEach(([k, v]) => {
                    userMetadata[`extra_${k}`] = v;
                  });
                }
              }

              // Use resolveAgentRoute to get proper sessionKey (like QQBot does)
              // This ensures session history is correctly associated
              //
              // Use "per-channel-peer" so the session key includes the channel name:
              //   agent:main:adp-openclaw:direct:{userId}
              // This is critical for cron job delivery inference — inferDeliveryFromSessionKey()
              // parses the session key to extract channel and target user:
              //   channel = "adp-openclaw", to = "{userId}"
              // Without the channel in the key, cron delivery defaults to channel:"last" which
              // fails when multiple channels (adp-openclaw, wecom, etc.) are configured.
              //
              // We use the user identifier (not conversationId) as peerId because:
              // 1. The outbound delivery target is the userId (inMsg.from), not the conversationId
              // 2. inferDeliveryFromSessionKey extracts peerId as the delivery "to" field
              const peerId = userIdentifier;
              const route = runtime.channel.routing.resolveAgentRoute({
                cfg: {
                  ...(cfg ?? {}),
                  session: {
                    ...(cfg?.session ?? {}),
                    // Override dmScope to "per-channel-peer" so session key encodes channel + userId
                    // Session key format: agent:main:adp-openclaw:direct:{userId}
                    dmScope: "per-channel-peer",
                  },
                },
                channel: "adp-openclaw",
                accountId: "default",
                peer: {
                  kind: "dm",  // direct message
                  id: peerId,
                },
              });

              // Get envelope format options and messages config (like QQBot does)
              const envelopeOptions = runtime.channel.reply.resolveEnvelopeFormatOptions(cfg ?? {});
              const messagesConfig = runtime.channel.reply.resolveEffectiveMessagesConfig(cfg ?? {}, route.agentId);

              // Use formatInboundEnvelope to format the message body (like QQBot does)
              const formattedBody = runtime.channel.reply.formatInboundEnvelope({
                channel: "ADP-OpenClaw",
                from: inMsg.user?.username ?? inMsg.from,
                timestamp: inMsg.timestamp || Date.now(),
                body: inMsg.text,
                chatType: "direct",
                sender: {
                  id: userIdentifier,
                  name: inMsg.user?.username,
                },
                envelope: envelopeOptions,
              });

              // Build delivery target for cron jobs
              // Format: adp-openclaw:{userId} for direct delivery via this channel
              const cronDeliveryTarget = `${userIdentifier}`;
              const nowMs = Date.now();

              // Build BodyForAgent with context info for cron jobs (like QQBot does)
              // This tells AI how to set up cron job delivery parameters
              const contextInfo = `你正在通过 ADP-OpenClaw 与用户对话。

【会话上下文】
- 用户: ${inMsg.user?.username || inMsg.from} (${userIdentifier})
- 场景: 私聊
- 当前时间戳(ms): ${nowMs}
- 定时提醒投递地址: channel=adp-openclaw, to=${cronDeliveryTarget}`;

              // If message is a command (starts with /), don't inject context
              const agentBody = inMsg.text.startsWith("/")
                ? inMsg.text
                : `${contextInfo}\n\n${inMsg.text}`;

              const ctx = runtime.channel.reply.finalizeInboundContext({
                Body: formattedBody,
                BodyForAgent: agentBody, // AI sees this context with cron delivery info
                RawBody: inMsg.text,
                CommandBody: inMsg.text,
                // User identity: format as "adp-openclaw:{tenantId}:{userId}" for multi-tenant support
                From: `adp-openclaw:${tenantPrefix}${userIdentifier}`,
                To: `adp-openclaw:bot`,
                // SessionKey from resolveAgentRoute for proper session history tracking
                SessionKey: route.sessionKey,
                AccountId: route.accountId,
                ChatType: "direct",
                // SenderId carries the raw user ID for identification
                SenderId: userIdentifier,
                SenderName: inMsg.user?.username,
                Provider: "adp-openclaw",
                Surface: inMsg.user?.source || "adp-openclaw",
                MessageSid: inMsg.id,
                MessageSidFull: inMsg.id,
                Timestamp: inMsg.timestamp || Date.now(),
                OriginatingChannel: "adp-openclaw",
                OriginatingTo: "adp-openclaw:bot",
                // Authorize slash commands (/new, /status, /reset, etc.)
                // Without this, commands are silently dropped (deny-by-default)
                CommandAuthorized: true,
                // Pass user metadata through context (like Feishu does)
                ...userMetadata,
              });

              // Generate unique stream ID for this response
              const streamId = `stream-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
              let chunkIndex = 0;
              let lastPartialText = ""; // Track last sent text for delta calculation
              let finalSent = false; // Track if outbound_end has been sent
              const displayName = inMsg.user?.username || inMsg.from;

              // Per-message AbortController for cancel support
              const generationController = new AbortController();
              const convId = inMsg.conversationId || streamId;
              // Cancel any previous generation for the same conversation
              if (activeGenerations.has(convId)) {
                activeGenerations.get(convId)!.abort();
              }
              activeGenerations.set(convId, generationController);
              
              // 收集上传结果，在发送最终回复时追加完整的下载链接
              let pendingUploadResults: AdpUploadToolResult[] = [];
              
              // 监听上传结果事件
              const uploadResultHandler = (event: { toolCallId: string; result: AdpUploadToolResult }) => {
                log?.info(`[adp-openclaw] Received upload result event for toolCallId=${event.toolCallId}`);
                if (event.result.ok && event.result.files && event.result.files.length > 0) {
                  pendingUploadResults.push(event.result);
                  // 打印完整的下载链接
                  for (const file of event.result.files) {
                    log?.info(`[adp-openclaw] Upload result - file.downloadUrl: ${file.downloadUrl}`);
                  }
                }
              };
              uploadResultEmitter.on(UPLOAD_RESULT_EVENT, uploadResultHandler);

              // Helper function to send outbound_end message
              const sendOutboundEnd = (text: string) => {
                if (finalSent) return; // Prevent duplicate sends
                finalSent = true;
                
                // 移除事件监听
                uploadResultEmitter.off(UPLOAD_RESULT_EVENT, uploadResultHandler);
                
                // 如果有上传结果，修正 LLM 可能篡改的签名 URL，并追加完整下载链接
                let finalText = text;
                if (pendingUploadResults.length > 0) {
                  // Build a map: COS path (without query params) -> correct full URL
                  // LLM may reproduce the URL with subtle character errors in the
                  // q-signature (hex hallucination). We fix this by matching on the
                  // base path and replacing the entire URL with the correct one.
                  const correctUrlMap = new Map<string, string>();
                  for (const result of pendingUploadResults) {
                    for (const file of (result.files || [])) {
                      if (file.downloadUrl) {
                        try {
                          const basePath = file.downloadUrl.split("?")[0];
                          correctUrlMap.set(basePath, file.downloadUrl);
                        } catch { /* ignore */ }
                      }
                    }
                  }
                  
                  // Replace any LLM-generated URLs that share the same base path
                  // but may have corrupted query parameters (signature hallucination)
                  if (correctUrlMap.size > 0) {
                    for (const [basePath, correctUrl] of correctUrlMap) {
                      // Escape basePath for use in regex
                      const escaped = basePath.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
                      // Match the base path followed by any query string
                      const pattern = new RegExp(escaped + "\\?[^\\s)\\]]*", "g");
                      const before = finalText;
                      finalText = finalText.replace(pattern, correctUrl);
                      if (finalText !== before) {
                        log?.info(`[adp-openclaw] Fixed LLM-corrupted URL for: ${basePath.substring(basePath.lastIndexOf("/") + 1)}`);
                      }
                    }
                  }
                  
                  const uploadLinks: string[] = [];
                  for (const result of pendingUploadResults) {
                    for (const file of (result.files || [])) {
                      if (file.downloadUrl) {
                        // 使用完整的下载链接，包括签名参数
                        uploadLinks.push(`📎 [${file.name}](${file.downloadUrl})`);
                        log?.info(`[adp-openclaw] Appending download link: ${file.downloadUrl.substring(0, 100)}...`);
                      }
                    }
                  }
                  if (uploadLinks.length > 0) {
                    finalText += `\n\n**文件下载链接（24小时有效）：**\n${uploadLinks.join("\n")}`;
                    log?.info(`[adp-openclaw] Added ${uploadLinks.length} download links to final response`);
                  }
                }
                
                if (chunkIndex > 0) {
                  log?.info(`[adp-openclaw] Sending outbound_end to ${displayName}: ${finalText.slice(0, 50)}... (chunks=${chunkIndex})`);
                  const endMsg: WSMessage = {
                    type: MsgType.OutboundEnd,
                    requestId: generateRequestId(),
                    payload: {
                      to: inMsg.from,
                      text: finalText,
                      conversationId: inMsg.conversationId,
                      recordId: inMsg.recordId, // Pass recordId back to server
                      streamId: streamId,
                      totalChunks: chunkIndex,
                      user: inMsg.user,
                    },
                    timestamp: Date.now(),
                  };
                  ws.send(JSON.stringify(endMsg));
                } else {
                  // No streaming chunks were sent, send as regular outbound message
                  log?.info(`[adp-openclaw] Sending outbound to ${displayName}: ${finalText.slice(0, 50)}...`);
                  const outMsg: WSMessage = {
                    type: MsgType.Outbound,
                    requestId: generateRequestId(),
                    payload: {
                      to: inMsg.from,
                      text: finalText,
                      conversationId: inMsg.conversationId,
                      recordId: inMsg.recordId, // Pass recordId back to server
                      user: inMsg.user,
                    },
                    timestamp: Date.now(),
                  };
                  ws.send(JSON.stringify(outMsg));
                }
              };

              log?.info(`[adp-openclaw] Starting dispatchReplyWithBufferedBlockDispatcher for ${displayName}`);
              await runtime.channel.reply.dispatchReplyWithBufferedBlockDispatcher({
                ctx,
                cfg: cfg ?? {},
                // Enable block streaming for SSE support
                replyOptions: {
                  disableBlockStreaming: false, // Force enable block streaming
                  abortSignal: generationController.signal, // Per-message cancel support
                  // Use onPartialReply for real-time streaming (character-level)
                  // onPartialReply receives cumulative text, so we need to calculate delta
                  onPartialReply: async (payload: { text?: string }) => {
                    const fullText = payload.text || "";
                    if (!fullText) return;
                    
                    // Calculate delta (new text since last send)
                    let delta = fullText;
                    if (fullText.startsWith(lastPartialText)) {
                      delta = fullText.slice(lastPartialText.length);
                    } else {
                      // Text was reset or non-monotonic, send full text
                      log?.debug?.(`[adp-openclaw] Partial text reset, sending full text`);
                    }
                    
                    // Skip if no new content
                    if (!delta) return;
                    
                    lastPartialText = fullText;
                    
                    // Send delta as streaming chunk
                    log?.debug?.(`[adp-openclaw] Partial delta[${chunkIndex}] to ${displayName}: ${delta.slice(0, 30)}...`);
                    
                    const chunkMsg: WSMessage = {
                      type: MsgType.OutboundChunk,
                      requestId: generateRequestId(),
                      payload: {
                        to: inMsg.from,
                        chunk: delta, // Send only the new delta, not cumulative
                        conversationId: inMsg.conversationId,
                        recordId: inMsg.recordId, // Pass recordId back to server
                        streamId: streamId,
                        index: chunkIndex,
                        isPartial: true, // Mark as incremental delta
                        user: inMsg.user,
                      },
                      timestamp: Date.now(),
                    };
                    
                    ws.send(JSON.stringify(chunkMsg));
                    chunkIndex++;
                  },
                },
                dispatcherOptions: {
                  // ⭐ Add responsePrefix from messagesConfig (like QQBot does)
                  // This tells the AI what tools are available
                  responsePrefix: messagesConfig.responsePrefix,
                  // Unified deliver callback - handles both streaming blocks and final reply
                  // SDK calls this with info.kind = "block" for streaming chunks, "final" for complete response
                  deliver: async (payload: { text?: string; toolName?: string; toolArgs?: unknown; toolCallId?: string; toolResult?: unknown }, info?: { kind?: string }) => {
                    const text = payload.text || "";
                    const kind = info?.kind;
                    
                    // Debug log for all deliver calls - log the actual info object
                    log?.info(`[adp-openclaw] deliver called: kind=${kind}, text.length=${text.length}, toolName=${payload.toolName || 'none'}, info=${JSON.stringify(info)}`);
                    
                    // Handle streaming block - IGNORE because handlePartial already sent deltas
                    // The "block" callback contains cumulative text (same as final), not incremental delta
                    // Sending it would cause duplicate data on the server side
                    if (kind === "block") {
                      log?.debug?.(`[adp-openclaw] Ignoring block callback (handlePartial already sent deltas), text.length=${text.length}`);
                      return;
                    }

                    // Handle tool result - check if it's adp_upload_file tool and send file links to user
                    if (kind === "tool") {
                      const toolName = payload.toolName;
                      const toolResult = payload.toolResult;
                      
                      log?.info(`[adp-openclaw] Tool result received: toolName=${toolName}, result=${JSON.stringify(toolResult)?.slice(0, 200)}`);
                      
                      // If it's our upload tool and it succeeded, send file links to user
                      if (toolName === ADP_UPLOAD_TOOL_NAME && toolResult && typeof toolResult === "object") {
                        const result = toolResult as AdpUploadToolResult;
                        if (result.ok && result.files && result.files.length > 0) {
                          // Debug: print full downloadUrl before formatting
                          for (const file of result.files) {
                            log?.info(`[adp-openclaw] File downloadUrl (full): ${file.downloadUrl}`);
                          }
                          
                          // Format upload result as user-readable message
                          const uploadMessage = formatUploadResultAsMarkdown(result);
                          
                          log?.info(`[adp-openclaw] Sending upload result to user: ${uploadMessage.slice(0, 100)}...`);
                          
                          // Send the file links as a message chunk
                          const chunkMsg: WSMessage = {
                            type: MsgType.OutboundChunk,
                            requestId: generateRequestId(),
                            payload: {
                              to: inMsg.from,
                              chunk: `\n\n${uploadMessage}\n\n`,
                              conversationId: inMsg.conversationId,
                              recordId: inMsg.recordId,
                              streamId: streamId,
                              index: chunkIndex,
                              isPartial: true,
                              user: inMsg.user,
                              isFileUpload: true, // Mark as file upload result
                            },
                            timestamp: Date.now(),
                          };
                          
                          ws.send(JSON.stringify(chunkMsg));
                          chunkIndex++;
                          
                          // Update lastPartialText to include the upload message
                          lastPartialText += `\n\n${uploadMessage}\n\n`;
                        }
                      }
                      return;
                    }

                    // Handle final reply or undefined kind - send outbound_end
                    // SDK may call deliver without kind when streaming ends
                    if (kind === "final" || kind === undefined) {
                      log?.info(`[adp-openclaw] deliver triggering sendOutboundEnd (kind=${kind})`);
                      log?.info(`[adp-openclaw] Final text content: ${text}`);
                      sendOutboundEnd(text || lastPartialText);
                    }
                  },
                  onError: (err: Error) => {
                    log?.error(`[adp-openclaw] Reply error: ${err.message}`);
                  },
                },
              });
              
              log?.info(`[adp-openclaw] dispatchReplyWithBufferedBlockDispatcher returned (finalSent=${finalSent}, chunkIndex=${chunkIndex})`);
              
              // Clean up active generation tracking
              activeGenerations.delete(convId);

              // If generation was cancelled (aborted), send outbound_end with partial text
              if (generationController.signal.aborted && !finalSent) {
                const cancelText = lastPartialText ? `${lastPartialText}\n\n[已停止生成]` : "[已停止生成]";
                log?.info(`[adp-openclaw] Generation cancelled, sending outbound_end with partial text`);
                sendOutboundEnd(cancelText);

                // Mark the session as aborted so the SDK injects an "abort hint"
                // on the next message, preventing the AI from resuming the cancelled task.
                // This aligns with openclaw's behavior: keep all transcript history intact,
                // only set abortedLastRun=true so next turn gets "Resume carefully" hint.
                await markSessionAborted({
                  sessionKey: route.sessionKey,
                  runtime,
                  cfg,
                  log,
                });
              }

              // IMPORTANT: After dispatchReplyWithBufferedBlockDispatcher completes,
              // ensure outbound_end is sent even if "final" deliver was not called.
              // This handles cases where the SDK only sends blocks without a final callback.
              if (!finalSent && chunkIndex > 0) {
                // Use the last accumulated partial text as the final text
                const finalText = lastPartialText || "";
                log?.info(`[adp-openclaw] dispatchReply completed without final, sending outbound_end (chunks=${chunkIndex})`);
                sendOutboundEnd(finalText);
              }
            } catch (err) {
              // Clean up on error
              activeGenerations.delete(convIdForCleanup);
              log?.error(`[adp-openclaw] Failed to process message: ${err}`);
            }
            break;
          }

          case MsgType.Ack:
            // Message acknowledgment
            log?.debug?.(`[adp-openclaw] Message acknowledged: ${msg.requestId}`);
            break;

          case MsgType.Error: {
            const error = msg.payload as { error: string; message: string };
            log?.error(`[adp-openclaw] Server error: ${error.error} - ${error.message}`);
            break;
          }

          // Handle cancel generation request from server (user clicked "stop generating")
          case MsgType.Cancel: {
            if (!authenticated) break;
            const cancelPayload = msg.payload as { conversationId?: string; streamId?: string };
            const cancelConvId = cancelPayload.conversationId || cancelPayload.streamId;
            log?.info(`[adp-openclaw] Received cancel request for conv=${cancelConvId}`);
            
            if (cancelConvId && activeGenerations.has(cancelConvId)) {
              activeGenerations.get(cancelConvId)!.abort();
              log?.info(`[adp-openclaw] Generation cancelled for conv=${cancelConvId}`);
            } else {
              // If no specific convId, cancel all active generations
              if (!cancelConvId && activeGenerations.size > 0) {
                for (const [id, controller] of activeGenerations) {
                  controller.abort();
                  log?.info(`[adp-openclaw] Generation cancelled for conv=${id} (cancel-all)`);
                }
              } else {
                log?.warn(`[adp-openclaw] No active generation found for conv=${cancelConvId}`);
              }
            }
            break;
          }

          // Handle fetch OpenClaw chat history request from GoServer
          case MsgType.ConvHistory: {
            if (!authenticated) break;

            const historyPayload = msg.payload as {
              sessionKey?: string;
              conversationId?: string;
              limit?: number;
            };

            log?.info(`[adp-openclaw] Received conv_history request: sessionKey=${historyPayload.sessionKey}, conversationId=${historyPayload.conversationId}`);

            try {
              const limit = historyPayload.limit ?? 200;
              
              let result: ChatHistoryResponse;
              
              // Check if sessionKey is in old format: agent:main:direct:{id}
              // We need to extract the id and merge old + new format histories
              const oldFormatMatch = historyPayload.sessionKey?.match(/^agent:main:direct:(.+)$/);
              
              if (oldFormatMatch) {
                // Old format sessionKey detected, need to merge with new format
                // Old: agent:main:direct:{conversationId}
                // New: agent:main:adp-openclaw:direct:{conversationId}
                const extractedId = oldFormatMatch[1];
                log?.info(`[adp-openclaw] Old format sessionKey detected, extracting id: ${extractedId}`);
                log?.info(`[adp-openclaw] Merging old and new session histories for id: ${extractedId}`);
                result = await getMergedChatHistory(extractedId, {
                  limit,
                  log,
                });
              } else if (historyPayload.sessionKey) {
                // Non-old-format sessionKey, use directly
                log?.info(`[adp-openclaw] Using provided sessionKey: ${historyPayload.sessionKey}`);
                result = await getChatHistory(historyPayload.sessionKey, {
                  limit,
                  log,
                });
              } else if (historyPayload.conversationId) {
                // If conversationId is provided, merge old and new session histories
                // Old format: agent:main:direct:{conversationId}
                // New format: agent:main:adp-openclaw:direct:{conversationId}
                log?.info(`[adp-openclaw] Merging old and new session histories for conversationId: ${historyPayload.conversationId}`);
                result = await getMergedChatHistory(historyPayload.conversationId, {
                  limit,
                  log,
                });
              } else {
                // Default to "main" session
                log?.info(`[adp-openclaw] Using default session: main`);
                result = await getChatHistory("main", {
                  limit,
                  log,
                });
              }

              log?.info(`[adp-openclaw] Sending conv_response: ${result.messages.length} messages (backend=cli)`);

              // Send response back to GoServer
              const responseMsg: WSMessage = {
                type: MsgType.ConvResponse,
                requestId: msg.requestId,
                payload: result as unknown as Record<string, unknown>,
                timestamp: Date.now(),
              };
              ws.send(JSON.stringify(responseMsg));
            } catch (err) {
              log?.error(`[adp-openclaw] Failed to fetch OpenClaw history: ${err}`);
              const errorMsg: WSMessage = {
                type: MsgType.ConvResponse,
                requestId: msg.requestId,
                payload: {
                  error: true,
                  message: String(err),
                  sessionKey: historyPayload.sessionKey || historyPayload.conversationId || "main",
                  messages: [],
                } as unknown as Record<string, unknown>,
                timestamp: Date.now(),
              };
              ws.send(JSON.stringify(errorMsg));
            }
            break;
          }

          // Handle fetch OpenClaw sessions list request from GoServer
          case MsgType.FetchOpenClawSessions: {
            if (!authenticated) break;

            const sessionsPayload = msg.payload as {
              limit?: number;
              includeLastMessage?: boolean;
              backend?: "file" | "cli" | "auto";
            };

            const backend = sessionsPayload.backend || "auto";

            log?.info(`[adp-openclaw] Received fetch_openclaw_sessions request: limit=${sessionsPayload.limit}, backend=${backend}`);

            try {
              // Use unified listSessions with backend selection
              const result: SessionsListResponse = await listSessions({
                limit: sessionsPayload.limit ?? 100,
                includeLastMessage: sessionsPayload.includeLastMessage ?? true,
                log,
                backend,
              });

              log?.info(`[adp-openclaw] Sending openclaw_sessions_response: ${result.sessions.length} sessions (backend=${backend})`);

              // Send response back to GoServer
              const responseMsg: WSMessage = {
                type: MsgType.OpenClawSessionsResponse,
                requestId: msg.requestId,
                payload: {
                  ...result as unknown as Record<string, unknown>,
                  backend,
                },
                timestamp: Date.now(),
              };
              ws.send(JSON.stringify(responseMsg));
            } catch (err) {
              log?.error(`[adp-openclaw] Failed to list OpenClaw sessions: ${err}`);
              const errorMsg: WSMessage = {
                type: MsgType.OpenClawSessionsResponse,
                requestId: msg.requestId,
                payload: {
                  error: true,
                  message: String(err),
                  sessions: [],
                } as unknown as Record<string, unknown>,
                timestamp: Date.now(),
              };
              ws.send(JSON.stringify(errorMsg));
            }
            break;
          }

          default:
            log?.warn(`[adp-openclaw] Unknown message type: ${msg.type}`);
        }
      } catch (err) {
        log?.error(`[adp-openclaw] Failed to parse message: ${err}`);
      }
    });

    ws.on("close", (code, reason) => {
      if (pingInterval) clearInterval(pingInterval);
      abortSignal?.removeEventListener("abort", abortHandler);
      // Clear active WebSocket when connection closes
      setActiveWebSocket(null);
      // Report disconnected status to SDK health-monitor
      setStatus?.({ connected: false, lastEventAt: Date.now() });
      log?.info(`[adp-openclaw] WebSocket closed: ${code} ${reason.toString()}`);
      resolve();
    });

    ws.on("error", (err) => {
      if (pingInterval) clearInterval(pingInterval);
      abortSignal?.removeEventListener("abort", abortHandler);
      reject(err);
    });
  });
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (signal?.aborted) {
      resolve();
      return;
    }
    const timeout = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(timeout);
      resolve();
    });
  });
}
