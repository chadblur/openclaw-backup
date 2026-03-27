import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { emptyPluginConfigSchema } from "openclaw/plugin-sdk";
import { adpOpenclawPlugin, type AdpOpenclawChannelConfig } from "./src/channel.js";
import { setAdpOpenclawRuntime } from "./src/runtime.js";
import {
  ADP_UPLOAD_TOOL_NAME,
  ADP_UPLOAD_TOOL_SCHEMA,
  parseAdpUploadToolParams,
  uploadFilesToAdpEndpoint,
  uploadResultEmitter,
  UPLOAD_RESULT_EVENT,
  type AdpUploadToolResult,
  type UploadedFileInfo,
} from "./src/adp-upload-tool.js";

// Export session history functions for external use
export {
  getOpenClawChatHistory,
  listOpenClawSessions,
  getSessionPreviews,
  getChatHistoryByConversationId,
  resolveSessionKey,
  configureSessionFileReader,
  getSessionFileConfig,
  // CLI-based functions
  getOpenClawChatHistoryViaCli,
  listOpenClawSessionsViaCli,
  // Auto-selecting functions
  getChatHistory,
  listSessions,
  type OpenClawSession,
  type OpenClawMessage,
  type ChatHistoryResponse,
  type SessionsListResponse,
  type SessionFileConfig,
} from "./src/session-history.js";

// Export ADP upload tool functions and types
export {
  // Tool name and schema
  ADP_UPLOAD_TOOL_NAME,
  ADP_UPLOAD_TOOL_SCHEMA,
  ADP_UPLOAD_TOOL_MAX_PATHS,
  ADP_UPLOAD_TOOL_MIN_PATHS,
  ADP_UPLOAD_TOOL_MAX_CONCURRENCY,
  ADP_UPLOAD_VALIDATION_MESSAGE,
  // Main functions
  adpUploadFile,
  adpUploadFiles,
  adpUploadFileWithConfig,
  adpUploadFilesWithConfig,
  getStorageCredential,
  uploadFileToCos,
  resolveClientToken,
  // Class
  AdpUploader,
  // Tool execution functions
  parseAdpUploadToolParams,
  uploadFilesToAdpEndpoint,
  executeAdpUploadTool,
  // Types
  type UploadResult,
  type AdpUploadToolParams,
  type AdpUploadToolResult,
  type UploadedFileInfo,
  type AdpUploadOptions,
  type DescribeRemoteBotStorageCredentialReq,
  type DescribeRemoteBotStorageCredentialRsp,
  type Credentials,
} from "./src/adp-upload-tool.js";

// Export tool result message blocks functions
export {
  dispatchToolResultToMessageBlocks,
  buildToolResultSessionContentBlocks,
  formatUploadResultForUser,
  formatUploadResultAsMarkdown,
  type ResourceLinkBlock,
  type TextBlock,
  type ContentBlock,
  type MessageBlock,
} from "./src/tool-result-message-blocks.js";

// Helper to format tool result as JSON string
const formatToolResultJson = (result: unknown): string => 
  JSON.stringify(result, null, 2) ?? "upload failed";

const plugin = {
  id: "adp-openclaw",
  name: "ADP OpenClaw",
  description: "ADP channel plugin backed by a Go WebSocket server",
  configSchema: emptyPluginConfigSchema(),
  register(api: OpenClawPluginApi) {
    // Log registration for debugging
    console.log("[adp-openclaw] register() called - starting plugin registration");
    api.logger.info?.("[adp-openclaw] Plugin register() called");
    
    setAdpOpenclawRuntime(api.runtime);
    
    // Register the ADP file upload tool as a factory function
    // Using a factory so the tool is only available when the message originates from the ADP channel.
    // This prevents other channels (e.g. DingTalk) with multimodal models from picking up this tool.
    console.log(`[adp-openclaw] Registering tool factory: ${ADP_UPLOAD_TOOL_NAME}`);
    api.logger.info?.(`[adp-openclaw] Registering tool factory: ${ADP_UPLOAD_TOOL_NAME}`);
    
    api.registerTool((ctx: { messageChannel?: string; [key: string]: unknown }) => {
      // Only expose this tool when the message comes from the adp-openclaw channel
      const channel = ctx.messageChannel ?? "";
      if (channel && channel !== "adp-openclaw") {
        api.logger.debug?.(`[adp-openclaw] Skipping ${ADP_UPLOAD_TOOL_NAME} for channel: ${channel}`);
        return null;
      }

      return {
      name: ADP_UPLOAD_TOOL_NAME,
      description: 
        "If you can see this tool description, you are connected to the ADP server. " +
        "Use this tool to send 1-10 local files of any type (including images and documents) to the user. " +
        "A successful call means the files were uploaded and download URLs are returned. " +
        "IMPORTANT: You MUST include the download URLs in your response to the user, " +
        "and remind the user that the download links are valid for 24 hours. " +
        "Parameters must be local filesystem paths only.",
      parameters: ADP_UPLOAD_TOOL_SCHEMA,
      async execute(toolCallId: string, params: unknown) {
        // Get bot token from channel config
        const getClientToken = (): string | undefined => {
          try {
            const cfg = api.runtime?.config?.loadConfig?.();
            const channelCfg = cfg?.channels?.["adp-openclaw"] as AdpOpenclawChannelConfig | undefined;
            return channelCfg?.clientToken?.trim() || process.env.ADP_OPENCLAW_CLIENT_TOKEN;
          } catch {
            return process.env.ADP_OPENCLAW_CLIENT_TOKEN;
          }
        };

        // Parse and validate parameters
        const parsed = parseAdpUploadToolParams(params);
        if (!parsed.ok) {
          const errorResult = {
            ok: false,
            error: formatToolResultJson(parsed.error),
          };
          api.logger.debug?.(`[${ADP_UPLOAD_TOOL_NAME}] validation failed toolCallId=${toolCallId} error=${errorResult.error}`);
          return {
            output: errorResult,
            result: errorResult,
            details: errorResult,
            content: [{ type: "text", text: formatToolResultJson(parsed.error) }],
            isError: true,
          };
        }

        // Get bot token
        const botToken = getClientToken();
        if (!botToken) {
          const errorResult = {
            ok: false,
            error: "missing bot token for file upload - please configure clientToken in adp-openclaw channel settings",
          };
          api.logger.debug?.(`[${ADP_UPLOAD_TOOL_NAME}] token missing toolCallId=${toolCallId} paths=${JSON.stringify(parsed.value.paths)}`);
          return {
            output: errorResult,
            result: errorResult,
            details: errorResult,
            content: [{ type: "text", text: errorResult.error }],
            isError: true,
          };
        }

        // Execute upload
        const uploadResult = await uploadFilesToAdpEndpoint(parsed.value.paths, {
          botToken,
          fileType: parsed.value.fileType,
        });

        if (!uploadResult.ok) {
          const errorResult = {
            ok: false,
            error: formatToolResultJson(uploadResult.error),
          };
          api.logger.debug?.(`[${ADP_UPLOAD_TOOL_NAME}] upload failed toolCallId=${toolCallId} paths=${JSON.stringify(parsed.value.paths)} error=${errorResult.error}`);
          return {
            output: errorResult,
            result: errorResult,
            details: errorResult,
            content: [{ type: "text", text: formatToolResultJson(uploadResult.error) }],
            isError: true,
          };
        }

        // Success - format result with download URLs
        const successResult: AdpUploadToolResult = {
          ok: true,
          files: uploadResult.files,
        };

        api.logger.debug?.(`[${ADP_UPLOAD_TOOL_NAME}] upload success toolCallId=${toolCallId} count=${successResult.files?.length ?? 0} paths=${JSON.stringify(parsed.value.paths)}`);

        // Debug: print full downloadUrl for each file
        for (const file of (successResult.files || [])) {
          api.logger.info?.(`[${ADP_UPLOAD_TOOL_NAME}] file.downloadUrl: ${file.downloadUrl}`);
          api.logger.info?.(`[${ADP_UPLOAD_TOOL_NAME}] file.uri: ${file.uri}`);
        }
        
        // 发射上传结果事件，让 monitor.ts 能够直接获取完整的下载链接
        uploadResultEmitter.emit(UPLOAD_RESULT_EVENT, {
          toolCallId,
          result: successResult,
        });

        // Build content with resource links and download URLs
        const content: Array<{ type: string; uri?: string; name?: string; mimeType?: string; text?: string; downloadUrl?: string }> = [];
        
        // Add resource links for each file
        for (const file of (successResult.files || [])) {
          content.push({
            type: "resource_link",
            uri: file.downloadUrl || file.uri,  // 使用带签名的下载链接
            name: file.name,
            mimeType: file.mimeType,
            downloadUrl: file.downloadUrl,  // 也包含 downloadUrl 字段
          });
        }

        // Add a text summary with download URLs for AI to include in response
        // 注意：URL 包含签名参数，必须完整保留，不能截断或修改
        const urlSummary = (successResult.files || [])
          .map((f: UploadedFileInfo) => {
            const url = f.downloadUrl || f.uri;
            // 把完整 URL 作为代码块，防止 AI 截断或修改
            return `- **${f.name}**: \`${url}\``;
          })
          .join("\n");
        
        api.logger.info?.(`[${ADP_UPLOAD_TOOL_NAME}] urlSummary: ${urlSummary}`);
        
        content.push({
          type: "text",
          text: `Files uploaded successfully:\n${urlSummary}\n\n⚠️ IMPORTANT: The URLs above contain authentication signatures. You MUST copy the ENTIRE URL exactly as shown (including all query parameters after the "?"). Do NOT truncate or modify the URLs in any way. The links are valid for 24 hours.`,
        });

        return {
          output: successResult,
          result: successResult,
          details: successResult,
          content,
          isError: false,
        };
      },
    }; // end of tool object
    }); // end of factory function passed to registerTool
    
    // Log tool registration success
    console.log(`[adp-openclaw] Tool ${ADP_UPLOAD_TOOL_NAME} registered successfully`);
    api.logger.info?.(`[adp-openclaw] Tool ${ADP_UPLOAD_TOOL_NAME} registered successfully`);

    // Register the channel plugin
    api.registerChannel({ plugin: adpOpenclawPlugin });
    
    console.log("[adp-openclaw] Plugin registration complete");
    api.logger.info?.("[adp-openclaw] Plugin registration complete");
  },
};

export default plugin;
