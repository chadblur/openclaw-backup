/**
 * Tool Result Message Blocks
 * 
 * 用于将工具结果转换为消息块，发送给用户
 * 参考 kimi-claw 的 tool-result-message-blocks.js 实现
 */

import { ADP_UPLOAD_TOOL_NAME, type AdpUploadToolResult, type UploadedFileInfo } from "./adp-upload-tool.js";

// ==================== 工具函数 ====================

const isRecord = (v: unknown): v is Record<string, unknown> =>
  !!v && typeof v === "object" && !Array.isArray(v);

const asTrimmedNonEmptyString = (v: unknown): string | undefined => {
  if (typeof v !== "string") return undefined;
  const trimmed = v.trim();
  return trimmed || undefined;
};

// ==================== 消息块类型 ====================

export interface ResourceLinkBlock {
  type: "resource_link";
  uri: string;
  name?: string;
  mimeType?: string;
  downloadUrl?: string;
}

export interface TextBlock {
  type: "text";
  text: string;
}

export interface ContentBlock {
  type: "content";
  content: ResourceLinkBlock | TextBlock;
}

export type MessageBlock = ContentBlock;

// ==================== 工具结果处理策略 ====================

interface ToolResultBlocksStrategy {
  canHandle: (toolName: string, result: unknown) => boolean;
  toMessageBlocks: (toolName: string, result: unknown) => MessageBlock[];
}

/**
 * ADP 上传工具结果处理策略
 */
const ADP_UPLOAD_TOOL_RESULT_BLOCKS_STRATEGY: ToolResultBlocksStrategy = {
  canHandle: (toolName: string, result: unknown): boolean => {
    return (
      toolName === ADP_UPLOAD_TOOL_NAME &&
      isRecord(result) &&
      result.ok === true
    );
  },

  toMessageBlocks: (toolName: string, result: unknown): MessageBlock[] => {
    if (!isRecord(result)) return [];

    const files = result.files;
    if (!Array.isArray(files)) return [];

    const blocks: MessageBlock[] = [];

    for (const file of files) {
      if (!isRecord(file)) continue;

      const uri = asTrimmedNonEmptyString(file.uri);
      if (!uri) continue;

      const name = asTrimmedNonEmptyString(file.name);
      const mimeType = asTrimmedNonEmptyString(file.mimeType);
      const downloadUrl = asTrimmedNonEmptyString(file.downloadUrl);

      const resourceLink: ResourceLinkBlock = {
        type: "resource_link",
        uri,
        ...(name ? { name } : {}),
        ...(mimeType ? { mimeType } : {}),
        ...(downloadUrl ? { downloadUrl } : {}),
      };

      blocks.push({
        type: "content",
        content: resourceLink,
      });
    }

    return blocks;
  },
};

// 所有支持的工具结果处理策略
const TOOL_RESULT_STRATEGIES: ToolResultBlocksStrategy[] = [
  ADP_UPLOAD_TOOL_RESULT_BLOCKS_STRATEGY,
];

// ==================== 公共接口 ====================

/**
 * 检查工具结果是否成功
 */
const isSuccessfulToolResult = (result: unknown): boolean =>
  isRecord(result) && result.ok === true;

/**
 * 将工具结果分发到对应的消息块处理器
 */
export const dispatchToolResultToMessageBlocks = (
  toolName: string,
  result: unknown,
  strategies: ToolResultBlocksStrategy[] = TOOL_RESULT_STRATEGIES
): MessageBlock[] => {
  if (!isSuccessfulToolResult(result)) return [];

  const blocks: MessageBlock[] = [];
  for (const strategy of strategies) {
    if (strategy.canHandle(toolName, result)) {
      blocks.push(...strategy.toMessageBlocks(toolName, result));
    }
  }
  return blocks;
};

/**
 * 从内容数组中解析可能的上传结果
 */
const maybeUploadResultFromContent = (
  content: unknown
): { ok: true; files: UploadedFileInfo[] } | undefined => {
  if (!Array.isArray(content)) return undefined;

  const files: UploadedFileInfo[] = [];
  for (const item of content) {
    if (!isRecord(item)) continue;

    const uri = asTrimmedNonEmptyString(item.uri);
    if (!uri) continue;

    const itemType = asTrimmedNonEmptyString(item.type);
    if (itemType && itemType !== "resource_link") continue;

    const fileInfo: UploadedFileInfo = {
      uri,
      name: asTrimmedNonEmptyString(item.name) || "",
      mimeType: asTrimmedNonEmptyString(item.mimeType) || "",
    };

    const downloadUrl = asTrimmedNonEmptyString(item.downloadUrl);
    if (downloadUrl) {
      fileInfo.downloadUrl = downloadUrl;
    }

    files.push(fileInfo);
  }

  return files.length ? { ok: true, files } : undefined;
};

/**
 * 从工具结果中提取有效负载
 */
const pickToolResultPayload = (
  result: Record<string, unknown>
): Record<string, unknown> | undefined => {
  // 尝试从不同的字段中提取结果
  const resultField = result.result;
  if (isRecord(resultField)) return resultField;

  const outputField = result.output;
  if (isRecord(outputField)) return outputField;

  const detailsField = result.details;
  if (isRecord(detailsField)) return detailsField;

  // 尝试从 content 数组中解析
  const contentField = result.content;
  if (isRecord(contentField)) return contentField;

  const maybeFromContent = maybeUploadResultFromContent(contentField);
  if (maybeFromContent) return maybeFromContent as unknown as Record<string, unknown>;

  return undefined;
};

/**
 * 从工具结果构建会话内容块
 * 
 * @param toolName - 工具名称
 * @param rawResult - 原始工具结果
 * @returns 消息块数组
 */
export const buildToolResultSessionContentBlocks = (
  toolName: string,
  rawResult: unknown
): MessageBlock[] => {
  const result = isRecord(rawResult) ? rawResult : undefined;
  if (!result) return [];

  const payload = pickToolResultPayload(result);
  if (!payload) return [];

  const blocks = dispatchToolResultToMessageBlocks(toolName, payload, TOOL_RESULT_STRATEGIES);
  return blocks.length ? blocks : [];
};

/**
 * 将上传结果转换为用户可读的文本消息
 * 
 * @param result - ADP 上传工具结果
 * @returns 用户可读的消息
 */
export const formatUploadResultForUser = (result: AdpUploadToolResult): string => {
  if (!result.ok || !result.files || result.files.length === 0) {
    const errorMsg = result.error?.data?.reason || result.error?.message || "上传失败";
    return `文件上传失败: ${errorMsg}`;
  }

  const lines: string[] = ["文件上传成功:"];
  for (const file of result.files) {
    if (file.downloadUrl) {
      lines.push(`📎 ${file.name}: ${file.downloadUrl}`);
    } else {
      lines.push(`📎 ${file.name} (${file.uri})`);
    }
  }

  return lines.join("\n");
};

/**
 * 将上传结果转换为 Markdown 格式的消息
 * 
 * @param result - ADP 上传工具结果
 * @returns Markdown 格式的消息
 */
export const formatUploadResultAsMarkdown = (result: AdpUploadToolResult): string => {
  if (!result.ok || !result.files || result.files.length === 0) {
    const errorMsg = result.error?.data?.reason || result.error?.message || "上传失败";
    return `**文件上传失败**: ${errorMsg}`;
  }

  const lines: string[] = ["**文件上传成功**:"];
  for (const file of result.files) {
    console.log(`[ADP-UPLOAD] formatUploadResultAsMarkdown - file.downloadUrl: ${file.downloadUrl}`);
    if (file.downloadUrl) {
      lines.push(`- [${file.name}](${file.downloadUrl})`);
    } else {
      lines.push(`- ${file.name} (\`${file.uri}\`)`);
    }
  }

  return lines.join("\n");
};
