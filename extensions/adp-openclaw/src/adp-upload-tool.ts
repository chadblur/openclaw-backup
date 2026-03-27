/**
 * ADP Upload Tool
 * 
 * 功能：
 * 1. 通过固定接口获取预签名的上传URL和下载URL
 * 2. 使用预签名URL直接PUT上传文件到COS
 * 3. 返回文件下载链接
 * 
 * 注意：机器人 token 从 adp-openclaw 插件配置的 clientToken 读取
 */

import { constants } from "node:fs";
import { access, readFile, stat } from "node:fs/promises";
import { basename, extname } from "node:path";
import { EventEmitter } from "node:events";
import type { AdpOpenclawChannelConfig } from "./channel.js";

// ==================== 上传结果事件 ====================

/** 上传结果事件发射器，用于通知其他模块上传完成 */
export const uploadResultEmitter = new EventEmitter();

/** 上传结果事件名称 */
export const UPLOAD_RESULT_EVENT = "adp-upload-result";

// ==================== 类型定义 ====================

/** 存储凭证请求参数 */
export interface DescribeRemoteBotStorageCredentialReq {
  token: string;       // 机器人token
  file_type?: string;  // 文件类型
}

/** 临时凭证信息 */
export interface Credentials {
  token: string;           // token
  tmp_secret_id: string;   // 临时证书密钥 ID
  tmp_secret_key: string;  // 临时证书密钥 Key
}

/** 存储凭证响应 */
export interface DescribeRemoteBotStorageCredentialRsp {
  credentials: Credentials;  // 密钥信息（兼容旧字段，新逻辑不使用）
  expired_time: number;      // 失效时间
  start_time: number;        // 起始时间
  bucket: string;            // 对象存储 桶
  region: string;            // 对象存储 可用区
  file_path: string;         // 文件目录
  upload_url: string;        // 预签名上传URL（需要decodeURIComponent）
  file_url: string;          // 预签名下载URL（需要decodeURIComponent）
}

/** 上传结果 */
export interface UploadResult {
  ok: boolean;
  fileUrl?: string;      // 文件下载签名链接
  error?: string;        // 错误信息
}

/** 文件元信息 */
interface FileMetadata {
  path: string;
  fileName: string;
  contentType: string;
  size: number;
}

// ==================== 常量配置 ====================

/** 获取临时密钥的接口地址 */
const CREDENTIAL_API_URL = "https://wss.lke.cloud.tencent.com/v1/gateway/storage/get_credential";

/** 请求超时时间 (毫秒) */
const REQUEST_TIMEOUT_MS = 30000;

// ==================== 工具函数 ====================

/** 扩展名到 MIME 类型的映射 */
const EXT_TO_MIME_MAP: Record<string, string> = {
  ".txt": "text/plain",
  ".md": "text/markdown",
  ".json": "application/json",
  ".html": "text/html",
  ".css": "text/css",
  ".js": "text/javascript",
  ".ts": "text/x.typescript",
  ".xml": "application/xml",
  ".yaml": "application/yaml",
  ".yml": "application/yaml",
  ".pdf": "application/pdf",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".svg": "image/svg+xml",
  ".csv": "text/csv",
  ".zip": "application/zip",
  ".doc": "application/msword",
  ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  ".xls": "application/vnd.ms-excel",
  ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  ".ppt": "application/vnd.ms-powerpoint",
  ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  ".mp3": "audio/mpeg",
  ".mp4": "video/mp4",
  ".wav": "audio/wav",
};

/** MIME 类型到扩展名的映射（反向映射） */
const MIME_TO_EXT_MAP: Record<string, string> = Object.fromEntries(
  Object.entries(EXT_TO_MIME_MAP).map(([ext, mime]) => [mime, ext.slice(1)]) // 去掉 "." 前缀
);

/**
 * 根据文件扩展名推断 MIME 类型
*/
function inferMimeType(fileName: string): string {
  const ext = extname(fileName).toLowerCase();
  return EXT_TO_MIME_MAP[ext] || "application/octet-stream";
}

/**
 * 把 MIME 类型或扩展名转换为纯扩展名（不带点）
 * 服务端 API 期望的是文件扩展名，而不是 MIME 类型
 */
function toFileExtension(fileTypeOrMime: string): string {
  if (!fileTypeOrMime) return "";
  
  // 如果已经是扩展名格式（不包含 /），直接返回（去掉可能的前导点）
  if (!fileTypeOrMime.includes("/")) {
    return fileTypeOrMime.replace(/^\./, "");
  }
  
  // 如果是 MIME 类型，转换为扩展名
  return MIME_TO_EXT_MAP[fileTypeOrMime] || fileTypeOrMime;
}

/**
 * 带超时的 fetch 请求
 */
async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeoutMs: number = REQUEST_TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return response;
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * 验证本地文件是否可读
 */
async function validateLocalFile(filePath: string): Promise<FileMetadata> {
  try {
    await access(filePath, constants.R_OK);
  } catch (e) {
    const error = e as NodeJS.ErrnoException;
    if (error.code === "ENOENT" || error.code === "ENOTDIR") {
      throw new Error(`文件不存在: ${filePath}`);
    }
    throw new Error(`文件不可读: ${filePath}`);
  }

  const fileStat = await stat(filePath);
  if (!fileStat.isFile() || fileStat.isDirectory()) {
    throw new Error(`路径不是有效文件: ${filePath}`);
  }

  const fileName = basename(filePath);
  return {
    path: filePath,
    fileName,
    contentType: inferMimeType(fileName),
    size: fileStat.size,
  };
}

// ==================== 核心功能 ====================

/**
 * 获取COS临时密钥
 */
export async function getStorageCredential(
  token: string,
  fileType?: string
): Promise<DescribeRemoteBotStorageCredentialRsp> {
  // 把 MIME 类型转换为文件扩展名，服务端 API 期望的是扩展名
  const fileExtension = toFileExtension(fileType || "");
  
  const requestBody: DescribeRemoteBotStorageCredentialReq = {
    token,
    file_type: fileExtension,
  };

  console.log(`[ADP-UPLOAD] Calling credential API: ${CREDENTIAL_API_URL}`);
  console.log(`[ADP-UPLOAD] Request body: token=${token.substring(0, 10)}..., file_type=${fileExtension} (original: ${fileType || ""})`);

  const response = await fetchWithTimeout(
    CREDENTIAL_API_URL,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    }
  );

  console.log(`[ADP-UPLOAD] Response status: ${response.status} ${response.statusText}`);

  if (!response.ok) {
    const errorText = await response.text().catch(() => response.statusText);
    console.error(`[ADP-UPLOAD] Error response: ${errorText}`);
    throw new Error(`获取存储凭证失败: ${response.status} ${errorText}`);
  }

  const responseData = await response.json();
  
  // 调试：打印原始响应的所有 key
  console.log(`[ADP-UPLOAD] Response keys: ${Object.keys(responseData).join(', ')}`);
  
  // 上游 API 返回的数据可能被包装在 result 字段中
  const result = responseData.result || responseData;
  
  console.log(`[ADP-UPLOAD] Result keys: ${Object.keys(result).join(', ')}`);
  console.log(`[ADP-UPLOAD] Response received: bucket=${result.bucket}, region=${result.region}, has_upload_url=${!!result.upload_url}, has_file_url=${!!result.file_url}`);
  
  // 检查响应数据的完整性 - 现在主要检查 upload_url 和 file_url
  if (!result.upload_url || !result.file_url) {
    console.error(`[ADP-UPLOAD] Missing required fields in response:`, JSON.stringify(responseData).substring(0, 500));
    throw new Error("获取存储凭证失败: 缺少upload_url或file_url信息");
  }

  console.log(`[ADP-UPLOAD] Credential file_url: ${result.file_url.substring(0, 100)}...`);

  return result as DescribeRemoteBotStorageCredentialRsp;
}

/**
 * 上传文件到COS（使用预签名URL直接PUT上传）
 * 
 * @param filePath - 本地文件路径
 * @param credential - 存储凭证（包含预签名的 upload_url 和 file_url）
 * @returns 文件下载URL
 */
export async function uploadFileToCos(
  filePath: string,
  credential: DescribeRemoteBotStorageCredentialRsp
): Promise<string> {
  // 1. 验证本地文件
  const fileMetadata = await validateLocalFile(filePath);

  // 2. 读取文件内容
  const fileContent = await readFile(filePath);

  // 3. 获取预签名的上传URL和下载URL
  // 注意：不要用 decodeURIComponent 解码整个 URL，会破坏签名参数
  // JSON 解析已经处理了 \u0026 -> & 的转换
  const uploadUrl = credential.upload_url;
  const fileUrl = credential.file_url;
  
  console.log(`[ADP-UPLOAD] upload_url: ${uploadUrl.substring(0, 100)}...`);
  console.log(`[ADP-UPLOAD] file_url (download): ${fileUrl}...`);

  // 4. 执行上传（直接PUT到预签名URL，不需要额外签名）
  const response = await fetchWithTimeout(
    uploadUrl,
    {
      method: "PUT",
      headers: {
        "Content-Type": fileMetadata.contentType,
        "Content-Length": String(fileContent.length),
      },
      body: fileContent,
    },
    REQUEST_TIMEOUT_MS * 2 // 上传超时时间翻倍
  );

  if (!response.ok) {
    const errorText = await response.text().catch(() => response.statusText);
    throw new Error(`上传文件失败: ${response.status} ${errorText}`);
  }

  console.log(`[ADP-UPLOAD] Upload success, returning file_url: ${fileUrl}...`);

  // 5. 返回下载URL
  return fileUrl;
}

// ==================== 从配置读取 Token 的辅助函数 ====================

/**
 * 从插件配置或环境变量中获取 clientToken
 */
export function resolveClientToken(channelCfg?: AdpOpenclawChannelConfig): string | null {
  // 优先从配置读取
  let clientToken = channelCfg?.clientToken?.trim();
  
  // 如果配置中没有，尝试从环境变量读取
  if (!clientToken) {
    clientToken = process.env.ADP_OPENCLAW_CLIENT_TOKEN || "";
  }
  
  return clientToken || null;
}

/**
 * ADP 上传工具主函数（使用显式传入的 token）
 * 
 * @param filePath - 要上传的本地文件路径
 * @param botToken - 机器人 token
 * @param fileType - 文件类型（可选）
 * @returns 上传结果，包含下载链接或错误信息
 */
export async function adpUploadFile(
  filePath: string,
  botToken: string,
  fileType?: string
): Promise<UploadResult> {
  try {
    // 如果没有传入 fileType，根据文件扩展名推断
    const actualFileType = fileType || inferMimeType(basename(filePath));
    
    // 1. 获取预签名URL
    const credential = await getStorageCredential(botToken, actualFileType);

    // 2. 上传文件到 COS（返回下载URL）
    const fileUrl = await uploadFileToCos(filePath, credential);

    return {
      ok: true,
      fileUrl: fileUrl,
    };
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : "未知错误",
    };
  }
}

/**
 * ADP 上传工具主函数（从配置读取 token）
 * 
 * @param filePath - 要上传的本地文件路径
 * @param channelCfg - adp-openclaw 插件配置（包含 clientToken）
 * @param fileType - 文件类型（可选）
 * @returns 上传结果，包含下载签名链接或错误信息
 */
export async function adpUploadFileWithConfig(
  filePath: string,
  channelCfg?: AdpOpenclawChannelConfig,
  fileType?: string
): Promise<UploadResult> {
  const botToken = resolveClientToken(channelCfg);
  
  if (!botToken) {
    return {
      ok: false,
      error: "未配置 clientToken，请在 adp-openclaw 插件配置中设置 clientToken",
    };
  }
  
  return adpUploadFile(filePath, botToken, fileType);
}

/**
 * 批量上传文件（使用显式传入的 token）
 * 
 * 注意：每个文件需要独立获取预签名URL，因为服务端为每个文件生成唯一路径
 * 
 * @param filePaths - 要上传的本地文件路径列表
 * @param botToken - 机器人 token
 * @param fileType - 文件类型（可选）
 * @returns 上传结果列表
 */
export async function adpUploadFiles(
  filePaths: string[],
  botToken: string,
  fileType?: string
): Promise<UploadResult[]> {
  // 逐个上传文件（每个文件需要独立的预签名URL）
  const results = await Promise.all(
    filePaths.map(async (filePath): Promise<UploadResult> => {
      return adpUploadFile(filePath, botToken, fileType);
    })
  );

  return results;
}

/**
 * 批量上传文件（从配置读取 token）
 * 
 * @param filePaths - 要上传的本地文件路径列表
 * @param channelCfg - adp-openclaw 插件配置（包含 clientToken）
 * @param fileType - 文件类型（可选）
 * @returns 上传结果列表
 */
export async function adpUploadFilesWithConfig(
  filePaths: string[],
  channelCfg?: AdpOpenclawChannelConfig,
  fileType?: string
): Promise<UploadResult[]> {
  const botToken = resolveClientToken(channelCfg);
  
  if (!botToken) {
    return filePaths.map(() => ({
      ok: false,
      error: "未配置 clientToken，请在 adp-openclaw 插件配置中设置 clientToken",
    }));
  }
  
  return adpUploadFiles(filePaths, botToken, fileType);
}

// ==================== AdpUploader 类 ====================

/**
 * ADP 上传工具类
 * 
 * 封装了从 adp-openclaw 配置读取 clientToken 的逻辑
 * 
 * @example
 * ```typescript
 * import { AdpUploader } from "./adp-upload-tool.js";
 * 
 * // 从插件配置创建上传器
 * const uploader = new AdpUploader(channelConfig);
 * 
 * // 或者从环境变量创建（不传参数）
 * const uploader = new AdpUploader();
 * 
 * // 单文件上传
 * const result = await uploader.upload("/path/to/file.pdf");
 * 
 * // 批量上传
 * const results = await uploader.uploadMultiple(["/path/to/file1.pdf", "/path/to/file2.png"]);
 * ```
 */
export class AdpUploader {
  private clientToken: string | null;

  /**
   * 创建 ADP 上传器实例
   * 
   * @param channelCfg - adp-openclaw 插件配置，如果不传则从环境变量读取
   */
  constructor(channelCfg?: AdpOpenclawChannelConfig) {
    this.clientToken = resolveClientToken(channelCfg);
  }

  /**
   * 检查是否已配置 token
   */
  isConfigured(): boolean {
    return Boolean(this.clientToken);
  }

  /**
   * 获取当前使用的 token（脱敏显示）
   */
  getTokenPreview(): string | null {
    if (!this.clientToken) return null;
    if (this.clientToken.length <= 10) return "***";
    return `${this.clientToken.substring(0, 5)}...${this.clientToken.substring(this.clientToken.length - 5)}`;
  }

  /**
   * 上传单个文件
   * 
   * @param filePath - 要上传的本地文件路径
   * @param fileType - 文件类型（可选）
   * @returns 上传结果
   */
  async upload(filePath: string, fileType?: string): Promise<UploadResult> {
    if (!this.clientToken) {
      return {
        ok: false,
        error: "未配置 clientToken，请在 adp-openclaw 插件配置中设置 clientToken 或设置环境变量 ADP_OPENCLAW_CLIENT_TOKEN",
      };
    }
    return adpUploadFile(filePath, this.clientToken, fileType);
  }

  /**
   * 批量上传文件
   * 
   * @param filePaths - 要上传的本地文件路径列表
   * @param fileType - 文件类型（可选）
   * @returns 上传结果列表
   */
  async uploadMultiple(filePaths: string[], fileType?: string): Promise<UploadResult[]> {
    if (!this.clientToken) {
      return filePaths.map(() => ({
        ok: false,
        error: "未配置 clientToken，请在 adp-openclaw 插件配置中设置 clientToken 或设置环境变量 ADP_OPENCLAW_CLIENT_TOKEN",
      }));
    }
    return adpUploadFiles(filePaths, this.clientToken, fileType);
  }
}

// ==================== Tool Schema (用于 OpenClaw 集成) ====================

export const ADP_UPLOAD_TOOL_NAME = "adp_upload_file";

const ADP_UPLOAD_TOOL_SCHEMA_TITLE = "ADP file upload";
const ADP_UPLOAD_TOOL_SCHEMA_DESCRIPTION = 
  "Upload local files from this machine to ADP storage. If the call succeeds, the files are uploaded and signed download URLs are returned. The user can access the files via these URLs. IMPORTANT: When presenting download links to users, format them as clickable Markdown links like [filename](url), do NOT wrap URLs in code blocks or backticks.";
const ADP_UPLOAD_TOOL_PATHS_DESCRIPTION = 
  "Local filesystem paths only (1-10 total). Any file type is accepted, including text, images, documents, PDFs, archives, and binary files. Use absolute or workspace-relative local paths that are readable files.";

export const ADP_UPLOAD_TOOL_SCHEMA = {
  title: ADP_UPLOAD_TOOL_SCHEMA_TITLE,
  type: "object",
  description: ADP_UPLOAD_TOOL_SCHEMA_DESCRIPTION,
  additionalProperties: false,
  required: ["paths"],
  properties: {
    paths: {
      type: "array",
      minItems: 1,
      maxItems: 10,
      description: ADP_UPLOAD_TOOL_PATHS_DESCRIPTION,
      items: {
        type: "string",
        minLength: 1,
        description: "Single local filesystem path only, e.g., /Users/name/project/a.pdf or ./relative/path/binary.bin",
      },
    },
    fileType: {
      type: "string",
      description: "文件类型（可选）",
    },
  },
};

export const ADP_UPLOAD_TOOL_MAX_PATHS = 10;
export const ADP_UPLOAD_TOOL_MIN_PATHS = 1;
export const ADP_UPLOAD_TOOL_MAX_CONCURRENCY = 3;

// ==================== 参数解析和验证 ====================

const INVALID_PARAMS = { code: -32602, message: "invalid params" };
export const ADP_UPLOAD_VALIDATION_MESSAGE = INVALID_PARAMS.message;

interface ValidationError {
  ok: false;
  error: {
    code: number;
    message: string;
    data: { field: string; reason: string };
  };
}

interface ValidationSuccess<T> {
  ok: true;
  value: T;
}

type ValidationResult<T> = ValidationError | ValidationSuccess<T>;

const invalid = (field: string, reason: string): ValidationError => ({
  ok: false,
  error: { ...INVALID_PARAMS, data: { field, reason } },
});

const isRecord = (v: unknown): v is Record<string, unknown> =>
  !!v && typeof v === "object" && !Array.isArray(v);

const asTrimmedNonEmptyString = (v: unknown): string | undefined => {
  if (typeof v !== "string") return undefined;
  const trimmed = v.trim();
  return trimmed || undefined;
};

export interface AdpUploadToolParams {
  paths: string[];
  fileType?: string;
}

/**
 * 解析 ADP 上传工具参数
 */
export const parseAdpUploadToolParams = (
  params: unknown
): ValidationResult<AdpUploadToolParams> => {
  if (!isRecord(params)) {
    return invalid("params", "must be an object");
  }

  // 检查是否有未知字段
  const unknownFields = Object.keys(params).filter(
    (k) => k !== "paths" && k !== "fileType"
  );
  if (unknownFields.length > 0) {
    return invalid("paths", "only paths and fileType are allowed");
  }

  // 检查 paths 是否存在
  if (!Object.prototype.hasOwnProperty.call(params, "paths")) {
    return invalid("paths", "is required");
  }

  const paths = params.paths;
  if (!Array.isArray(paths)) {
    return invalid("paths", "must be an array");
  }

  if (paths.length < ADP_UPLOAD_TOOL_MIN_PATHS) {
    return invalid("paths", `must contain at least ${ADP_UPLOAD_TOOL_MIN_PATHS} item`);
  }

  if (paths.length > ADP_UPLOAD_TOOL_MAX_PATHS) {
    return invalid("paths", `must contain no more than ${ADP_UPLOAD_TOOL_MAX_PATHS} items`);
  }

  const validatedPaths: string[] = [];
  for (let i = 0; i < paths.length; i += 1) {
    const p = asTrimmedNonEmptyString(paths[i]);
    if (!p) {
      return invalid(`paths[${i}]`, "must be a non-empty string");
    }
    validatedPaths.push(p);
  }

  // 解析可选的 fileType
  const fileType = asTrimmedNonEmptyString(params.fileType);

  return {
    ok: true,
    value: { paths: validatedPaths, fileType },
  };
};

// ==================== 上传结果类型（兼容 kimi-upload-tool 格式） ====================

/** 成功上传的文件信息 */
export interface UploadedFileInfo {
  uri: string;        // 文件 URI，格式: adp-file://{cosKey}
  name: string;       // 文件名
  mimeType: string;   // MIME 类型
  downloadUrl?: string; // 签名下载链接
}

/** 完整上传结果 */
export interface AdpUploadToolResult {
  ok: boolean;
  files?: UploadedFileInfo[];
  /** 预格式化的消息，AI 可以直接展示给用户（不要用代码块包装） */
  message?: string;
  error?: {
    code: number;
    message: string;
    data?: { field: string; reason: string };
  };
}

// ==================== 并发控制工具 ====================

const normalizeConcurrency = (c: number | undefined): number =>
  !c || c < 1 ? 1 : c;

const runWithConcurrency = async <T, R>(
  items: T[],
  concurrency: number,
  fn: (item: T, index: number) => Promise<R>
): Promise<R[]> => {
  if (items.length === 0) return [];
  
  const limit = Math.min(normalizeConcurrency(concurrency), items.length);
  const results = new Array<R>(items.length);
  let nextIndex = 0;

  const workers = Array.from({ length: limit }, async () => {
    for (;;) {
      const idx = nextIndex;
      nextIndex += 1;
      if (idx >= items.length) return;
      results[idx] = await fn(items[idx], idx);
    }
  });

  await Promise.all(workers);
  return results;
};

// ==================== 上传到 ADP 的主函数 ====================

export interface AdpUploadOptions {
  botToken: string;
  fileType?: string;
  maxConcurrency?: number;
  requestTimeoutMs?: number;
}

/**
 * 上传文件到 ADP 存储（返回兼容 kimi 格式的结果）
 * 
 * 注意：每个文件需要独立获取预签名URL，因为服务端为每个文件生成唯一路径
 */
export const uploadFilesToAdpEndpoint = async (
  paths: string[],
  options: AdpUploadOptions
): Promise<AdpUploadToolResult> => {
  const { botToken, fileType, maxConcurrency = ADP_UPLOAD_TOOL_MAX_CONCURRENCY } = options;

  if (!botToken || typeof botToken !== "string" || !botToken.trim()) {
    return {
      ok: false,
      error: { ...INVALID_PARAMS, data: { field: "botToken", reason: "bot token is required" } },
    };
  }

  // 验证所有本地文件
  const validationResults = await runWithConcurrency(
    paths,
    maxConcurrency,
    async (path, index): Promise<ValidationResult<FileMetadata>> => {
      try {
        const metadata = await validateLocalFile(path);
        return { ok: true, value: metadata };
      } catch (error) {
        return invalid(
          `paths[${index}]`,
          error instanceof Error ? error.message : "must be readable"
        );
      }
    }
  );

  // 检查验证结果
  const validationError = validationResults.find((r): r is ValidationError => !r.ok);
  if (validationError) {
    return validationError;
  }

  const fileMetadatas = validationResults.map(
    (r) => (r as ValidationSuccess<FileMetadata>).value
  );

  // 并发上传所有文件（每个文件独立获取预签名URL）
  try {
    const uploadResults = await runWithConcurrency(
      fileMetadatas,
      maxConcurrency,
      async (metadata, _index): Promise<UploadedFileInfo> => {
        // 为每个文件获取预签名URL，优先使用传入的 fileType，否则使用文件的 contentType
        const actualFileType = fileType || metadata.contentType || "";
        const credential = await getStorageCredential(botToken, actualFileType);
        // 上传文件并获取下载URL
        const downloadUrl = await uploadFileToCos(metadata.path, credential);
        return {
          uri: downloadUrl,
          name: metadata.fileName,
          mimeType: metadata.contentType,
          downloadUrl: downloadUrl,
        };
      }
    );

    // 构建预格式化的消息，告诉 AI 如何展示给用户
    const formattedLinks = uploadResults.map(f => 
      `- [${f.name}](${f.downloadUrl})`
    ).join("\n");
    const message = `Files uploaded successfully. Present these download links to the user as clickable Markdown links (do NOT use code blocks):\n${formattedLinks}`;

    return {
      ok: true,
      files: uploadResults,
      message,
    };
  } catch (error) {
    return {
      ok: false,
      error: {
        ...INVALID_PARAMS,
        data: {
          field: "upload",
          reason: error instanceof Error ? error.message : "upload failed",
        },
      },
    };
  }
};

/**
 * 执行 ADP 上传工具（解析参数并上传）
 */
export const executeAdpUploadTool = async (
  params: unknown,
  botToken: string,
  options?: Partial<AdpUploadOptions>
): Promise<AdpUploadToolResult> => {
  // 解析参数
  const parsed = parseAdpUploadToolParams(params);
  if (!parsed.ok) {
    return parsed;
  }

  // 执行上传
  return uploadFilesToAdpEndpoint(parsed.value.paths, {
    botToken,
    fileType: parsed.value.fileType,
    ...options,
  });
};
