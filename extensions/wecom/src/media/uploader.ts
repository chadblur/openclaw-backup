/**
 * WeCom WS 模式媒体上传工具模块
 *
 * 接收 deliver callback 已加载的 Buffer，执行：
 *   detectWeComMediaType → applyFileSizeLimits → wsClient.uploadMedia → wsClient.sendMediaMessage
 *
 * 不含 resolveMediaFile（由 deliver callback 提供 Buffer）。
 */

import type { WeComMediaType, WSClient } from "@wecom/aibot-node-sdk";
import {
  IMAGE_MAX_BYTES,
  VIDEO_MAX_BYTES,
  VOICE_MAX_BYTES,
  ABSOLUTE_MAX_BYTES,
  VOICE_SUPPORTED_MIMES,
} from "./const.js";

// ============================================================================
// 类型定义
// ============================================================================

/** 文件大小检查结果 */
export interface FileSizeCheckResult {
  /** 最终确定的企微媒体类型（可能被降级） */
  finalType: WeComMediaType;
  /** 是否需要拒绝（超过绝对限制） */
  shouldReject: boolean;
  /** 拒绝原因（仅 shouldReject=true 时有值） */
  rejectReason?: string;
  /** 是否发生了降级 */
  downgraded: boolean;
  /** 降级说明（仅 downgraded=true 时有值） */
  downgradeNote?: string;
}

/** uploadAndSendMediaBuffer 的参数 */
export interface UploadAndSendMediaBufferOptions {
  /** WSClient 实例 */
  wsClient: WSClient;
  /** 文件数据（deliver callback 已读取） */
  buffer: Buffer;
  /** MIME 类型（deliver callback 已检测） */
  contentType: string;
  /** 文件名（deliver callback 已提取） */
  fileName: string;
  /** 目标会话 ID（单聊为 userid，群聊为 chatid） */
  chatId: string;
  /** 日志函数 */
  log?: (msg: string) => void;
  /** 错误日志函数 */
  errorLog?: (msg: string) => void;
}

/** uploadAndSendMediaBuffer 的返回结果 */
export interface UploadAndSendMediaResult {
  /** 是否发送成功 */
  ok: boolean;
  /** 最终的企微媒体类型 */
  finalType?: WeComMediaType;
  /** 是否被拒绝（文件过大） */
  rejected?: boolean;
  /** 拒绝原因 */
  rejectReason?: string;
  /** 是否发生了降级 */
  downgraded?: boolean;
  /** 降级说明 */
  downgradeNote?: string;
  /** 错误信息 */
  error?: string;
}

// ============================================================================
// MIME → 企微媒体类型映射
// ============================================================================

/**
 * 根据 MIME 类型检测企微媒体类型
 */
export function detectWeComMediaType(mimeType: string): WeComMediaType {
  const mime = mimeType.toLowerCase();

  if (mime.startsWith("image/")) return "image";
  if (mime.startsWith("video/")) return "video";
  if (mime.startsWith("audio/") || mime === "application/ogg") return "voice";

  return "file";
}

// ============================================================================
// 文件大小检查与降级
// ============================================================================

/**
 * 检查文件大小并执行降级策略
 *
 * 降级规则：
 * - voice 非 AMR 格式 → 降级为 file
 * - image 超过 10MB → 降级为 file
 * - video 超过 10MB → 降级为 file
 * - voice 超过 2MB → 降级为 file
 * - file 超过 20MB → 拒绝发送
 */
export function applyFileSizeLimits(
  fileSize: number,
  detectedType: WeComMediaType,
  contentType?: string,
): FileSizeCheckResult {
  const fileSizeMB = (fileSize / (1024 * 1024)).toFixed(2);

  // 绝对上限 20MB
  if (fileSize > ABSOLUTE_MAX_BYTES) {
    return {
      finalType: detectedType,
      shouldReject: true,
      rejectReason: `文件大小 ${fileSizeMB}MB 超过企业微信最大限制 20MB，无法发送`,
      downgraded: false,
    };
  }

  switch (detectedType) {
    case "image":
      if (fileSize > IMAGE_MAX_BYTES) {
        return {
          finalType: "file",
          shouldReject: false,
          downgraded: true,
          downgradeNote: `图片大小 ${fileSizeMB}MB 超过 10MB 限制，已转为文件格式发送`,
        };
      }
      break;

    case "video":
      if (fileSize > VIDEO_MAX_BYTES) {
        return {
          finalType: "file",
          shouldReject: false,
          downgraded: true,
          downgradeNote: `视频大小 ${fileSizeMB}MB 超过 10MB 限制，已转为文件格式发送`,
        };
      }
      break;

    case "voice":
      // 企微语音仅支持 AMR 格式
      if (contentType && !VOICE_SUPPORTED_MIMES.has(contentType.toLowerCase())) {
        return {
          finalType: "file",
          shouldReject: false,
          downgraded: true,
          downgradeNote: `语音格式 ${contentType} 不支持，企微仅支持 AMR 格式，已转为文件格式发送`,
        };
      }
      if (fileSize > VOICE_MAX_BYTES) {
        return {
          finalType: "file",
          shouldReject: false,
          downgraded: true,
          downgradeNote: `语音大小 ${fileSizeMB}MB 超过 2MB 限制，已转为文件格式发送`,
        };
      }
      break;

    case "file":
      break;
  }

  return {
    finalType: detectedType,
    shouldReject: false,
    downgraded: false,
  };
}

// ============================================================================
// 核心：接收 Buffer → 上传 → 发送
// ============================================================================

/**
 * 接收已有 Buffer，通过 WSClient 上传临时素材并发送媒体消息
 *
 * 流程：detectWeComMediaType → applyFileSizeLimits → uploadMedia → sendMediaMessage
 */
export async function uploadAndSendMediaBuffer(
  options: UploadAndSendMediaBufferOptions,
): Promise<UploadAndSendMediaResult> {
  const { wsClient, buffer, contentType, fileName, chatId, log, errorLog } = options;

  try {
    // 1. 检测企微媒体类型
    const detectedType = detectWeComMediaType(contentType);
    log?.(`media-upload: type=${detectedType} contentType=${contentType} size=${buffer.length} fileName=${fileName}`);

    // 2. 文件大小检查与降级
    const sizeCheck = applyFileSizeLimits(buffer.length, detectedType, contentType);

    if (sizeCheck.shouldReject) {
      errorLog?.(`media-upload: rejected — ${sizeCheck.rejectReason}`);
      return {
        ok: false,
        rejected: true,
        rejectReason: sizeCheck.rejectReason,
        finalType: sizeCheck.finalType,
      };
    }

    const finalType = sizeCheck.finalType;

    if (sizeCheck.downgraded) {
      log?.(`media-upload: downgraded ${detectedType}→${finalType} — ${sizeCheck.downgradeNote}`);
    }

    // 3. 分片上传获取 media_id
    log?.(`media-upload: uploading ${finalType} (${buffer.length} bytes)...`);
    const uploadResult = await wsClient.uploadMedia(buffer, {
      type: finalType,
      filename: fileName,
    });
    log?.(`media-upload: uploaded media_id=${uploadResult.media_id}`);

    // 4. 通过 sendMediaMessage 主动发送
    const videoOptions = finalType === "video" ? { title: fileName } : undefined;
    await wsClient.sendMediaMessage(chatId, finalType, uploadResult.media_id, videoOptions);
    log?.(`media-upload: sent to chatId=${chatId} type=${finalType}`);

    return {
      ok: true,
      finalType,
      downgraded: sizeCheck.downgraded,
      downgradeNote: sizeCheck.downgradeNote,
    };
  } catch (err) {
    const errMsg = String(err);
    errorLog?.(`media-upload: failed — ${errMsg}`);
    return {
      ok: false,
      error: errMsg,
    };
  }
}
