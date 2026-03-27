/**
 * WeCom 媒体文件大小限制常量
 *
 * 对标企业微信智能机器人临时素材上传限制。
 * 超出限制时，uploader 会按规则降级（如图片→文件）或拒绝。
 */

/** 图片最大字节数 (10 MB)，超出则降级为 file 类型 */
export const IMAGE_MAX_BYTES = 10 * 1024 * 1024;

/** 视频最大字节数 (10 MB)，超出则降级为 file 类型 */
export const VIDEO_MAX_BYTES = 10 * 1024 * 1024;

/** 语音最大字节数 (2 MB)，超出则降级为 file 类型 */
export const VOICE_MAX_BYTES = 2 * 1024 * 1024;

/** 文件最大字节数 (20 MB)，超出则拒绝发送 */
export const FILE_MAX_BYTES = 20 * 1024 * 1024;

/** 绝对大小上限，等于 FILE_MAX_BYTES */
export const ABSOLUTE_MAX_BYTES = FILE_MAX_BYTES;

/** 语音类型支持的 MIME 集合（企微仅支持 AMR 格式语音消息） */
export const VOICE_SUPPORTED_MIMES = new Set(["audio/amr"]);
