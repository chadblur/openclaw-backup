/**
 * 媒体上传模块公共导出
 */
export { uploadAndSendMediaBuffer } from "./uploader.js";
export type { UploadAndSendMediaBufferOptions, UploadAndSendMediaResult } from "./uploader.js";
export { detectWeComMediaType, applyFileSizeLimits } from "./uploader.js";
export type { FileSizeCheckResult } from "./uploader.js";
export {
  IMAGE_MAX_BYTES,
  VIDEO_MAX_BYTES,
  VOICE_MAX_BYTES,
  FILE_MAX_BYTES,
  ABSOLUTE_MAX_BYTES,
  VOICE_SUPPORTED_MIMES,
} from "./const.js";
