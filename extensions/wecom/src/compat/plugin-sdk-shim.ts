/**
 * OpenClaw Plugin SDK 兼容 Shim — 同步接口
 *
 * 解决 v2026.3.23 中 plugin-sdk 子路径重构导致部分符号从主入口消失的问题。
 *
 * 策略：
 * 1. 纯类型 → 直接从 `openclaw/plugin-sdk` 主入口重导出（所有版本均可）
 * 2. DEFAULT_ACCOUNT_ID → 硬编码常量 "default"（所有版本一致）
 * 3. deleteAccountFromConfigSection / setAccountEnabledInConfigSection
 *    → 通过在构建/加载时探测子路径，回退到主入口 compat 层
 * 4. readJsonFileWithFallback / writeJsonFileAtomically / withFileLock
 *    → 异步解析（仅 mcp-config.ts 需要，该模块中函数本身就是 async）
 * 5. promptAccountId → 异步解析（仅 onboarding.ts 需要，configure 回调本身是 async）
 */

// ─── 类型重导出 ───
export type { OpenClawConfig } from "openclaw/plugin-sdk";
export type { PluginRuntime } from "openclaw/plugin-sdk";
export type { OpenClawPluginApi } from "openclaw/plugin-sdk";
export type { ChannelPlugin, ChannelConfigSchema } from "openclaw/plugin-sdk";
export type { ChannelAccountSnapshot } from "openclaw/plugin-sdk";
export type { ChannelGatewayContext } from "openclaw/plugin-sdk";
export type { WizardPrompter } from "openclaw/plugin-sdk";
export type {
  ChannelOutboundAdapter,
  ChannelOutboundContext,
} from "openclaw/plugin-sdk";

// ─── 值：emptyPluginConfigSchema（主入口始终导出） ───
export { emptyPluginConfigSchema } from "openclaw/plugin-sdk";

// ─── DEFAULT_ACCOUNT_ID ───
export const DEFAULT_ACCOUNT_ID = "default";

// ─── 安全动态导入 ───
async function tryImport<T>(specifier: string): Promise<T | undefined> {
  try {
    return await import(specifier) as T;
  } catch {
    return undefined;
  }
}

// ────────────────────────────────────────────────────
// deleteAccountFromConfigSection / setAccountEnabledInConfigSection
// 这些函数在 channel.ts 的 config 回调中同步使用。
// 使用 "eager init + cache" 模式：模块加载时立即解析并缓存。
// ────────────────────────────────────────────────────
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyFn = (...args: any[]) => any;

let _deleteAccountFromConfigSection: AnyFn | undefined;
let _setAccountEnabledInConfigSection: AnyFn | undefined;

// 立即触发的异步自解析
const _configHelpersReady = (async () => {
  for (const subpath of [
    "openclaw/plugin-sdk/core",
    "openclaw/plugin-sdk/channel-plugin-common",
  ]) {
    const mod = await tryImport<Record<string, AnyFn>>(subpath);
    if (mod?.deleteAccountFromConfigSection && mod?.setAccountEnabledInConfigSection) {
      _deleteAccountFromConfigSection = mod.deleteAccountFromConfigSection;
      _setAccountEnabledInConfigSection = mod.setAccountEnabledInConfigSection;
      return;
    }
  }
  // 如果子路径都没有，抛出明确错误
  throw new Error(
    "[wecom-compat] Cannot resolve config section helpers. " +
    "Ensure openclaw >=2026.2.24 is installed.",
  );
})();

/**
 * 等待 config section helpers 解析完成。
 * 在使用 deleteAccountFromConfigSection / setAccountEnabledInConfigSection 之前调用。
 */
export async function ensureConfigHelpers(): Promise<void> {
  await _configHelpersReady;
}

/** 同步获取 deleteAccountFromConfigSection（须确保 ensureConfigHelpers 已完成） */
export function deleteAccountFromConfigSection(...args: unknown[]): unknown {
  if (!_deleteAccountFromConfigSection) {
    throw new Error("[wecom-compat] Config helpers not initialized. Call ensureConfigHelpers() first.");
  }
  return _deleteAccountFromConfigSection(...args);
}

/** 同步获取 setAccountEnabledInConfigSection（须确保 ensureConfigHelpers 已完成） */
export function setAccountEnabledInConfigSection(...args: unknown[]): unknown {
  if (!_setAccountEnabledInConfigSection) {
    throw new Error("[wecom-compat] Config helpers not initialized. Call ensureConfigHelpers() first.");
  }
  return _setAccountEnabledInConfigSection(...args);
}

// ────────────────────────────────────────────────────
// promptAccountId (仅 onboarding.ts 需要，异步调用)
// ────────────────────────────────────────────────────
type PromptAccountIdFn = (params: {
  cfg: unknown;
  prompter: unknown;
  label: string;
  currentId: string;
  listAccountIds: (cfg: unknown) => string[];
  defaultAccountId: string;
}) => Promise<string>;

let _promptAccountId: PromptAccountIdFn | undefined;

export async function resolvePromptAccountId(): Promise<PromptAccountIdFn> {
  if (_promptAccountId) return _promptAccountId;

  for (const subpath of [
    "openclaw/plugin-sdk/matrix",
    "openclaw/plugin-sdk/channel-setup",
    "openclaw/plugin-sdk/setup",
  ]) {
    const mod = await tryImport<{ promptAccountId?: PromptAccountIdFn }>(subpath);
    if (mod?.promptAccountId) {
      _promptAccountId = mod.promptAccountId;
      return _promptAccountId;
    }
  }

  // 兜底实现
  _promptAccountId = async (params) => params.currentId || params.defaultAccountId;
  return _promptAccountId;
}

// ────────────────────────────────────────────────────
// readJsonFileWithFallback / writeJsonFileAtomically / withFileLock
// (仅 mcp-config.ts 需要，函数本身为 async)
// ────────────────────────────────────────────────────
type FileLockFn = <T>(
  filePath: string,
  options: unknown,
  fn: () => Promise<T>,
) => Promise<T>;

type ReadJsonFn = <T>(
  filePath: string,
  fallback: T,
) => Promise<{ value: T; exists: boolean }>;

type WriteJsonFn = (filePath: string, value: unknown) => Promise<void>;

export type FileIoHelpers = {
  withFileLock: FileLockFn;
  readJsonFileWithFallback: ReadJsonFn;
  writeJsonFileAtomically: WriteJsonFn;
};

let _fileIo: FileIoHelpers | undefined;

export async function resolveFileIoHelpers(): Promise<FileIoHelpers> {
  if (_fileIo) return _fileIo;

  const jsonStore = await tryImport<Partial<FileIoHelpers>>("openclaw/plugin-sdk/json-store");
  const msteams = await tryImport<{ withFileLock?: FileLockFn }>("openclaw/plugin-sdk/msteams");

  const readFn = jsonStore?.readJsonFileWithFallback;
  const writeFn = jsonStore?.writeJsonFileAtomically;
  const lockFn = msteams?.withFileLock;

  if (readFn && writeFn && lockFn) {
    _fileIo = { readJsonFileWithFallback: readFn, writeJsonFileAtomically: writeFn, withFileLock: lockFn };
    return _fileIo;
  }

  // ── Node.js 原生回退实现 ──
  const fs = await import("node:fs/promises");
  const nodePath = await import("node:path");

  const fallbackRead: ReadJsonFn = async <T>(filePath: string, fallback: T) => {
    try {
      const raw = await fs.readFile(filePath, "utf-8");
      return { value: JSON.parse(raw) as T, exists: true };
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === "ENOENT") {
        return { value: fallback, exists: false };
      }
      return { value: fallback, exists: false };
    }
  };

  const fallbackWrite: WriteJsonFn = async (filePath: string, value: unknown) => {
    const dir = nodePath.dirname(filePath);
    await fs.mkdir(dir, { recursive: true, mode: 0o700 });
    const content = JSON.stringify(value, null, 2) + "\n";
    const tmpPath = `${filePath}.tmp.${process.pid}`;
    await fs.writeFile(tmpPath, content, { mode: 0o600 });
    await fs.rename(tmpPath, filePath);
  };

  const fallbackLock: FileLockFn = async <T>(
    _filePath: string,
    _options: unknown,
    fn: () => Promise<T>,
  ): Promise<T> => fn();

  _fileIo = {
    readJsonFileWithFallback: readFn ?? fallbackRead,
    writeJsonFileAtomically: writeFn ?? fallbackWrite,
    withFileLock: lockFn ?? fallbackLock,
  };
  return _fileIo;
}
