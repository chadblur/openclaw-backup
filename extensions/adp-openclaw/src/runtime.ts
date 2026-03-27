// Runtime singleton for adp-openclaw plugin
import type { PluginRuntime } from "openclaw/plugin-sdk";
import type { WebSocket } from "ws";

let adpOpenclawRuntime: PluginRuntime | null = null;

// Plugin-level config storage (from plugins.entries.adp-openclaw.config)
export type PluginConfig = {
  wsUrl?: string;
  clientToken?: string;
  signKey?: string;
  // Session file reader config (for reading chat history from local files)
  sessionsStorePath?: string;   // Path to sessions.json
  openclawDir?: string;         // Base OpenClaw directory (default: ~/.openclaw)
  agentId?: string;             // Agent ID for session lookup (default: "main")
};

let pluginConfig: PluginConfig = {};

// Active WebSocket connection for outbound messaging
let activeWebSocket: WebSocket | null = null;

export function setActiveWebSocket(ws: WebSocket | null): void {
  activeWebSocket = ws;
}

export function getActiveWebSocket(): WebSocket | null {
  return activeWebSocket;
}

export function setAdpOpenclawRuntime(runtime: PluginRuntime): void {
  adpOpenclawRuntime = runtime;
}

export function getAdpOpenclawRuntime(): PluginRuntime {
  if (!adpOpenclawRuntime) {
    throw new Error("ADP OpenClaw runtime not initialized");
  }
  return adpOpenclawRuntime;
}

export function setPluginConfig(config: PluginConfig): void {
  pluginConfig = config;
}

export function getPluginConfig(): PluginConfig {
  return pluginConfig;
}
