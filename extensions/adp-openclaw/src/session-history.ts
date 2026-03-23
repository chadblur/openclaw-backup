// Session History: Read OpenClaw session/chat history
// This module provides functions to retrieve AI conversation history
// Supports multiple backends:
// 1. Direct file reading (for local session files)
// 2. CLI execution (via openclaw commands)

import type { PluginLogger } from "openclaw/plugin-sdk";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";
import { execSync, spawn, type SpawnOptions } from "node:child_process";

// OpenClaw session info returned from sessions.list
export type OpenClawSession = {
  key: string;                    // Session key (e.g., "main", "dm:peer:xxx")
  title?: string;                 // Session title
  derivedTitle?: string;          // AI-derived title
  lastMessage?: {                 // Last message in session
    role: "user" | "assistant";
    content: string;
    timestamp?: number;
  };
  messageCount?: number;          // Total messages in session
  createdAt?: number;             // Session creation timestamp
  updatedAt?: number;             // Last activity timestamp
  sessionId?: string;             // Session UUID
  sessionFile?: string;           // Path to session file
};

// OpenClaw chat message from chat.history
export type OpenClawMessage = {
  id?: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp?: number;
  metadata?: Record<string, unknown>;
};

// Chat history response
export type ChatHistoryResponse = {
  sessionKey: string;
  messages: OpenClawMessage[];
  hasMore?: boolean;
  cursor?: string;
};

// Sessions list response
export type SessionsListResponse = {
  sessions: OpenClawSession[];
  total?: number;
};

// Session store entry (from sessions.json)
type SessionStoreEntry = {
  sessionId: string;
  sessionFile?: string;
  updatedAt?: number;
  chatType?: string;
  origin?: {
    label?: string;
    provider?: string;
    surface?: string;
    chatType?: string;
    from?: string;
    to?: string;
  };
  lastChannel?: string;
  // ... other fields
};

// Session file line format
type SessionFileLine = {
  type?: string;
  message?: {
    role: "user" | "assistant" | "system";
    content: string | Array<{ type: string; text?: string }>;
    id?: string;
    timestamp?: string;
  };
};

/**
 * Configuration for session file reading
 */
export type SessionFileConfig = {
  // Path to sessions.json store file
  sessionsStorePath?: string;
  // Base directory for OpenClaw data (default: ~/.openclaw)
  openclawDir?: string;
  // Agent ID for session file lookup
  agentId?: string;
  // Backend to use: "file" (default), "cli", or "auto"
  backend?: "file" | "cli" | "auto";
  // Path to openclaw CLI binary (for CLI backend)
  cliBinary?: string;
  // Gateway URL (for CLI backend, optional)
  gatewayUrl?: string;
};

let sessionFileConfig: SessionFileConfig = {};

/**
 * Configure the session file reader
 */
export function configureSessionFileReader(config: SessionFileConfig): void {
  sessionFileConfig = { ...sessionFileConfig, ...config };
}

/**
 * Get the configured session file config
 */
export function getSessionFileConfig(): SessionFileConfig {
  return { ...sessionFileConfig };
}

/**
 * Resolve the sessions.json store path
 */
function resolveSessionsStorePath(config: SessionFileConfig): string[] {
  const candidates: string[] = [];
  
  if (config.sessionsStorePath) {
    candidates.push(config.sessionsStorePath);
  }
  
  const baseDir = config.openclawDir || path.join(os.homedir(), ".openclaw");
  const agentId = config.agentId || "main";
  
  // Standard locations
  candidates.push(path.join(baseDir, "agents", agentId, "sessions.json"));
  candidates.push(path.join(baseDir, "sessions.json"));
  candidates.push(path.join(process.cwd(), "sessions.json"));
  
  return candidates;
}

/**
 * Load the sessions store from file
 */
function loadSessionsStore(log?: PluginLogger): Record<string, SessionStoreEntry> {
  const candidates = resolveSessionsStorePath(sessionFileConfig);
  
  for (const storePath of candidates) {
    try {
      if (fs.existsSync(storePath)) {
        const raw = fs.readFileSync(storePath, "utf-8");
        const store = JSON.parse(raw) as Record<string, SessionStoreEntry>;
        log?.debug?.(`[session-history] Loaded sessions store from: ${storePath}`);
        return store;
      }
    } catch (err) {
      log?.debug?.(`[session-history] Failed to load sessions store from ${storePath}: ${err}`);
    }
  }
  
  log?.warn?.(`[session-history] No sessions store found in candidates: ${candidates.join(", ")}`);
  return {};
}

/**
 * Resolve session transcript file candidates
 */
function resolveSessionTranscriptCandidates(
  sessionId: string,
  sessionFile?: string,
  config?: SessionFileConfig,
): string[] {
  const candidates: string[] = [];
  const cfg = config || sessionFileConfig;
  
  // Direct session file path from store
  if (sessionFile) {
    candidates.push(sessionFile);
  }
  
  const baseDir = cfg.openclawDir || path.join(os.homedir(), ".openclaw");
  const agentId = cfg.agentId || "main";
  
  // Agent sessions directory
  candidates.push(path.join(baseDir, "agents", agentId, "sessions", `${sessionId}.jsonl`));
  
  // Legacy sessions directory
  candidates.push(path.join(baseDir, "sessions", `${sessionId}.jsonl`));
  
  // Current working directory
  if (cfg.sessionsStorePath) {
    const dir = path.dirname(cfg.sessionsStorePath);
    candidates.push(path.join(dir, `${sessionId}.jsonl`));
  }
  
  return candidates;
}

/**
 * Extract text content from message content (handles both string and array formats)
 */
function extractMessageContent(content: string | Array<{ type: string; text?: string }>): string {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .filter((part) => part.type === "text" && part.text)
      .map((part) => part.text)
      .join("\n");
  }
  return String(content);
}

/**
 * Read messages from a session transcript file (.jsonl)
 */
function readSessionTranscript(
  sessionId: string,
  sessionFile?: string,
  options?: {
    limit?: number;
    log?: PluginLogger;
  },
): OpenClawMessage[] {
  const { limit = 200, log } = options ?? {};
  const candidates = resolveSessionTranscriptCandidates(sessionId, sessionFile);
  
  const filePath = candidates.find((p) => fs.existsSync(p));
  if (!filePath) {
    log?.debug?.(`[session-history] No transcript file found for session ${sessionId}, tried: ${candidates.join(", ")}`);
    return [];
  }
  
  log?.debug?.(`[session-history] Reading transcript from: ${filePath}`);
  
  try {
    const lines = fs.readFileSync(filePath, "utf-8").split(/\r?\n/);
    const messages: OpenClawMessage[] = [];
    
    for (const line of lines) {
      if (!line.trim()) {
        continue;
      }
      try {
        const parsed = JSON.parse(line) as SessionFileLine;
        // Skip header lines (type: "session")
        if (parsed.type === "session") {
          continue;
        }
        // Extract message if present, skip tool-related roles (toolResult, tool)
        if (parsed.message && parsed.message.role && parsed.message.content) {
          // Filter out tool-related messages for cleaner display
          const role = parsed.message.role;
          if (role === "toolResult" || role === "tool") {
            continue;
          }
          const msg: OpenClawMessage = {
            role: parsed.message.role,
            content: extractMessageContent(parsed.message.content),
            id: parsed.message.id,
            timestamp: parsed.message.timestamp ? new Date(parsed.message.timestamp).getTime() : undefined,
          };
          messages.push(msg);
        }
      } catch {
        // ignore bad lines
      }
    }
    
    // Apply limit (return last N messages)
    if (messages.length > limit) {
      return messages.slice(-limit);
    }
    return messages;
  } catch (err) {
    log?.error?.(`[session-history] Failed to read transcript file ${filePath}: ${err}`);
    return [];
  }
}

/**
 * Get chat history for a specific session by reading session files directly
 * @param sessionKey - The session key (e.g., "main", "dm:peer:user123", "conv-xxx")
 * @param options - Optional parameters
 * @returns Chat history with messages
 */
export async function getOpenClawChatHistory(
  sessionKey: string = "main",
  options?: {
    limit?: number;
    cursor?: string;
    log?: PluginLogger;
  },
): Promise<ChatHistoryResponse> {
  const { limit = 200, log } = options ?? {};

  log?.info?.(`[session-history] Fetching chat history for session: ${sessionKey}, limit: ${limit}`);

  // Load sessions store to find the session entry
  const store = loadSessionsStore(log);
  const entry = store[sessionKey];
  
  if (!entry?.sessionId) {
    log?.warn?.(`[session-history] Session not found in store: ${sessionKey}`);
    return {
      sessionKey,
      messages: [],
      hasMore: false,
    };
  }
  
  // Read messages from transcript file
  const messages = readSessionTranscript(entry.sessionId, entry.sessionFile, {
    limit,
    log,
  });
  
  log?.info?.(`[session-history] Got ${messages.length} messages for session: ${sessionKey}`);
  
  return {
    sessionKey,
    messages,
    hasMore: false, // File-based reading doesn't support pagination
  };
}

/**
 * List all OpenClaw sessions from sessions.json
 * @param options - Optional parameters
 * @returns List of sessions
 */
export async function listOpenClawSessions(
  options?: {
    limit?: number;
    includeLastMessage?: boolean;
    includeDerivedTitles?: boolean;
    log?: PluginLogger;
  },
): Promise<SessionsListResponse> {
  const {
    limit = 100,
    includeLastMessage = true,
    log,
  } = options ?? {};

  log?.info?.(`[session-history] Listing sessions, limit: ${limit}`);

  const store = loadSessionsStore(log);
  const sessionEntries = Object.entries(store);
  
  // Sort by updatedAt descending
  sessionEntries.sort((a, b) => {
    const aTime = a[1].updatedAt ?? 0;
    const bTime = b[1].updatedAt ?? 0;
    return bTime - aTime;
  });
  
  // Apply limit
  const limited = sessionEntries.slice(0, limit);
  
  const sessions: OpenClawSession[] = [];
  
  for (const [key, entry] of limited) {
    const session: OpenClawSession = {
      key,
      sessionId: entry.sessionId,
      sessionFile: entry.sessionFile,
      updatedAt: entry.updatedAt,
    };
    
    // Include last message if requested
    if (includeLastMessage && entry.sessionId) {
      const messages = readSessionTranscript(entry.sessionId, entry.sessionFile, {
        limit: 1,
        log,
      });
      if (messages.length > 0) {
        const last = messages[messages.length - 1];
        session.lastMessage = {
          role: last.role as "user" | "assistant",
          content: last.content.slice(0, 200), // Truncate for preview
          timestamp: last.timestamp,
        };
        session.messageCount = messages.length;
      }
    }
    
    sessions.push(session);
  }
  
  log?.info?.(`[session-history] Got ${sessions.length} sessions`);
  
  return {
    sessions,
    total: sessionEntries.length,
  };
}

/**
 * Get session previews (batch fetch with last messages)
 * @param sessionKeys - Array of session keys to fetch
 * @param options - Optional parameters
 * @returns Array of session previews
 */
export async function getSessionPreviews(
  sessionKeys: string[],
  options?: {
    log?: PluginLogger;
  },
): Promise<OpenClawSession[]> {
  const { log } = options ?? {};

  log?.info?.(`[session-history] Getting previews for ${sessionKeys.length} sessions`);

  const store = loadSessionsStore(log);
  const sessions: OpenClawSession[] = [];
  
  for (const key of sessionKeys) {
    const entry = store[key];
    if (!entry?.sessionId) {
      continue;
    }
    
    const session: OpenClawSession = {
      key,
      sessionId: entry.sessionId,
      sessionFile: entry.sessionFile,
      updatedAt: entry.updatedAt,
    };
    
    // Get last few messages for preview
    const messages = readSessionTranscript(entry.sessionId, entry.sessionFile, {
      limit: 5,
      log,
    });
    
    if (messages.length > 0) {
      const last = messages[messages.length - 1];
      session.lastMessage = {
        role: last.role as "user" | "assistant",
        content: last.content.slice(0, 200),
        timestamp: last.timestamp,
      };
      session.messageCount = messages.length;
    }
    
    sessions.push(session);
  }
  
  log?.info?.(`[session-history] Got ${sessions.length} session previews`);
  
  return sessions;
}

/**
 * Find session key by conversation ID or user ID pattern
 * OpenClaw uses session keys like "dm:peer:{conversationId}" for per-peer sessions
 * @param conversationId - The conversation ID from GoServer
 * @returns The corresponding OpenClaw session key
 */
export function resolveSessionKey(conversationId: string): string {
  // For per-peer DM sessions, the key format is "dm:peer:{peerId}"
  // But in our case, we use "conv-{hash}" format
  // Check if it's already a session key format
  if (conversationId.startsWith("conv-") || conversationId.startsWith("dm:") || conversationId === "main") {
    return conversationId;
  }
  // Otherwise, assume it's a conversation ID hash
  return `conv-${conversationId}`;
}

/**
 * Get chat history by conversation ID (convenience wrapper)
 * @param conversationId - GoServer conversation ID
 * @param options - Optional parameters
 * @returns Chat history response
 */
export async function getChatHistoryByConversationId(
  conversationId: string,
  options?: {
    limit?: number;
    log?: PluginLogger;
  },
): Promise<ChatHistoryResponse> {
  const sessionKey = resolveSessionKey(conversationId);
  return getOpenClawChatHistory(sessionKey, options);
}

/**
 * Get merged chat history from both old and new sessionKey formats.
 * Old format: agent:main:direct:{conversationId}
 * New format: agent:main:adp-openclaw:direct:{conversationId}
 * 
 * This function fetches history from both session keys and merges them by timestamp.
 * 
 * @param conversationId - The conversation ID
 * @param options - Optional parameters
 * @returns Merged chat history sorted by timestamp
 */
export async function getMergedChatHistory(
  conversationId: string,
  options?: {
    limit?: number;
    log?: PluginLogger;
    filterAfterNewSession?: boolean;  // Default true: filter out messages before last /new
  },
): Promise<ChatHistoryResponse> {
  const { limit = 200, log, filterAfterNewSession = true } = options ?? {};
  
  // Build both old and new session key formats
  const oldSessionKey = `agent:main:direct:${conversationId}`;
  const newSessionKey = `agent:main:adp-openclaw:direct:${conversationId}`;
  
  log?.info?.(`[session-history] Fetching merged history for conversationId: ${conversationId}`);
  log?.info?.(`[session-history] Old sessionKey: ${oldSessionKey}`);
  log?.info?.(`[session-history] New sessionKey: ${newSessionKey}`);
  
  // Fetch history from both session keys in parallel
  // Note: getChatHistory already applies filterAfterNewSession for each session individually
  const [oldResult, newResult] = await Promise.all([
    getChatHistory(oldSessionKey, { limit, log }).catch((err) => {
      log?.debug?.(`[session-history] Failed to fetch old session history: ${err}`);
      return { sessionKey: oldSessionKey, messages: [], hasMore: false } as ChatHistoryResponse;
    }),
    getChatHistory(newSessionKey, { limit, log }).catch((err) => {
      log?.debug?.(`[session-history] Failed to fetch new session history: ${err}`);
      return { sessionKey: newSessionKey, messages: [], hasMore: false } as ChatHistoryResponse;
    }),
  ]);
  
  log?.info?.(`[session-history] Old session messages: ${oldResult.messages.length}`);
  log?.info?.(`[session-history] New session messages: ${newResult.messages.length}`);
  
  // Merge messages from both sessions
  const allMessages = [...oldResult.messages, ...newResult.messages];
  
  // Sort by timestamp (ascending, oldest first)
  allMessages.sort((a, b) => {
    const timeA = a.timestamp ?? 0;
    const timeB = b.timestamp ?? 0;
    return timeA - timeB;
  });
  
  // Deduplicate by message id (if available) or content+timestamp
  const seen = new Set<string>();
  let dedupedMessages = allMessages.filter((msg) => {
    // Create a unique key for deduplication
    const key = msg.id || `${msg.role}:${msg.timestamp}:${msg.content.slice(0, 100)}`;
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
  
  // Filter out messages before the last /new session on the MERGED result
  // This is important because the /new might be in the old sessionKey but we still
  // want to filter the combined history based on the latest /new across both sessions
  if (filterAfterNewSession) {
    dedupedMessages = filterMessagesAfterLastNewSession(dedupedMessages, log);
  }
  
  // Apply limit (return last N messages if exceeds limit)
  const limitedMessages = dedupedMessages.length > limit
    ? dedupedMessages.slice(-limit)
    : dedupedMessages;
  
  log?.info?.(`[session-history] Merged total: ${limitedMessages.length} messages`);
  
  return {
    sessionKey: newSessionKey, // Return new sessionKey as the primary
    messages: limitedMessages,
    hasMore: false,
  };
}

// ============================================================================
// CLI-based Backend
// ============================================================================

/**
 * Find the openclaw CLI binary path
 */
function findOpenClawCli(config: SessionFileConfig): string | null {
  // User-specified path
  if (config.cliBinary) {
    if (fs.existsSync(config.cliBinary)) {
      return config.cliBinary;
    }
    return null;
  }
  
  // Check common locations
  const candidates = [
    // npm global
    "openclaw",
    // npx
    "npx openclaw",
    // Local node_modules
    path.join(process.cwd(), "node_modules", ".bin", "openclaw"),
    // System paths
    "/usr/local/bin/openclaw",
    "/usr/bin/openclaw",
  ];
  
  for (const candidate of candidates) {
    try {
      execSync(`which ${candidate.split(" ")[0]} 2>/dev/null`, { encoding: "utf-8" });
      return candidate;
    } catch {
      // Not found, try next
    }
  }
  
  return null;
}
/**
 * Extract valid JSON from CLI output that may contain TUI decoration characters
 * (e.g. │, ◇, ◆, spinner frames) mixed into stdout.
 */
function extractJsonFromOutput(raw: string): string {
  // First, filter out TUI decoration lines (box-drawing, UI separators, warnings)
  const filtered = raw
    .split("\n")
    .filter((line) => {
      // Skip empty lines
      if (!line.trim()) return false;
      // Skip lines starting with box-drawing characters
      if (/^[│├┌┐└┘┤┬┴┼─]/.test(line)) return false;
      // Skip UI separator lines
      if (/^[◆◇]/.test(line)) return false;
      // Skip config warning messages (non-JSON)
      if (line.startsWith("Config warnings")) return false;
      return true;
    })
    .join("\n");

  const trimmed = filtered.trim();

  // Fast path: already valid JSON
  if (
    (trimmed.startsWith("{") && trimmed.endsWith("}")) ||
    (trimmed.startsWith("[") && trimmed.endsWith("]"))
  ) {
    return trimmed;
  }

  // Try to find the first top-level JSON object or array in the output
  const jsonStart = trimmed.search(/[\[{]/);
  if (jsonStart === -1) {
    return trimmed; // no JSON-like content, return as-is and let caller handle the error
  }

  const opener = trimmed[jsonStart];
  const closer = opener === "{" ? "}" : "]";

  // Walk forward tracking brace/bracket depth to find the matching close
  let depth = 0;
  let inString = false;
  let escaped = false;

  for (let i = jsonStart; i < trimmed.length; i++) {
    const ch = trimmed[i];

    if (escaped) {
      escaped = false;
      continue;
    }
    if (ch === "\\") {
      if (inString) escaped = true;
      continue;
    }
    if (ch === '"') {
      inString = !inString;
      continue;
    }
    if (inString) continue;

    if (ch === opener || ch === (opener === "{" ? "[" : "{")) {
      // count both kinds of nesting
      if (ch === "{" || ch === "[") depth++;
    }
    if (ch === closer || ch === (closer === "}" ? "]" : "}")) {
      if (ch === "}" || ch === "]") depth--;
    }

    if (depth === 0) {
      return trimmed.slice(jsonStart, i + 1);
    }
  }

  // Fallback: return from jsonStart onwards
  return trimmed.slice(jsonStart);
}

/**
 * Execute an openclaw CLI command and return the result.
 * @param subcommands - Array of subcommands, e.g. ["gateway", "call", "chat.history"] or ["sessions"]
 * @param args - Additional CLI flags, e.g. ["--json", "--params", '{"sessionKey":"main"}']
 */
async function executeClawCommand(
  subcommands: string[],
  args: string[],
  options?: {
    log?: PluginLogger;
    timeout?: number;
    gatewayUrl?: string;
    gatewayToken?: string;
  },
): Promise<string> {
  const { log, timeout = 30000, gatewayUrl, gatewayToken } = options ?? {};
  
  const cliPath = findOpenClawCli(sessionFileConfig);
  if (!cliPath) {
    throw new Error("OpenClaw CLI not found");
  }
  
  // Build command arguments
  const fullArgs = [...args];
  if (gatewayUrl) {
    fullArgs.push("--url", gatewayUrl);
  }
  if (gatewayToken) {
    fullArgs.push("--token", gatewayToken);
  }
  
  const cmdStr = `${cliPath} ${subcommands.join(" ")} ${fullArgs.join(" ")}`;
  log?.debug?.(`[session-history] Executing: ${cmdStr}`);
  
  return new Promise((resolve, reject) => {
    const parts = cliPath.split(" ");
    const binary = parts[0];
    const preArgs = parts.slice(1);
    
    // Ensure node's bin directory is in PATH (fixes nvm/pnpm environments
    // where the spawned shell may not have node in its PATH)
    const nodeDir = path.dirname(process.execPath);
    const currentPath = process.env.PATH || "";
    const envPath = currentPath.includes(nodeDir)
      ? currentPath
      : `${nodeDir}:${currentPath}`;

    const spawnOpts: SpawnOptions = {
      timeout,
      env: { ...process.env, PATH: envPath },
    };
    
    const child = spawn(binary, [...preArgs, ...subcommands, ...fullArgs], spawnOpts);
    
    let stdout = "";
    let stderr = "";
    
    child.stdout?.on("data", (data) => {
      stdout += data.toString();
    });
    
    child.stderr?.on("data", (data) => {
      stderr += data.toString();
    });
    
    child.on("error", (err) => {
      log?.error?.(`[session-history] CLI error: ${err.message}`);
      reject(err);
    });
    
    child.on("close", (code) => {
      if (code !== 0) {
        log?.error?.(`[session-history] CLI exited with code ${code}: ${stderr}`);
        reject(new Error(`CLI exited with code ${code}: ${stderr}`));
      } else {
        resolve(extractJsonFromOutput(stdout));
      }
    });
  });
}

/**
 * Filter messages to only include those after the last /new or session reset.
 * This handles the case where OpenClaw returns old session history after a /new command.
 * 
 * We look for messages containing "New session started" which is the indicator
 * that a new session was started via /new or /reset command.
 */
function filterMessagesAfterLastNewSession(
  messages: OpenClawMessage[],
  log?: PluginLogger,
): OpenClawMessage[] {
  if (messages.length === 0) {
    return messages;
  }
  
  // Find the index of the last "New session started" message
  // This message is typically from the assistant after processing /new
  let lastNewSessionIndex = -1;
  
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    // Check for the "New session started" indicator from OpenClaw
    // This appears in assistant messages after /new command
    if (
      msg.role === "assistant" &&
      msg.content &&
      (msg.content.includes("New session started") || 
       msg.content.includes("✅ New session started"))
    ) {
      lastNewSessionIndex = i;
      break;
    }
  }
  
  if (lastNewSessionIndex === -1) {
    // No /new found, return all messages
    log?.debug?.(`[session-history] No "New session started" marker found, returning all ${messages.length} messages`);
    return messages;
  }
  
  // Return messages from the last "New session started" onwards
  const filteredMessages = messages.slice(lastNewSessionIndex);
  log?.info?.(`[session-history] Filtered messages after last /new: ${messages.length} -> ${filteredMessages.length} messages`);
  
  return filteredMessages;
}

/**
 * Get chat history via CLI command
 * Uses: openclaw gateway call chat.history --params '{"sessionKey":"<key>","limit":<n>}' --json
 */
export async function getOpenClawChatHistoryViaCli(
  sessionKey: string = "main",
  options?: {
    limit?: number;
    log?: PluginLogger;
    gatewayUrl?: string;
    filterAfterNewSession?: boolean;  // Default true: filter out messages before last /new
  },
): Promise<ChatHistoryResponse> {
  const { limit = 200, log, gatewayUrl, filterAfterNewSession = true } = options ?? {};
  
  log?.info?.(`[session-history] Fetching chat history via CLI for session: ${sessionKey}`);
  
  try {
    // Build RPC params object
    const params: Record<string, unknown> = {};
    if (sessionKey) {
      params.sessionKey = sessionKey;
    }
    if (limit) {
      params.limit = limit;
    }
    
    // Use: openclaw gateway call chat.history --params '{...}' --json
    const args = ["--json", "--params", JSON.stringify(params)];
    
    const result = await executeClawCommand(
      ["gateway", "call", "chat.history"],
      args,
      {
        log,
        gatewayUrl: gatewayUrl || sessionFileConfig.gatewayUrl,
      },
    );
    
    // Parse JSON result
    const parsed = JSON.parse(result) as {
      messages?: Array<{
        role: string;
        content: string | Array<{ type: string; text?: string }>;
        id?: string;
        timestamp?: string;
      }>;
      sessionId?: string;  // OpenClaw may return the current sessionId
    };
    
    let messages: OpenClawMessage[] = (parsed.messages ?? []).map((msg) => ({
      role: msg.role as "user" | "assistant" | "system",
      content: typeof msg.content === "string"
        ? msg.content
        : msg.content
            .filter((p) => p.type === "text" && p.text)
            .map((p) => p.text)
            .join("\n"),
      id: msg.id,
      timestamp: msg.timestamp ? new Date(msg.timestamp).getTime() : undefined,
    }));
    
    log?.info?.(`[session-history] Got ${messages.length} messages via CLI`);
    
    // Filter out messages before the last /new session if enabled
    // This handles the case where OpenClaw returns old session history
    if (filterAfterNewSession) {
      messages = filterMessagesAfterLastNewSession(messages, log);
    }
    
    return {
      sessionKey,
      messages,
      hasMore: false,
    };
  } catch (err) {
    log?.error?.(`[session-history] CLI chat.history failed: ${err}`);
    throw err;
  }
}

/**
 * List sessions via CLI command
 * Uses: openclaw sessions --json (local, reads session store directly)
 * Falls back to: openclaw gateway call sessions.list --params '{...}' --json (via Gateway RPC)
 */
export async function listOpenClawSessionsViaCli(
  options?: {
    limit?: number;
    includeGlobal?: boolean;
    includeUnknown?: boolean;
    log?: PluginLogger;
    gatewayUrl?: string;
  },
): Promise<SessionsListResponse> {
  const { limit = 100, includeGlobal = false, includeUnknown = false, log, gatewayUrl } = options ?? {};
  
  log?.info?.(`[session-history] Listing sessions via CLI`);
  
  // Try local "openclaw sessions --json" first (no Gateway needed)
  try {
    const localArgs = ["--json"];
    const result = await executeClawCommand(["sessions"], localArgs, { log });
    
    const parsed = JSON.parse(result) as {
      sessions?: Array<{
        key: string;
        sessionId?: string;
        sessionFile?: string;
        updatedAt?: number;
        label?: string;
        displayName?: string;
      }>;
      count?: number;
    };
    
    let sessions: OpenClawSession[] = (parsed.sessions ?? []).map((s) => ({
      key: s.key,
      sessionId: s.sessionId,
      sessionFile: s.sessionFile,
      updatedAt: s.updatedAt,
      title: s.label || s.displayName,
    }));
    
    // Apply limit
    if (limit && sessions.length > limit) {
      sessions = sessions.slice(0, limit);
    }
    
    log?.info?.(`[session-history] Got ${sessions.length} sessions via CLI (local)`);
    
    return {
      sessions,
      total: parsed.count ?? sessions.length,
    };
  } catch (localErr) {
    log?.debug?.(`[session-history] Local sessions command failed, trying gateway RPC: ${localErr}`);
  }
  
  // Fall back to Gateway RPC: openclaw gateway call sessions.list --params '{...}' --json
  try {
    const params: Record<string, unknown> = {};
    if (limit) {
      params.limit = limit;
    }
    if (includeGlobal) {
      params.includeGlobal = true;
    }
    if (includeUnknown) {
      params.includeUnknown = true;
    }
    
    const args = ["--json", "--params", JSON.stringify(params)];
    
    const result = await executeClawCommand(
      ["gateway", "call", "sessions.list"],
      args,
      {
        log,
        gatewayUrl: gatewayUrl || sessionFileConfig.gatewayUrl,
      },
    );
    
    const parsed = JSON.parse(result) as {
      sessions?: Array<{
        key: string;
        sessionId?: string;
        sessionFile?: string;
        updatedAt?: number;
        label?: string;
        displayName?: string;
      }>;
      count?: number;
    };
    
    const sessions: OpenClawSession[] = (parsed.sessions ?? []).map((s) => ({
      key: s.key,
      sessionId: s.sessionId,
      sessionFile: s.sessionFile,
      updatedAt: s.updatedAt,
      title: s.label || s.displayName,
    }));
    
    log?.info?.(`[session-history] Got ${sessions.length} sessions via CLI (gateway RPC)`);
    
    return {
      sessions,
      total: parsed.count ?? sessions.length,
    };
  } catch (err) {
    log?.error?.(`[session-history] CLI sessions.list failed: ${err}`);
    throw err;
  }
}

// ============================================================================
// Auto Backend Selection
// ============================================================================

/**
 * Get chat history using the configured backend (auto, file, or cli)
 * @param sessionKey - Session key
 * @param options - Options
 * @returns Chat history
 */
export async function getChatHistory(
  sessionKey: string = "main",
  options?: {
    limit?: number;
    log?: PluginLogger;
  },
): Promise<ChatHistoryResponse> {
  // Only CLI backend is supported
  return getOpenClawChatHistoryViaCli(sessionKey, options);
}

/**
 * List sessions using the configured backend (auto, file, or cli)
 * @param options - Options
 * @returns Sessions list
 */
export async function listSessions(
  options?: {
    limit?: number;
    includeLastMessage?: boolean;
    includeDerivedTitles?: boolean;
    log?: PluginLogger;
    backend?: "file" | "cli" | "auto";
  },
): Promise<SessionsListResponse> {
  const backend = options?.backend || sessionFileConfig.backend || "auto";
  const { log } = options ?? {};
  
  if (backend === "cli") {
    return listOpenClawSessionsViaCli(options);
  }
  
  if (backend === "file") {
    return listOpenClawSessions(options);
  }
  
  // Auto: try file first, fall back to CLI
  try {
    const result = await listOpenClawSessions(options);
    if (result.sessions.length > 0) {
      return result;
    }
  } catch (err) {
    log?.debug?.(`[session-history] File backend failed, trying CLI: ${err}`);
  }
  
  // Fall back to CLI
  try {
    return await listOpenClawSessionsViaCli(options);
  } catch (err) {
    log?.warn?.(`[session-history] CLI backend also failed: ${err}`);
    return {
      sessions: [],
      total: 0,
    };
  }
}
