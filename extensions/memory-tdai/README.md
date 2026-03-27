# @tdai/memory-tdai

**Four-layer local memory system plugin for [OpenClaw](https://github.com/openclaw/openclaw).**

为 AI Agent 提供完全本地化的长期记忆能力。通过 L0→L1→L2→L3 四层渐进式管线，自动将对话内容提炼为结构化记忆、场景块和用户画像，无需任何外部 API 依赖。

## ✨ 核心功能

- **L0 — 对话录制**：自动捕获每轮对话原始消息到本地 SQLite + JSONL 双写
- **L1 — 记忆提取**：由本地 LLM 从对话中提取结构化记忆，支持向量去重与冲突检测
- **L2 — 场景归纳**：基于 L1 记忆自动归纳场景块（Scene Block），由 LLM 增量提取
- **L3 — 用户画像**：基于场景块自动生成/更新用户画像（Persona）
- **自动召回（Auto-Recall）**：对话开始前自动注入相关记忆和用户画像到上下文
- **语义搜索工具**：Agent 可调用 `tdai_memory_search`（L1 记忆搜索）和 `tdai_conversation_search`（L0 对话搜索）
- **向量搜索**：基于 sqlite-vec + node-llama-cpp 的本地 embedding，支持 hybrid（关键词 + 向量 RRF 融合）搜索策略
- **Session 隔离**：不同渠道/Agent 的对话独立调度、独立提取
- **本地数据清理**：可配置 L0/L1 数据保留天数，定时自动清理过期文件
- **支持纯本地**：支持纯本地部署，不强依赖任何外部 API 依赖
- **支持零配置**：支持零配置工作，简单易用

## 🏗️ 关键原理

```
对话开始
  → Auto-Recall: 向量/混合搜索相关记忆 + 加载 Persona → 注入系统上下文

对话结束
  → Auto-Capture (L0): 录制对话消息 → SQLite vec0 + JSONL 双写
  → Pipeline Scheduler: 达到 N 轮后按序触发 L1 → L2 → L3
     ├── L1: 本地 LLM 提取结构化记忆 + 向量去重 → 写入 JSONL + SQLite
     ├── L2: 本地 LLM 归纳场景块 → Markdown 文件
     └── L3: 本地 LLM 生成/更新用户画像 → persona.md
```

### 数据目录结构

```
<pluginDataDir>/
├── conversations/     — L0 每日 JSONL 分片（每行一条消息）
├── records/           — L1 每日 JSONL 分片（提取的记忆）
├── scene_blocks/      — L2 场景块 .md 文件
├── vectors.db         — SQLite + vec0 向量数据库
├── .metadata/         — checkpoint, scene_index.json
└── .backup/           — 滚动备份（persona, scene_blocks）
```

## 📋 前置依赖

| 依赖 | 版本要求 | 说明 |
|------|---------|------|
| [OpenClaw](https://github.com/nicepkg/openclaw) | `>= 2026.3.13` | 宿主框架，提供插件 SDK 及 Gateway 运行环境 |
| [Node.js](https://nodejs.org/) | `>= 22.16.0` | 运行时环境 |
| [`node-llama-cpp`](https://github.com/withcatai/node-llama-cpp) | `^3.16.2` | 本地 embedding 模型（GGUF 格式），提供离线向量化能力。首次运行会自动下载模型 |
| [`sqlite-vec`](https://github.com/asg017/sqlite-vec) | `0.1.7-alpha.2` | SQLite 向量搜索扩展，提供 KNN 近邻查询 |

> `node-llama-cpp` 和 `sqlite-vec` 作为 dependencies 会随插件自动安装，无需手动处理。

## 📦 安装

```bash
# 安装插件
openclaw plugins install @tdai/memory-tdai

# 更新插件
openclaw plugins update memory-tdai

# 卸载插件
openclaw plugins uninstall memory-tdai
```

安装完成后，**重启 Gateway** 使插件生效：

```bash
openclaw gateway restart
```

## ⚙️ 配置

插件配置位于 `~/.openclaw/openclaw.json` 中 `memory-tdai` 字段下。**所有字段均有合理默认值，零配置即可使用。**

### 最小配置
最小配置，安装启用后即为该状态
```json
{
  "memory-tdai": {
    "enabled": true
  }
}
```

### 完整配置
用户可按需配置，提升使用体验

```json
{
  "memory-tdai": {
    "capture": {
      "enabled": true,
      "excludeAgents": ["bench-judge-*"],
      "l0l1RetentionDays": 90,
      "cleanTime": "03:00"
    },
    "extraction": {
      "enabled": true,
      "enableDedup": true,
      "maxMemoriesPerSession": 10,
      "model": "provider/model-name"
    },
    "pipeline": {
      "everyNConversations": 5,
      "enableWarmup": true,
      "l1IdleTimeoutSeconds": 60,
      "l2DelayAfterL1Seconds": 90,
      "l2MinIntervalSeconds": 300,
      "l2MaxIntervalSeconds": 1800,
      "sessionActiveWindowHours": 24
    },
    "recall": {
      "enabled": true,
      "maxResults": 5,
      "scoreThreshold": 0.3,
      "strategy": "hybrid"
    },
    "embedding": {
      "enabled": true,
      "provider": "local", 
      "model": "",
      "dimensions": 768
    },
    "persona": {
      "triggerEveryN": 50,
      "maxScenes": 20,
      "backupCount": 3,
      "sceneBackupCount": 10,
      "model": "provider/model-name"
    }
  }
}
```

### 配置说明

#### capture — 对话捕获 (L0)

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | boolean | `true` | 是否启用自动对话捕获 |
| `excludeAgents` | string[] | `[]` | Agent 排除 glob 模式列表，匹配的 agent 不参与捕获/召回/调度 |
| `l0l1RetentionDays` | number | `0` | L0/L1 本地文件保留天数。`0` = 不清理；非 0 时需 >= 3（除非开启 `allowAggressiveCleanup`） |
| `allowAggressiveCleanup` | boolean | `false` | 是否允许 1–2 天的高风险清理配置 |
| `cleanTime` | string | `"03:00"` | 每日清理执行时间（HH:mm 格式） |

#### extraction — 记忆提取 (L1)

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | boolean | `true` | 是否启用后台记忆提取 |
| `enableDedup` | boolean | `true` | 启用 L1 智能去重（基于向量相似度冲突检测） |
| `maxMemoriesPerSession` | number | `10` | 单次 L1 提取每 session 最大记忆条数 |
| `model` | string | *(默认)* | 提取使用的 LLM 模型（格式：`provider/model`） |

#### pipeline — 管线调度 (L1→L2→L3)

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `everyNConversations` | number | `5` | 每 N 轮对话触发一次 L1 批处理 |
| `enableWarmup` | boolean | `true` | Warm-up 模式：新 session 从 1 轮触发开始，每次 L1 后翻倍（1→2→4→...→N） |
| `l1IdleTimeoutSeconds` | number | `60` | 用户停止对话后多久触发 L1（秒） |
| `l2DelayAfterL1Seconds` | number | `90` | L1 完成后延迟多久触发 L2（秒） |
| `l2MinIntervalSeconds` | number | `300` | 同一 session 两次 L2 的最小间隔（秒） |
| `l2MaxIntervalSeconds` | number | `1800` | 活跃 session 的 L2 最大轮询间隔（秒） |
| `sessionActiveWindowHours` | number | `24` | 超过此时间不活跃的 session 停止 L2 轮询 |

#### recall — 记忆召回

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | boolean | `true` | 是否启用对话前自动召回 |
| `maxResults` | number | `5` | 召回最大结果数 |
| `scoreThreshold` | number | `0.3` | 最低分数阈值 |
| `strategy` | string | `"hybrid"` | 搜索策略：`keyword`（关键词）、`embedding`（向量）、`hybrid`（混合 RRF 融合，推荐） |

#### embedding — 向量搜索

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | boolean | `true` | 是否启用向量搜索 |
| `provider` | string | `"local"` | Embedding 提供者：`local`（node-llama-cpp 本地模型）或 `openai`（OpenAI 兼容 API） |
| `baseUrl` | string | — | API Base URL（远端模式必填）：填写对应 provider 的 API 地址 |
| `apiKey` | string | — | API Key（远端模式必填） |
| `model` | string | *(自动)* | 模型名称（`local` 留空自动选择；远端模式必填） |
| `dimensions` | number | *(自动)* | 向量维度（`local` = 768；远端模式必填，需与所选模型匹配） |
| `conflictRecallTopK` | number | `5` | 冲突检测时召回 Top-K 数 |
| `modelCacheDir` | string | — | 本地模型缓存目录（仅 `local` provider） |

#### persona — 场景归纳与用户画像 (L2/L3)

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `triggerEveryN` | number | `50` | 每 N 条新记忆触发一次画像生成 |
| `maxScenes` | number | `20` | 最大场景块数量 |
| `backupCount` | number | `3` | 画像备份保留数量 |
| `sceneBackupCount` | number | `10` | 场景块备份保留数量 |
| `model` | string | *(默认)* | L2/L3 使用的 LLM 模型（格式：`provider/model`） |

## 🔧 Agent 工具

插件注册了两个 Agent 可调用的工具：

### `tdai_memory_search`

搜索用户的 L1 结构化长期记忆。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | ✅ | 搜索查询 |
| `limit` | number | — | 返回结果上限（默认 5，最大 20） |
| `type` | string | — | 按记忆类型过滤：`persona` / `episodic` / `instruction` |
| `scene` | string | — | 按场景名过滤 |

### `tdai_conversation_search`

搜索 L0 原始对话历史。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | ✅ | 搜索查询 |
| `limit` | number | — | 返回结果上限（默认 5，最大 20） |
| `session_key` | string | — | 按 session 过滤 |

## 📁 数据与日志

- **数据目录**：`~/.openclaw/state/memory-tdai/`（自动创建）
- **Gateway 日志**：插件运行日志通过 `[memory-tdai]` 前缀标记，可在 Gateway 日志中搜索查看
- **配置文件**：`~/.openclaw/openclaw.json`

## 📄 License

[MIT](LICENSE)
