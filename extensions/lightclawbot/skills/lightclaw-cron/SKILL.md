---
name: lightclawbot-cron
description: LightClawBot 定时任务提醒。支持一次性提醒、周期性任务。可设置、查询、取消提醒。
---

# LightClawBot 智能提醒

让 AI 帮用户设置、管理定时提醒，通过 LightClawBot 通道投递消息。

---

## ⛔ 最重要的一条规则（读三遍）

> **调用 cron 工具时，payload.kind 必须是 `"agentTurn"`。绝对不能用 `"systemEvent"`！**
>
> `systemEvent` 只会在 AI 会话里插入一条文本，用户根本收不到消息。
> 只有 `agentTurn` + `deliver: true` + `channel: "lightclawbot"` + `to` 才能真正把消息发到 LightClaw。

---

## 🤖 AI 决策指南

### 时间确认规则

> 设置提醒前，先确认当前系统时间（查看上下文中的时间信息，或执行 `date`）。
> 纯相对时间（"5分钟后"、"1小时后"）可以跳过确认，直接算 `Date.now() + 延迟毫秒`。

### 用户意图识别

| 用户说法 | 意图 | cron 工具 action |
|----------|------|------------------|
| "5分钟后提醒我喝水" | 创建一次性提醒 | `add`（schedule.kind=at） |
| "每天8点提醒我打卡" | 创建周期提醒 | `add`（schedule.kind=cron） |
| "我有哪些提醒" | 查询 | `list` |
| "取消喝水提醒" | 删除 | `remove` |
| "修改提醒时间" | 删除+重建 | `remove` → `add` |
| "提醒我" (无时间) | **需追问** | 询问具体时间 |

### 必须追问的情况

1. **没有时间**："提醒我喝水" → "请问什么时候提醒你？"
2. **时间模糊**："晚点提醒我" → "具体几点呢？"
3. **周期不明**："定期提醒我" → "多久一次？每天？每周？"

---

## 📋 创建提醒（最重要）

### 🚨🚨🚨 工具调用参数模板（AI 必须严格遵循）

> **AI 调用 cron 工具时，传的是 JSON 参数，不是 CLI 命令。以下是唯一正确的 JSON 格式：**

**一次性提醒（N 分钟后）**：
```json
{
  "action": "add",
  "job": {
    "name": "{任务名}",
    "schedule": {
      "kind": "at",
      "atMs": {当前时间戳毫秒 + N分钟*60000}
    },
    "sessionTarget": "isolated",
    "wakeMode": "now",
    "deleteAfterRun": true,
    "payload": {
      "kind": "agentTurn",
      "message": "你是一个暖心的提醒助手。请用温暖、有趣的方式提醒用户：{提醒内容}。要求：(1) 不要回复HEARTBEAT_OK (2) 不要解释你是谁 (3) 直接输出一条暖心的提醒消息 (4) 可以加一句简短的鸡汤或关怀的话 (5) 控制在2-3句话以内 (6) 用emoji点缀",
      "deliver": true,
      "channel": "lightclawbot",
      "to": "lightclawbot:dm:{userId}"
    }
  }
}
```

**周期提醒（每天/每周）**：
```json
{
  "action": "add",
  "job": {
    "name": "{任务名}",
    "schedule": {
      "kind": "cron",
      "expr": "0 8 * * *",
      "tz": "Asia/Shanghai"
    },
    "sessionTarget": "isolated",
    "wakeMode": "now",
    "payload": {
      "kind": "agentTurn",
      "message": "你是一个暖心的提醒助手。请用温暖、有趣的方式提醒用户：{提醒内容}。要求：(1) 不要回复HEARTBEAT_OK (2) 不要解释你是谁 (3) 直接输出一条暖心的提醒消息 (4) 可以加一句简短的鸡汤或关怀的话 (5) 控制在2-3句话以内 (6) 用emoji点缀",
      "deliver": true,
      "channel": "lightclawbot",
      "to": "lightclawbot:dm:{userId}"
    }
  }
}
```

> 🚨 **绝对不可更改的 5 个字段**（改了提醒就废了）：
> 1. `payload.kind` 必须是 `"agentTurn"` — ❌ 绝对不能用 `"systemEvent"`
> 2. `payload.deliver` 必须是 `true`
> 3. `payload.channel` 必须是 `"lightclawbot"`
> 4. `payload.to` 必须是 `"lightclawbot:dm:{userId}"` 格式（userId 从系统消息获取）
> 5. `sessionTarget` 必须是 `"isolated"`
>
> 🚫 **`payload.kind: "systemEvent"` 只会在 AI 会话中注入文本，不会发送消息给用户！**
>
> ⚠️ **`schedule.atMs` 必须是绝对毫秒时间戳**（如 `1770733800000`），不支持相对时间字符串如 `"5m"`！
> 需要自行计算：`当前时间戳 + 延迟毫秒数`。例如 5 分钟后 = `Date.now() + 5 * 60 * 1000`。

### 查询提醒

使用 cron 工具 `action: "list"` 查询。

### 删除提醒

使用 cron 工具 `action: "remove"` + `jobId`。

---

## 💬 用户交互模板

> **创建提醒后的反馈要简洁友好，不要啰嗦**

### 创建成功反馈（推荐简洁版）

**一次性提醒**：
```
⏰ 好的，{时间}后提醒你{提醒内容}~
```

**周期提醒**：
```
⏰ 收到，{周期描述}提醒你{提醒内容}~
```

### 查询提醒反馈

```
📋 你的提醒：

1. ⏰ {提醒名} - {时间}
2. 🔄 {提醒名} - {周期}

说"取消xx提醒"可删除~
```

### 无提醒时反馈

```
📋 目前没有提醒哦~

说"5分钟后提醒我xxx"试试？
```

### 删除成功反馈

```
✅ 已取消"{提醒名称}"
```

---

## ⏱️ 时间格式

### 一次性提醒（schedule.kind = "at"）

> ⚠️ `schedule.atMs` 只接受**绝对毫秒时间戳**，需要自己计算！

| 用户说法 | 计算方式 |
|----------|----------|
| 5分钟后 | `Date.now() + 5 * 60 * 1000` |
| 半小时后 | `Date.now() + 30 * 60 * 1000` |
| 1小时后 | `Date.now() + 60 * 60 * 1000` |
| 明天早上8点 | 先确认当前日期，计算目标时间的毫秒时间戳 |

### 周期提醒（schedule.kind = "cron"）

> 必须加 `"tz": "Asia/Shanghai"`

| 用户说法 | schedule.expr |
|----------|---------------|
| 每天早上8点 | `"0 8 * * *"` |
| 每天晚上10点 | `"0 22 * * *"` |
| 每个工作日早上9点 | `"0 9 * * 1-5"` |
| 每周一早上9点 | `"0 9 * * 1"` |
| 每周末上午10点 | `"0 10 * * 0,6"` |
| 每小时整点 | `"0 * * * *"` |

---

## 📌 参数说明

### 工具调用 job 对象必填字段

| 字段 | 说明 | 示例 |
|------|------|------|
| `job.name` | 任务名 | `"喝水提醒"` |
| `job.schedule.kind` | `"at"` 或 `"cron"` | `"at"` |
| `job.schedule.atMs` | **绝对毫秒时间戳**（不支持 `"5m"`！） | `1770734100000` |
| `job.sessionTarget` | 必须 `"isolated"` | `"isolated"` |
| `job.wakeMode` | 推荐 `"now"` | `"now"` |
| `job.payload.kind` | 必须 `"agentTurn"`（❌ 不能用 `"systemEvent"`） | `"agentTurn"` |
| `job.payload.message` | 提醒内容的 prompt | `"你是一个暖心的提醒助手..."` |
| `job.payload.deliver` | 必须 `true` | `true` |
| `job.payload.channel` | 必须 `"lightclawbot"` | `"lightclawbot"` |
| `job.payload.to` | 用户地址 | `"lightclawbot:dm:{userId}"` |
| `job.deleteAfterRun` | 一次性任务必须 `true` | `true` |

### payload.message 暖心提醒模板

> 💡 **`payload.message` 是一个 prompt，告诉 AI 以暖心方式生成提醒**。每次触发时 AI 会自由发挥，生成不重复的、有温度的提醒消息。

**统一模板**（把 `{提醒内容}` 替换成具体事项）：
```
你是一个暖心的提醒助手。请用温暖、有趣的方式提醒用户：{提醒内容}。要求：(1) 不要回复HEARTBEAT_OK (2) 不要解释你是谁 (3) 直接输出一条暖心的提醒消息 (4) 可以加一句简短的鸡汤或关怀的话 (5) 控制在2-3句话以内 (6) 用emoji点缀
```

---

## 🎯 使用场景示例

### 场景1：一次性提醒

**用户**: 5分钟后提醒我喝水

**AI 调用 cron 工具**（假设当前时间戳为 1770734000000）:
```json
{
  "action": "add",
  "job": {
    "name": "喝水提醒",
    "schedule": { "kind": "at", "atMs": 1770734300000 },
    "sessionTarget": "isolated",
    "wakeMode": "now",
    "deleteAfterRun": true,
    "payload": {
      "kind": "agentTurn",
      "message": "你是一个暖心的提醒助手。请用温暖、有趣的方式提醒用户：该喝水了。要求：(1) 不要回复HEARTBEAT_OK (2) 不要解释你是谁 (3) 直接输出一条暖心的提醒消息 (4) 可以加一句简短的鸡汤或关怀的话 (5) 控制在2-3句话以内 (6) 用emoji点缀",
      "deliver": true,
      "channel": "lightclawbot",
      "to": "lightclawbot:dm:{userId}"
    }
  }
}
```
> `atMs` = 当前时间戳 + 5 * 60 * 1000 = 1770734000000 + 300000 = 1770734300000

**AI 回复**: `⏰ 好的，5分钟后提醒你喝水~`

---

### 场景2：每日周期提醒

**用户**: 每天早上8点提醒我打卡

**AI 调用 cron 工具**:
```json
{
  "action": "add",
  "job": {
    "name": "打卡提醒",
    "schedule": { "kind": "cron", "expr": "0 8 * * *", "tz": "Asia/Shanghai" },
    "sessionTarget": "isolated",
    "wakeMode": "now",
    "payload": {
      "kind": "agentTurn",
      "message": "你是一个暖心的提醒助手。请用温暖、有趣的方式提醒用户：该打卡了。要求：(1) 不要回复HEARTBEAT_OK (2) 不要解释你是谁 (3) 直接输出一条暖心的提醒消息 (4) 可以加一句简短的鸡汤或关怀的话 (5) 控制在2-3句话以内 (6) 用emoji点缀",
      "deliver": true,
      "channel": "lightclawbot",
      "to": "lightclawbot:dm:{userId}"
    }
  }
}
```
> 周期任务**不加** `deleteAfterRun`

**AI 回复**: `⏰ 收到，每天早上8点提醒你打卡~`

---

### 场景3：工作日提醒

**用户**: 工作日下午6点提醒我写日报

**AI 调用 cron 工具**:
```json
{
  "action": "add",
  "job": {
    "name": "日报提醒",
    "schedule": { "kind": "cron", "expr": "0 18 * * 1-5", "tz": "Asia/Shanghai" },
    "sessionTarget": "isolated",
    "wakeMode": "now",
    "payload": {
      "kind": "agentTurn",
      "message": "你是一个暖心的提醒助手。请用温暖、有趣的方式提醒用户：该写日报了。要求：(1) 不要回复HEARTBEAT_OK (2) 不要解释你是谁 (3) 直接输出一条暖心的提醒消息 (4) 可以加一句简短的鸡汤或关怀的话 (5) 控制在2-3句话以内 (6) 用emoji点缀",
      "deliver": true,
      "channel": "lightclawbot",
      "to": "lightclawbot:dm:{userId}"
    }
  }
}
```

**AI 回复**: `⏰ 收到，工作日下午6点提醒你写日报~`

---

### 场景4：查询提醒

**用户**: 我有哪些提醒？

**AI 调用 cron 工具**：`{ "action": "list" }`

**AI 回复**（根据返回结果）:
```
📋 你的提醒：

1. ⏰ 喝水提醒 - 3分钟后
2. 🔄 打卡提醒 - 每天08:00

说"取消xx提醒"可删除~
```

---

### 场景5：取消提醒

**用户**: 取消打卡提醒

**AI 执行**:
1. 先用 `{ "action": "list" }` 找到对应任务 ID
2. 再用 `{ "action": "remove", "jobId": "{id}" }` 删除

**AI 回复**:
```
✅ 已取消"打卡提醒"
```

---

## ⏱️ 消息发送说明

### 定时提醒（cron add）

定时提醒通过隔离会话 + deliver 机制发送：

```
┌─────────────────────┐
│ 定时任务触发         │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ AI 通过 agentTurn   │
│ 在隔离会话中执行     │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ deliver=true 投递    │
│ channel="lightclawbot" │
│ to="lightclawbot:dm:…" │
└──────────┬──────────┘
           ↓
    ✅ 用户收到提醒
```

---

## 🔧 用户标识

| 类型 | 格式 | 来源 |
|------|------|------|
| 用户 ID | 用户的 clientId | 系统消息自动提供 |
| 用户地址 | `lightclawbot:dm:{userId}` | 从 SenderId 构建 |

> 💡 这些信息在系统消息中格式如：
> - `当前用户 SenderId: {userId}`
> - 用户地址格式为 `lightclawbot:dm:{userId}`
