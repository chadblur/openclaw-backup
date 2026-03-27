# LightClawBot

[OpenClaw](https://github.com/openclaw/openclaw) 的 Channel 插件

[![npm version](https://img.shields.io/npm/v/lightclawbot)](https://www.npmjs.com/package/lightclawbot)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 特性

- ⚡ 流式回复，AI 响应实时推送
- ⏰ 定时任务，支持一次性提醒和周期性 Cron 任务
- 💬 主动消息推送和历史消息读取

## 安装

```bash
openclaw plugins install lightclawbot
```

## 配置

编辑 `~/.openclaw/openclaw.json`：

```json
{
  "channels": {
    "lightclawbot": {
      "apiKeys": ["key-1", "key-2"],
      "enabled": true
    }
  }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `apiKeys` | `string[]` | API Key 数组，每个对应一个用户身份 |
| `enabled` | `boolean` | 是否启用 |

## 开发

```bash
npm run build        # 编译
npm run typecheck    # 类型检查
```

项目附带 Mock 用于本地调试：

```bash
cd test-server
npm install
npm start            # 启动 Mock 服务端
npm run client       # 启动测试客户端
```

## 项目结构

```
src/
├── channel.ts          # 插件入口，注册、能力声明、生命周期
├── config.ts           # 配置解析，API Key 映射管理
├── gateway.ts          # Socket.IO 连接管理，心跳，重连
├── inbound.ts          # 入站消息处理 → 文件处理 → AI 分发
├── outbound.ts         # 出站消息（WebSocket）
├── socket-handlers.ts  # Socket.IO 事件绑定
├── socket-registry.ts  # Socket 注册表，离线缓冲
├── file-storage.ts     # 文件存储封装
├── upload-tool.ts      # AI 工具：上传文件
├── download-tool.ts    # AI 工具：下载/获取文件 URL
├── dedup.ts            # 消息去重与防抖
├── media.ts            # 媒体工具函数
├── format-urls.ts      # 文件 URL 自动转 Markdown 链接
├── types.ts            # 类型定义
└── history/            # 历史消息读取
```
