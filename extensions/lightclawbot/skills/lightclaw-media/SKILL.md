---
name: lightclawbot-media
description: LightClawBot 文件收发能力。用户发来的文件自动下载并保存，AI 生成的文件通过 lightclaw_upload_file 上传后以标准 Markdown 链接返回给用户。禁止使用其他存储工具。
---

# LightClawBot 文件上传与下载

让 AI 帮用户处理文件的上传、下载和分享，通过 LightClawBot 通道投递。

---

## ⛔ 最重要的规则（读三遍）

> 1. **上传文件必须使用 `lightclaw_upload_file` 工具，禁止使用其他任何存储工具！**
> 3. **返回文件给用户时，必须使用标准 Markdown 链接格式：`[文件名](下载链接)`**
>    - ❌ 错误：`下载链接: https://xxx`
>    - ❌ 错误：`📎 文件下载链接: https://xxx`
>    - ✅ 正确：`[report.pdf](https://lightai.cloud.tencent.com/drive/preview?filePath=2026-03-15/report.pdf)`

---

## 🔧 可用工具

本技能使用以下两个专属工具，**不要使用其他任何文件/存储工具**：

| 工具名 | 用途 | 何时使用 |
|--------|------|----------|
| `lightclaw_upload_file` | 上传本地文件到云端，获取公网下载链接 | AI 生成了文件需要分享给用户时 |

---

## 📤 上传文件（lightclaw_upload_file）

### 功能

将本地文件上传到云端存储，返回公网可访问的下载链接。支持批量上传（最多 5 个文件）。

### 参数

```json
{
  "paths": ["/absolute/path/to/file1.pdf", "/absolute/path/to/file2.xlsx"]
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `paths` | string[] | ✅ | 本地文件绝对路径数组，最多 5 个 |

### ⚠️ 关键约束

1. **必须使用绝对路径**（以 `/` 开头），禁止使用相对路径
   - ❌ `./output/report.pdf`
   - ✅ `/Users/xxx/.openclaw/workspace/output/report.pdf`
2. **文件必须存在**：上传前确认文件已生成并写入磁盘
3. **单次最多 5 个文件**：超出需分批上传

### 使用示例

**上传单个文件**：
```json
{
  "paths": ["/Users/xxx/.openclaw/workspace/report.pdf"]
}
```

**批量上传**：
```json
{
  "paths": [
    "/Users/xxx/.openclaw/workspace/chart1.png",
    "/Users/xxx/.openclaw/workspace/chart2.png",
    "/Users/xxx/.openclaw/workspace/summary.xlsx"
  ]
}
```

---

## 📋 返回文件给用户的格式（最重要）

### 🚨🚨🚨 必须使用标准 Markdown 链接格式

> **当需要将文件/文档/图片链接返回给用户时，必须使用标准 Markdown 链接格式：**
>
> ```
> [<文件名>](<下载链接>)
> ```

### ✅ 正确示例

**单个文件**：

```
好的，报告已生成，请点击下载：

[report.pdf](https://lightai.cloud.tencent.com/drive/preview?filePath=2026-03-15/report.pdf)
```

**多个文件**：

```
所有文件已准备好：

- [数据分析报告.pdf](https://lightai.cloud.tencent.com/drive/preview?filePath=2026-03-15/数据分析报告.pdf)
- [原始数据.xlsx](https://lightai.cloud.tencent.com/drive/preview?filePath=2026-03-15/原始数据.xlsx)
- [趋势图.png](https://lightai.cloud.tencent.com/drive/preview?filePath=2026-03-15/趋势图.png)
```

**附带说明文字**：

```
✅ 周报已生成完毕！

[本周工作周报.pdf](https://lightai.cloud.tencent.com/drive/preview?filePath=2026-03-15/本周工作周报.pdf)

报告包含了本周的进度汇总和下周计划。
```

### ❌ 错误示例（绝对不要这样做）

❌ **裸链接**：
```
下载链接: https://lightai.cloud.tencent.com/drive/preview?filePath=2026-03-15/report.pdf
```

❌ **带 emoji 前缀的裸链接**：
```
📎 文件下载链接: https://lightai.cloud.tencent.com/drive/preview?filePath=2026-03-15/report.pdf
```

❌ **使用代码块包裹链接**：
```
`https://lightai.cloud.tencent.com/drive/preview?filePath=2026-03-15/report.pdf`
```

❌ **只返回路径，没有文件名**：
```
[下载](https://lightai.cloud.tencent.com/drive/preview?filePath=2026-03-15/report.pdf)
```

---

## 📸 接收用户文件

用户通过 LightClawBot 发来的文件（图片、文档等）**自动下载到本地**。文件信息在上下文的【Attachments】字段中，包含：

| 字段 | 说明 |
|------|------|
| `name` | 文件名 |
| `mimeType` | 文件 MIME 类型 |
| `url` | 文件的公网下载链接 |

本地文件路径在上下文的 `MediaPath` / `MediaPaths` 字段中。

### 处理用户文件的流程

1. **查看上下文**：从 Attachments 获取文件信息
2. **本地处理**：使用 `MediaPath` 中的本地路径读取和处理文件
3. **如需返回处理后的文件**：用 `lightclaw_upload_file` 上传 → 用 Markdown 链接返回

---

## 🎯 完整使用场景

### 场景 1：用户要求生成报告

**用户**：帮我生成一份数据分析报告

**AI 执行步骤**：
1. 生成报告文件并保存到本地（如 `/Users/xxx/.openclaw/workspace/数据分析报告.pdf`）
2. 调用 `lightclaw_upload_file` 上传：
   ```json
   { "paths": ["/Users/xxx/.openclaw/workspace/数据分析报告.pdf"] }
   ```
3. 获取到下载链接后，以 Markdown 格式回复：
   ```
   ✅ 报告已生成：

   [数据分析报告.pdf](https://lightai.cloud.tencent.com/drive/preview?filePath=2026-03-15/数据分析报告.pdf)
   ```

### 场景 2：用户要求处理发来的文件

**用户**：帮我把这个 Excel 转成 PDF（附带了 data.xlsx）

**AI 执行步骤**：
1. 从上下文 `MediaPath` 获取本地文件路径
2. 读取并处理 Excel 文件，转换为 PDF
3. 调用 `lightclaw_upload_file` 上传 PDF
4. 以 Markdown 格式回复：
   ```
   ✅ 已将 Excel 转为 PDF：

   [data.pdf](https://lightai.cloud.tencent.com/drive/preview?filePath=2026-03-15/data.pdf)
   ```

### 场景 3：用户要求获取之前上传的文件

**用户**：我昨天上传的报告还能下载吗？

**AI 执行步骤**：
1. 以 Markdown 格式回复：
   ```
   可以的，这是下载链接：

   [report.pdf](https://lightai.cloud.tencent.com/drive/preview?filePath=2026-03-14/report.pdf)
   ```

### 场景 4：批量上传多个文件

**用户**：帮我生成三张数据图表

**AI 执行步骤**：
1. 生成三张图表并保存到本地
2. 调用 `lightclaw_upload_file` 批量上传：
   ```json
   {
     "paths": [
       "/Users/xxx/.openclaw/workspace/chart1.png",
       "/Users/xxx/.openclaw/workspace/chart2.png",
       "/Users/xxx/.openclaw/workspace/chart3.png"
     ]
   }
   ```
3. 以 Markdown 格式回复：
   ```
   ✅ 三张图表已生成：

   - [chart1.png](https://lightai.cloud.tencent.com/drive/preview?filePath=2026-03-15/chart1.png)
   - [chart2.png](https://lightai.cloud.tencent.com/drive/preview?filePath=2026-03-15/chart2.png)
   - [chart3.png](https://lightai.cloud.tencent.com/drive/preview?filePath=2026-03-15/chart3.png)
   ```

## 🚫 禁止事项

1. **❌ 禁止使用其他存储工具**：所有文件操作必须通过 `lightclaw_upload_file`
2. **❌ 禁止返回裸链接**：必须使用 `[文件名](链接)` 格式
3. **❌ 禁止说"无法发送文件"**：你有能力上传和分享任何本地文件
4. **❌ 禁止使用相对路径**：工具参数中的文件路径必须是绝对路径
5. **❌ 禁止使用 message tool 发送文件**：直接在回复文本中写 Markdown 链接

---

## 📌 快速参考

| 场景 | 工具 | 回复格式 |
|------|------|----------|
| 上传生成的文件 | `lightclaw_upload_file` | `[文件名](下载链接)` |
| 接收用户文件 | 自动处理，查看上下文 Attachments | 确认收到并告知文件信息 |
