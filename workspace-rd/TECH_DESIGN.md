# TECH_DESIGN — FEISHU-20260317-001

任务：飞书群消息@提及兜底（把 `@ou_xxx` 自动转为 `<at ...></at>`）

## 背景
当前 pm/系统模板可能输出纯文本 `@ou_...`，飞书端不可点击（不触发 mention），导致派发触达失败。
虽然已在模板层修正为 `<at user_id="..."></at>`，仍需要在 openclaw-lark 出站链路做兜底，避免模型回退/非预期输出。

## 目标/验收标准
1. 覆盖三类出站：
   - text（OpenClaw message.tool `sendText` 路径，最终在飞书以 `post` 发送）
   - post（同上，内部就是 `post`）
   - card（`interactive`）
2. 识别与转换：把 `@ou_\w+`（实际 open_id 格式 `ou_` + base62）转换为可点击的 mention。
3. 保留已有 `<at ...>`：不重复包裹，不破坏已有 at 标签。
4. 不误伤：不要把邮箱/URL 中的 `@` 或 `ou_` 片段转换。
5. 可配置开关：允许禁用兜底（默认开启）。

## 方案
### 1) text/post 路径
在 `src/messaging/outbound/deliver.js` 的文本预处理 `prepareTextForLark` 中增加一步：
- 在 `normalizeAtMentions()` 之后执行 `convertOpenIdMentions(processed, 'text')`
- 输出格式：`<at user_id="ou_xxx"></at>`（name 为空也可点击）

### 2) card 路径
在 `sendCardLark` 发送前对 card JSON 做深度遍历：
- 仅对 string 叶子节点做替换
- 跳过常见 URL/资源字段（url/href/src/img_key/...）避免误伤
- 替换格式：`<at id=ou_xxx></at>`

### 3) 误伤规避策略（保守）
- 仅当 `@` 前一个字符为：行首/空白/开括号类标点 时才转换
  - 避免 `https://.../@ou_xxx`、`foo=@ou_xxx` 等 URL/query 误匹配
- 仅当 open_id 后面是边界字符（空白/标点/行尾）才转换
- 如果整段字符串看起来像 URL 或包含邮箱特征，则 card string 直接跳过

### 4) 配置开关
新增配置：`channels.feishu.mentionFallback?: boolean`
- 未配置 => 默认 true
- 配置为 false => 完全不做 `@ou_...` 转换
- 支持按 accountId 覆盖（由于 account config merge 机制，账号级覆盖天然支持）

## 影响范围 & 风险
- 只影响出站渲染，不改变入站。
- 采用保守匹配可能漏转极端场景（比如紧贴中文文本不带空格的 `张三@ou_xxx`），但优先保证不误伤 URL/邮箱。

## 回滚
- 将 `channels.feishu.mentionFallback` 设为 false 可快速关闭该能力。
