# CHANGELOG — FEISHU-20260317-001

## 变更摘要
- openclaw-lark 出站消息预处理增加 @mention 兜底：将纯文本 `@ou_xxx` 转换为可点击的 Feishu mention 标签。

## 具体变更
- 新增 `src/messaging/outbound/mention-fallback.js`
  - `convertOpenIdMentions(text, mode)`：text/post 使用 `<at user_id="..."></at>`；card 使用 `<at id=...></at>`
  - `convertOpenIdMentionsInCard(card)`：深度遍历替换字符串叶子，跳过 URL/资源字段
  - `isMentionFallbackEnabled(larkClient)`：读取 `channels.feishu.mentionFallback`（默认 true）
- 修改 `src/messaging/outbound/deliver.js`
  - sendTextLark：文本预处理时执行兜底转换
  - sendCardLark：发送前对 card JSON 做兜底转换
- 修改 `src/core/config-schema.js`
  - 增加配置 `mentionFallback?: boolean`

## 影响范围
- 仅影响 Feishu/Lark 渠道出站消息渲染。

## 回滚方案
- 配置层设置 `channels.feishu.mentionFallback: false` 关闭。
- 或回退上述文件变更。
