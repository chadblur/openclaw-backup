# SELF_TEST — FEISHU-20260317-001

## 环境
- 插件：`@larksuite/openclaw-lark`
- 修改点：出站消息预处理（text/post/card）

## 用例
### Case 1：text/post 基础转换
输入：
- `你好 @ou_ea12f8080576101b89e65bf6ee9fb6fd 请看一下`
期望：
- 出站 payload 中出现 `<at user_id="ou_ea12f..."></at>`

### Case 2：保留已有 at 标签
输入：
- `你好 <at user_id="ou_xxx"></at> @ou_yyy`
期望：
- `<at ...>` 保持不变
- `@ou_yyy` 被转为 `<at user_id="ou_yyy"></at>`

### Case 3：不误伤 URL
输入：
- `打开 https://example.com/@ou_abc 或者 http://a.com?q=@ou_abc`
期望：
- 不转换（或至少不把 URL 中的片段转换）

### Case 4：不误伤邮箱
输入：
- `联系 foo@ou_abc.com 或 a@b.com`
期望：
- 不转换

### Case 5：card JSON 深度替换
输入 card（任意 markdown/content 字段含 `@ou_xxx`）：
- body.elements[].content = `hi @ou_xxx`
期望：
- 替换为 `<at id=ou_xxx></at>`
- url/href/src 等字段不替换

## 手工验证截图
- 待补（需要在测试群实际发送验证 mention 可点击）
