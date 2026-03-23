import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { emptyPluginConfigSchema } from "openclaw/plugin-sdk";

import { handleWecomWebhookRequest } from "./src/monitor.js";
import { setWecomRuntime } from "./src/runtime.js";
import { wecomPlugin } from "./src/channel.js";
import { createWeComMcpTool } from "./src/mcp/index.js";

const plugin = {
  id: "wecom",
  name: "WeCom",
  description: "OpenClaw WeCom (WeChat Work) intelligent bot channel plugin",
  configSchema: emptyPluginConfigSchema(),
  /**
   * **register (注册插件)**
   *
   * OpenClaw 插件入口点。
   * 1. 注入 Runtime 环境 (api.runtime)。
   * 2. 注册 WeCom 渠道插件 (ChannelPlugin)。
   * 3. 注册 Webhook HTTP 路由（推荐 /plugins/wecom/*，兼容 /wecom*）。
   * 4. 注册 wecom_mcp 工具 (MCP Streamable HTTP 调用)。
   * 5. 注入 MEDIA 指令提示词（仅 wecom 通道），指导 LLM 使用 MEDIA: 语法发送文件。
   */
  register(api: OpenClawPluginApi) {
    setWecomRuntime(api.runtime);
    api.registerChannel({ plugin: wecomPlugin });
    const routes = ["/plugins/wecom", "/wecom"];
    for (const path of routes) {
      api.registerHttpRoute({
        path,
        handler: handleWecomWebhookRequest,
        auth: "plugin",
        match: "prefix",
      });
    }

    // 注册 wecom_mcp：通过 HTTP 直接调用企业微信 MCP Server
    api.registerTool(createWeComMcpTool(), { name: "wecom_mcp" });

    // 注入媒体发送指令和文件大小限制提示词（与官方 @wecom/wecom-openclaw-plugin 保持一致）。
    // 仅 wecom 通道注入，避免影响其他通道。
    // deliver 回调中保留兜底正则解析，作为双重保障。
    api.on("before_prompt_build", (_event, ctx) => {
      if (ctx.channelId !== "wecom") return;
      return {
        prependContext: [
          "【发送文件/图片/视频/语音】",
          "当你需要向用户发送文件、图片、视频或语音时，必须在回复中单独一行使用 MEDIA: 指令，后面跟文件的本地路径。",
          "格式：MEDIA: /文件的绝对路径",
          "文件优先存放到 ~/.openclaw 目录下，确保路径可访问。",
          "示例：",
          "  MEDIA: ~/.openclaw/output.png",
          "  MEDIA: ~/.openclaw/report.pdf",
          "系统会自动识别文件类型并发送给用户。",
          "",
          "注意事项：",
          "- MEDIA: 必须在行首，后面紧跟文件路径（不是 URL）",
          "- 如果路径中包含空格，可以用反引号包裹：MEDIA: `/path/to/my file.png`",
          "- 每个文件单独一行 MEDIA: 指令",
          "- 可以在 MEDIA: 指令前后附带文字说明",
          "",
          "【文件大小限制】",
          "- 图片不超过 10MB，视频不超过 10MB，语音不超过 2MB（仅支持 AMR 格式），文件不超过 20MB",
          "- 语音消息仅支持 AMR 格式（.amr），如需发送语音请确保文件为 AMR 格式",
          "- 超过大小限制的图片/视频/语音会被自动转为文件格式发送",
          "- 如果文件超过 20MB，将无法发送，请提前告知用户并尝试缩减文件大小",
        ].join("\n"),
      };
    });
  },
};

export default plugin;
