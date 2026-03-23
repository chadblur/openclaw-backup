import { z } from "zod";

export const AdpOpenclawConfigSchema = z.object({
  enabled: z.boolean().optional(),
  wsUrl: z.string().optional(), // WebSocket URL (optional, default: wss://wss.lke.cloud.tencent.com/bot/gateway/conn)
  clientToken: z.string().optional(),
  signKey: z.string().optional(),
});
