import { z } from "zod";

export const AdpOpenclawConfigSchema = z.object({
  enabled: z.boolean().optional(),
  wsUrl: z.string().optional(), // WebSocket URL (optional, auto-detected from clientToken prefix: sk-adp- → domestic, sk-adp_intl- → international)
  clientToken: z.string().optional(),
  signKey: z.string().optional(),
});
