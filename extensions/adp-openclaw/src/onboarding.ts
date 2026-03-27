// Setup wizard adapter for ADP OpenClaw channel plugin
// Migrated from ChannelOnboardingAdapter to ChannelSetupWizard for openclaw >= 2026.3.22
import type {
  ChannelSetupWizard,
  OpenClawConfig,
} from "openclaw/plugin-sdk";

const channel = "adp-openclaw" as const;

type AdpOpenclawChannelConfig = {
  enabled?: boolean;
  wsUrl?: string;
  clientToken?: string;
  signKey?: string;
};

function getChannelConfig(cfg: OpenClawConfig): AdpOpenclawChannelConfig | undefined {
  return cfg.channels?.["adp-openclaw"] as AdpOpenclawChannelConfig | undefined;
}

function isConfigured(channelCfg?: AdpOpenclawChannelConfig): boolean {
  // clientToken is required for configured status
  const clientToken = channelCfg?.clientToken?.trim() || process.env.ADP_OPENCLAW_CLIENT_TOKEN;
  return Boolean(clientToken);
}

function updateAdpOpenclawConfig(
  cfg: OpenClawConfig,
  updates: { wsUrl?: string; clientToken?: string; signKey?: string; enabled?: boolean },
): OpenClawConfig {
  return {
    ...cfg,
    channels: {
      ...cfg.channels,
      "adp-openclaw": {
        ...cfg.channels?.["adp-openclaw"],
        ...updates,
        enabled: updates.enabled ?? true,
      },
    },
  };
}

export const adpOpenclawSetupWizard: ChannelSetupWizard = {
  channel,

  status: {
    configuredLabel: "configured",
    unconfiguredLabel: "needs clientToken",
    configuredHint: "configured",
    unconfiguredHint: "requires clientToken",
    configuredScore: 1,
    unconfiguredScore: 10,
    resolveConfigured: ({ cfg }) => {
      const channelCfg = getChannelConfig(cfg);
      return isConfigured(channelCfg);
    },
    resolveStatusLines: ({ cfg, configured }) => {
      return [`ADP OpenClaw: ${configured ? "configured" : "needs clientToken"}`];
    },
  },

  introNote: {
    title: "ADP OpenClaw setup",
    lines: [
      "ADP OpenClaw connects to a WebSocket server for real-time messaging.",
      "You need a clientToken to authenticate with the server.",
      "The signKey is used for HMAC signature generation (default: ADPOpenClaw).",
    ],
  },

  envShortcut: {
    prompt: "ADP_OPENCLAW_CLIENT_TOKEN detected in env. Use environment variables?",
    preferredEnvVar: "ADP_OPENCLAW_CLIENT_TOKEN",
    isAvailable: () => Boolean(process.env.ADP_OPENCLAW_CLIENT_TOKEN?.trim()),
    apply: ({ cfg }) => updateAdpOpenclawConfig(cfg, { enabled: true }),
  },

  credentials: [
    {
      inputKey: "token",
      providerHint: "ADP OpenClaw",
      credentialLabel: "Client Token",
      preferredEnvVar: "ADP_OPENCLAW_CLIENT_TOKEN",
      envPrompt: "ADP_OPENCLAW_CLIENT_TOKEN detected. Use it?",
      keepPrompt: "Client Token already set. Keep it?",
      inputPrompt: "Client Token",
      inspect: ({ cfg }) => {
        const channelCfg = getChannelConfig(cfg);
        const configuredValue = channelCfg?.clientToken?.trim();
        const envValue = process.env.ADP_OPENCLAW_CLIENT_TOKEN?.trim();
        return {
          accountConfigured: Boolean(configuredValue || envValue),
          hasConfiguredValue: Boolean(configuredValue),
          resolvedValue: configuredValue || envValue,
          envValue,
        };
      },
      applySet: ({ cfg, resolvedValue }) => {
        return updateAdpOpenclawConfig(cfg, { clientToken: resolvedValue, enabled: true });
      },
      applyUseEnv: ({ cfg }) => {
        return updateAdpOpenclawConfig(cfg, { enabled: true });
      },
    },
  ],

  textInputs: [
    {
      inputKey: "name",
      message: "Sign Key (press Enter for default: ADPOpenClaw)",
      placeholder: "ADPOpenClaw",
      required: false,
      currentValue: ({ cfg }) => {
        const channelCfg = getChannelConfig(cfg);
        return channelCfg?.signKey?.trim() || process.env.ADP_OPENCLAW_SIGN_KEY?.trim();
      },
      applySet: ({ cfg, value }) => {
        return updateAdpOpenclawConfig(cfg, {
          ...(value ? { signKey: value } : {}),
          enabled: true,
        });
      },
    },
  ],

  disable: (cfg) => {
    return {
      ...cfg,
      channels: {
        ...cfg.channels,
        "adp-openclaw": {
          ...cfg.channels?.["adp-openclaw"],
          enabled: false,
        },
      },
    };
  },
};
