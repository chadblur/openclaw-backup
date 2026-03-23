// Onboarding adapter for ADP OpenClaw channel plugin
import type {
  ChannelOnboardingAdapter,
  OpenClawConfig,
  WizardPrompter,
} from "openclaw/plugin-sdk";
import { DEFAULT_ACCOUNT_ID } from "openclaw/plugin-sdk";

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

async function noteAdpOpenclawSetup(prompter: WizardPrompter): Promise<void> {
  await prompter.note(
    [
      "ADP OpenClaw connects to a WebSocket server for real-time messaging.",
      "You need a clientToken to authenticate with the server.",
      "The signKey is used for HMAC signature generation (default: ADPOpenClaw).",
    ].join("\n"),
    "ADP OpenClaw setup",
  );
}

export const adpOpenclawOnboardingAdapter: ChannelOnboardingAdapter = {
  channel,
  getStatus: async ({ cfg }) => {
    const channelCfg = getChannelConfig(cfg);
    const configured = isConfigured(channelCfg);

    return {
      channel,
      configured,
      statusLines: [`ADP OpenClaw: ${configured ? "configured" : "needs clientToken"}`],
      selectionHint: configured ? "configured" : "requires clientToken",
      quickstartScore: configured ? 1 : 10,
    };
  },
  configure: async ({ cfg, prompter }) => {
    let next = cfg;
    const accountId = DEFAULT_ACCOUNT_ID;

    await noteAdpOpenclawSetup(prompter);

    const channelCfg = getChannelConfig(next);
    const existingClientToken = channelCfg?.clientToken?.trim();
    const existingSignKey = channelCfg?.signKey?.trim();

    // Check for env vars
    const envClientToken = process.env.ADP_OPENCLAW_CLIENT_TOKEN?.trim();
    const envSignKey = process.env.ADP_OPENCLAW_SIGN_KEY?.trim();

    if (envClientToken) {
      const useEnv = await prompter.confirm({
        message: "ADP_OPENCLAW_CLIENT_TOKEN detected in env. Use environment variables?",
        initialValue: true,
      });
      if (useEnv) {
        next = updateAdpOpenclawConfig(next, { enabled: true });
        return { cfg: next, accountId };
      }
    }

    // Prompt for clientToken (required)
    const clientTokenInput = await prompter.text({
      message: "Client Token",
      placeholder: "your-client-token",
      initialValue: existingClientToken || undefined,
      validate: (value) => (String(value ?? "").trim() ? undefined : "Required"),
    });
    const clientToken = String(clientTokenInput).trim();

    // Prompt for signKey (optional, has default)
    const signKeyInput = await prompter.text({
      message: "Sign Key (press Enter for default: ADPOpenClaw)",
      placeholder: "ADPOpenClaw",
      initialValue: existingSignKey || envSignKey || undefined,
    });
    const signKey = String(signKeyInput ?? "").trim() || undefined;

    next = updateAdpOpenclawConfig(next, {
      clientToken,
      ...(signKey ? { signKey } : {}),
      enabled: true,
    });

    return { cfg: next, accountId };
  },
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
