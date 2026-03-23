# media-team (3 agents)

This folder contains a 3-agent team for AI self-media work:

- **media-manager**: Orchestrator. Task decomposition, assignment, and final aggregation. Chat-only permissions.
- **media-wechat**: WeChat Official Account long-form, practical articles. Feishu doc editing permission.
- **media-xhs**: Xiaohongshu seeding/planting posts (种草文案). Feishu doc editing permission.

Each agent has its own minimal workspace files (AGENTS.md/SOUL.md/USER.md/etc.) under its subfolder.

## How to use

1. Talk to **media-manager** with a topic + goal + target audience.
2. media-manager will create tasks for wechat/xhs agents and collect drafts.
3. wechat/xhs agents output:
   - A draft
   - A “ready to paste” version
   - Suggested title/cover hooks
   - Feishu doc operations (create/update) when configured.
