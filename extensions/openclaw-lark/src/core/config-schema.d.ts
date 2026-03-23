/**
 * Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
 * SPDX-License-Identifier: MIT
 *
 * Zod-based configuration schema for the OpenClaw Lark/Feishu channel plugin.
 *
 * Provides runtime validation, sensible defaults, and cross-field refinements
 * so that every consuming module can rely on well-typed configuration objects.
 */
import { z } from 'zod';
export { z };
export declare const UATConfigSchema: any;
export declare const FeishuGroupSchema: any;
export declare const FeishuAccountConfigSchema: any;
export declare const FeishuConfigSchema: any;
/**
 * JSON Schema derived from FeishuConfigSchema.
 *
 * - `io: "input"` exposes the input type for `.transform()` schemas (e.g. AllowFromSchema).
 * - `unrepresentable: "any"` degrades `.superRefine()` constraints to `{}`.
 * - `target: "draft-07"` matches the plugin system's expected JSON Schema version.
 */
export declare const FEISHU_CONFIG_JSON_SCHEMA: Record<string, unknown>;
