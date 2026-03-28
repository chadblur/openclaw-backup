"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = {
    entry: { index: 'index.ts' },
    format: 'esm',
    target: 'node22',
    platform: 'node',
    clean: false,
    outDir: '.',
    dts: true,
    deps: {
        neverBundle: [
            /^openclaw(\/.*)?$/,
            /^@larksuiteoapi\//,
            /^@sinclair\//,
            'image-size', 'zod', /^node:/,
        ],
    },
};
