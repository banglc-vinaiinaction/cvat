// Copyright (C) 2026 CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

export type GpuTier = 'dgpu' | 'igpu' | 'software';

export type RenderingMethod =
    | 'dgpu-hardware-offscreen'
    | 'igpu-balanced'
    | 'software-low-overhead';

export interface GpuCapabilities {
    tier: GpuTier;
    isDedicatedGpu: boolean;
    vendor: string;
    renderer: string;
    isWebGL2: boolean;
    maxTextureSize: number;
    supportsOffscreenCanvas: boolean;
    renderingMethod: RenderingMethod;
    antialias: boolean;
    pixelRatio: number;
    orthoRoiRadiusPadding: number;
    orthoMaxPoints: number;
}

const SOFTWARE_GPU_PATTERNS = [
    /swiftshader/i,
    /llvmpipe/i,
    /softpipe/i,
    /microsoft basic render/i,
    /svga3d/i,
    /mesa offscreen/i,
    /gdi generic/i,
];

const INTEGRATED_GPU_PATTERNS = [
    /intel\(r\)\s*(uhd|hd|iris)/i,
    /intel\s*(uhd|hd|iris)/i,
    /amd radeon\(tm\)\s*graphics/i,
    /radeon\s*vega/i,
    /adreno/i,
    /mali/i,
    /powervr/i,
    /apple m[1-4](\s|$)/i,
];

const DEDICATED_GPU_PATTERNS = [
    /nvidia/i,
    /geforce/i,
    /\brtx\b/i,
    /\bgtx\b/i,
    /quadro/i,
    /tesla/i,
    /titan/i,
    /\bl4\b/i,
    /\ba100\b/i,
    /\bh100\b/i,
    /radeon\s*rx/i,
    /radeon\s*pro/i,
    /firepro/i,
    /intel\(r\)\s*arc/i,
    /intel\s*arc/i,
    /apple m[1-4]\s*(pro|max|ultra)/i,
];

let cachedCapabilities: GpuCapabilities | null = null;

/**
 * Probes WebGL hardware capabilities once per session, classifies the GPU tier
 * ('dgpu' | 'igpu' | 'software'), and selects the optimal rendering method and
 * point-cloud budget for both the main perspective renderer and orthographic views.
 */
export function detectGpuCapabilities(): GpuCapabilities {
    if (cachedCapabilities) {
        return cachedCapabilities;
    }

    let vendor = 'Unknown';
    let renderer = 'Unknown';
    let isWebGL2 = false;
    let maxTextureSize = 4096;
    const supportsOffscreenCanvas = typeof OffscreenCanvas !== 'undefined';

    if (typeof window !== 'undefined' && typeof document !== 'undefined') {
        try {
            const probeCanvas = document.createElement('canvas');
            probeCanvas.width = 1;
            probeCanvas.height = 1;

            const gl = (
                probeCanvas.getContext('webgl2', { powerPreference: 'high-performance' }) ||
                probeCanvas.getContext('webgl', { powerPreference: 'high-performance' })
            ) as WebGLRenderingContext | WebGL2RenderingContext | null;

            if (gl) {
                isWebGL2 = typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext;
                const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
                if (debugInfo) {
                    vendor = String(gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) || '');
                    renderer = String(gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) || '');
                }
                if (!vendor) {
                    vendor = String(gl.getParameter(gl.VENDOR) || 'Unknown');
                }
                if (!renderer) {
                    renderer = String(gl.getParameter(gl.RENDERER) || 'Unknown');
                }
                maxTextureSize = Number(gl.getParameter(gl.MAX_TEXTURE_SIZE)) || 4096;

                // Immediately release temporary probe context so zero WebGL contexts are wasted
                const loseContextExt = gl.getExtension('WEBGL_lose_context');
                if (loseContextExt) {
                    loseContextExt.loseContext();
                }
            }
        } catch {
            // Ignore probe errors in restricted environments
        }
    }

    const combinedInfo = `${vendor} ${renderer}`;
    let tier: GpuTier = 'igpu';

    if (SOFTWARE_GPU_PATTERNS.some((pattern) => pattern.test(combinedInfo))) {
        tier = 'software';
    } else if (
        DEDICATED_GPU_PATTERNS.some((pattern) => pattern.test(combinedInfo)) &&
        !INTEGRATED_GPU_PATTERNS.some((pattern) => pattern.test(combinedInfo))
    ) {
        tier = 'dgpu';
    } else if (INTEGRATED_GPU_PATTERNS.some((pattern) => pattern.test(combinedInfo))) {
        tier = 'igpu';
    } else if (isWebGL2 && maxTextureSize >= 16384) {
        tier = 'dgpu';
    }

    const isDedicatedGpu = tier === 'dgpu';
    const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) ? window.devicePixelRatio : 1;

    if (isDedicatedGpu) {
        cachedCapabilities = {
            tier,
            isDedicatedGpu: true,
            vendor,
            renderer,
            isWebGL2,
            maxTextureSize,
            supportsOffscreenCanvas,
            renderingMethod: 'dgpu-hardware-offscreen',
            antialias: true,
            pixelRatio: Math.min(dpr, 2),
            orthoRoiRadiusPadding: 12.0,
            orthoMaxPoints: 25000,
        };
    } else if (tier === 'software') {
        cachedCapabilities = {
            tier,
            isDedicatedGpu: false,
            vendor,
            renderer,
            isWebGL2,
            maxTextureSize,
            supportsOffscreenCanvas,
            renderingMethod: 'software-low-overhead',
            antialias: false,
            pixelRatio: 1,
            orthoRoiRadiusPadding: 3.5,
            orthoMaxPoints: 2500,
        };
    } else {
        cachedCapabilities = {
            tier,
            isDedicatedGpu: false,
            vendor,
            renderer,
            isWebGL2,
            maxTextureSize,
            supportsOffscreenCanvas,
            renderingMethod: 'igpu-balanced',
            antialias: false,
            pixelRatio: 1,
            orthoRoiRadiusPadding: 4.5,
            orthoMaxPoints: 4000,
        };
    }

    return cachedCapabilities;
}
