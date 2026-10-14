/**
 * Dedicated GPU (dGPU) vs Integrated/Software GPU Detector (gpuDetector.ts)
 * Probes WebGL UNMASKED_RENDERER_WEBGL / UNMASKED_VENDOR_WEBGL once (releasing the
 * temporary probe context immediately via WEBGL_lose_context) and selects the optimal
 * rendering method and pipeline parameters for both the Main 3D WebGLRenderer and
 * the OffscreenCanvas OrthoViewsWorker WebGLRenderer.
 */

export type GpuTier = 'dgpu' | 'igpu' | 'software';

export interface GpuProfile {
  tier: GpuTier;
  vendor: string;
  renderer: string;
  shortGpuLabel: string;
  renderingMethod: string;
  antialias: boolean;
  pixelRatio: number;
  dampingFactor: number;
  orthoRoiMultiplier: number;
  pointSize: number;
  supportsOffscreenCanvas: boolean;
}

let cachedGpuProfile: GpuProfile | null = null;

/**
 * Classifies an unmasked WebGL renderer/vendor string into 'dgpu', 'igpu', or 'software'.
 */
export function classifyGpuTier(rendererRaw: string, vendorRaw: string): GpuTier {
  const combined = `${rendererRaw} ${vendorRaw}`.toLowerCase();

  // 1. Check for Software / CPU rasterizers first
  if (
    /swiftshader|llvmpipe|softpipe|basic render driver|mesa offscreen|software|virgl/i.test(
      combined
    )
  ) {
    return 'software';
  }

  // 2. Check for Dedicated / Discrete GPUs (dGPU)
  if (
    /nvidia|geforce|rtx|gtx|quadro|tesla|a100|h100|l4|titan/i.test(combined) ||
    /radeon\s+rx|radeon\s+pro|firepro|radeon\s+vii|radeon\s+hd\s+[678]\d{3}/i.test(combined) ||
    /intel.*arc/i.test(combined) ||
    /apple\s+m\d+\s+(pro|max|ultra)/i.test(combined)
  ) {
    return 'dgpu';
  }

  // 3. Default to Integrated GPU (iGPU: Intel UHD/HD/Iris, AMD Radeon Graphics APU, base Apple M1/M2/M3, Adreno/Mali)
  return 'igpu';
}

/**
 * Extracts a concise, human-readable GPU name from the raw WebGL ANGLE/Mesa renderer string.
 */
function formatShortGpuLabel(rendererRaw: string, tier: GpuTier): string {
  if (!rendererRaw || rendererRaw === 'Unknown Renderer') {
    return tier === 'dgpu' ? 'Dedicated GPU' : tier === 'igpu' ? 'Integrated GPU' : 'Software WebGL';
  }

  // Extract inner GPU model from ANGLE(..., Model, ...) if present
  const angleMatch = rendererRaw.match(/ANGLE\s*\([^,]+,\s*([^,()]+)/i);
  const cleaned = (angleMatch ? angleMatch[1] : rendererRaw)
    .replace(/Direct3D11.*$/i, '')
    .replace(/OpenGL.*$/i, '')
    .replace(/Mesa\s+/i, '')
    .replace(/\(TM\)|\(R\)/gi, '')
    .trim();

  if (cleaned.length > 32) {
    return `${cleaned.slice(0, 30).trim()}…`;
  }
  return cleaned || rendererRaw;
}

/**
 * Probes the browser's WebGL context once to detect dGPU vs iGPU vs Software rendering,
 * caches the result as a singleton, and returns the optimal rendering configuration.
 */
export function detectGpuCapabilities(): GpuProfile {
  if (cachedGpuProfile) {
    return cachedGpuProfile;
  }

  let vendor = 'Unknown Vendor';
  let renderer = 'Unknown Renderer';
  const supportsOffscreenCanvas =
    typeof HTMLCanvasElement !== 'undefined' &&
    typeof HTMLCanvasElement.prototype.transferControlToOffscreen === 'function';

  if (typeof document !== 'undefined') {
    try {
      const probeCanvas = document.createElement('canvas');
      probeCanvas.width = 1;
      probeCanvas.height = 1;

      const gl =
        (probeCanvas.getContext('webgl2', { powerPreference: 'high-performance' }) as
          | WebGL2RenderingContext
          | null) ||
        (probeCanvas.getContext('webgl', { powerPreference: 'high-performance' }) as
          | WebGLRenderingContext
          | null);

      if (gl) {
        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
        if (debugInfo) {
          vendor =
            (gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) as string) || vendor;
          renderer =
            (gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) as string) || renderer;
        } else {
          vendor = (gl.getParameter(gl.VENDOR) as string) || vendor;
          renderer = (gl.getParameter(gl.RENDERER) as string) || renderer;
        }

        // Immediately release the probe WebGL context so it never consumes a context slot
        const loseCtx = gl.getExtension('WEBGL_lose_context');
        loseCtx?.loseContext();
      }
    } catch {
      // Ignore probe errors and fall back to conservative iGPU profile
    }
  }

  const tier = classifyGpuTier(renderer, vendor);
  const shortGpuLabel = formatShortGpuLabel(renderer, tier);
  const deviceDpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;

  if (tier === 'dgpu') {
    cachedGpuProfile = {
      tier: 'dgpu',
      vendor,
      renderer,
      shortGpuLabel,
      renderingMethod: 'dGPU Hardware Dual-Context + OffscreenCanvas Worker',
      antialias: true,
      pixelRatio: Math.min(deviceDpr, 2),
      dampingFactor: 0.28,
      orthoRoiMultiplier: 2.2,
      pointSize: 0.12,
      supportsOffscreenCanvas,
    };
  } else {
    cachedGpuProfile = {
      tier,
      vendor,
      renderer,
      shortGpuLabel,
      renderingMethod: 'iGPU/SW Low-Overhead Culling + Offscreen Worker',
      antialias: false,
      pixelRatio: 1,
      dampingFactor: 0.35,
      orthoRoiMultiplier: 1.3,
      pointSize: 0.10,
      supportsOffscreenCanvas,
    };
  }

  return cachedGpuProfile;
}
