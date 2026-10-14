/**
 * Dual-Worker Pipeline Controller (sceneWorkers.ts)
 * Uses a module-level Singleton Worker Pool (sharedMainWorker + sharedOrthoWorker)
 * to guarantee strictly 2 Web Workers total in the browser tab across React 19
 * StrictMode mounts/remounts:
 *   1. MainSceneWorker (Worker 1): Main Three.js scene point cloud generation,
 *      O(1) spatial grid indexing, dynamic color filtering (Ground Stripping &
 *      Object Isolation), and RANSAC/GPF + PCA road snapping.
 *   2. OrthoViewsWorker (Worker 2): Dedicated OffscreenCanvas Three.js WebGL
 *      Rendering Worker for the Top (BEV), Side, and Front orthographic views.
 */

import { snapAndOrientCuboid, type CuboidParams, type SnapResult } from './snappingAndOrientation';
import type { GpuProfile } from './gpuDetector';
import {
  buildSyntheticLiDARPointCloud,
  computeFilteredColors,
  type MainWorkerRequest,
  type MainWorkerResponse,
} from '../workers/mainSceneWorker';
import {
  computeOrthoRoiSubset,
  type OrthoFrustumBounds,
  type OrthoZoomLevels,
  type OrthoWorkerRequest,
  type OrthoWorkerResponse,
} from '../workers/orthoViewsWorker';

export type { OrthoFrustumBounds, OrthoZoomLevels };

export interface DualWorkerCallbacks {
  onMainSceneInitialized: (positions: Float32Array, colors: Float32Array, pointCount: number) => void;
  onMainPointCloudFiltered: (colors: Float32Array) => void;
  onMainSnapResult: (result: SnapResult) => void;
  onOrthoRoiMetrics: (bounds: OrthoFrustumBounds) => void;
  onOrthoViewsFallbackUpdated?: (
    roiPositions: Float32Array,
    roiColors: Float32Array,
    bounds: OrthoFrustumBounds
  ) => void;
}

/**
 * Module-level Singleton Worker Pool — guarantees strictly 2 workers total
 * (1 MainSceneWorker + 1 OrthoViewsWorker) in the entire application lifecycle.
 */
let sharedMainWorker: Worker | null = null;
let sharedOrthoWorker: Worker | null = null;

function getSharedWorkerPair(): { mainWorker: Worker | null; orthoWorker: Worker | null } {
  if (typeof Worker === 'undefined') {
    return { mainWorker: null, orthoWorker: null };
  }

  try {
    if (!sharedMainWorker) {
      sharedMainWorker = new Worker(
        new URL('../workers/mainSceneWorker.ts', import.meta.url),
        { type: 'module', name: 'MainSceneWorker' }
      );
    }
    if (!sharedOrthoWorker) {
      sharedOrthoWorker = new Worker(
        new URL('../workers/orthoViewsWorker.ts', import.meta.url),
        { type: 'module', name: 'OrthoViewsWorker' }
      );
    }
  } catch {
    sharedMainWorker = null;
    sharedOrthoWorker = null;
  }

  return {
    mainWorker: sharedMainWorker,
    orthoWorker: sharedOrthoWorker,
  };
}

export class DualSceneWorkerController {
  private mainWorker: Worker | null = null;
  private orthoWorker: Worker | null = null;
  private isOffscreenActive = false;
  private isDisposed = false;
  private filterRequestId = 0;
  private snapRequestId = 0;
  private orthoRequestId = 0;
  private callbacks: DualWorkerCallbacks;

  // Fallback caches if Worker constructor is unavailable
  private fallbackPositions: Float32Array | null = null;
  private fallbackColors: Float32Array | null = null;

  constructor(callbacks: DualWorkerCallbacks) {
    this.callbacks = callbacks;

    const { mainWorker, orthoWorker } = getSharedWorkerPair();
    this.mainWorker = mainWorker;
    this.orthoWorker = orthoWorker;

    if (this.mainWorker && this.orthoWorker) {
      this.mainWorker.onmessage = (event: MessageEvent<MainWorkerResponse>) => {
        if (this.isDisposed) return;
        const msg = event.data;
        if (msg.type === 'SCENE_INITIALIZED') {
          // Transfer a point cloud copy to Worker 2 (OrthoViewsWorker)
          const orthoPosCopy = new Float32Array(msg.positions);
          const orthoColCopy = new Float32Array(msg.colors);
          this.postToOrthoWorker(
            {
              type: 'SET_POINT_CLOUD',
              positions: orthoPosCopy,
              colors: orthoColCopy,
            },
            [orthoPosCopy.buffer, orthoColCopy.buffer]
          );

          this.callbacks.onMainSceneInitialized(msg.positions, msg.colors, msg.pointCount);
        } else if (msg.type === 'POINT_CLOUD_FILTERED') {
          if (msg.requestId === this.filterRequestId) {
            this.callbacks.onMainPointCloudFiltered(msg.colors);
          }
        } else if (msg.type === 'SNAP_RESULT') {
          if (msg.requestId === this.snapRequestId) {
            this.callbacks.onMainSnapResult(msg.result);
          }
        }
      };

      this.orthoWorker.onmessage = (event: MessageEvent<OrthoWorkerResponse>) => {
        if (this.isDisposed) return;
        const msg = event.data;
        if (msg.type === 'ROI_METRICS') {
          if (msg.requestId === 0 || msg.requestId === this.orthoRequestId) {
            this.callbacks.onOrthoRoiMetrics(msg.bounds);
          }
        } else if (msg.type === 'ORTHO_VIEWS_UPDATED') {
          if (msg.requestId === this.orthoRequestId) {
            this.callbacks.onOrthoRoiMetrics(msg.bounds);
            this.callbacks.onOrthoViewsFallbackUpdated?.(
              msg.roiPositions,
              msg.roiColors,
              msg.bounds
            );
          }
        }
      };
    }
  }

  private postToMainWorker(msg: MainWorkerRequest, transfer?: Transferable[]): void {
    if (!this.mainWorker || this.isDisposed) return;
    if (transfer && transfer.length > 0) {
      this.mainWorker.postMessage(msg, transfer);
    } else {
      this.mainWorker.postMessage(msg);
    }
  }

  private postToOrthoWorker(msg: OrthoWorkerRequest, transfer?: Transferable[]): void {
    if (!this.orthoWorker || this.isDisposed) return;
    if (transfer && transfer.length > 0) {
      this.orthoWorker.postMessage(msg, transfer);
    } else {
      this.orthoWorker.postMessage(msg);
    }
  }

  /**
   * Transfers an OffscreenCanvas to Worker 2 so all Top, Side, and Front Three.js
   * WebGL rendering executes 100% off the main thread, configured with the detected GpuProfile.
   */
  public initOffscreenCanvas(
    canvas: OffscreenCanvas,
    width: number,
    height: number,
    pixelRatio: number,
    cuboid: CuboidParams,
    activeColor: string,
    isGroundStripped: boolean,
    gpuProfile?: GpuProfile
  ): boolean {
    if (!this.orthoWorker || this.isDisposed) return false;
    this.isOffscreenActive = true;
    this.postToOrthoWorker(
      {
        type: 'INIT_OFFSCREEN_CANVAS',
        canvas,
        width,
        height,
        pixelRatio: gpuProfile ? gpuProfile.pixelRatio : pixelRatio,
        antialias: gpuProfile ? gpuProfile.antialias : true,
        orthoRoiMultiplier: gpuProfile ? gpuProfile.orthoRoiMultiplier : 1.6,
        cuboid,
        activeColor,
        isGroundStripped,
      },
      [canvas]
    );
    return true;
  }

  public isOffscreenRendering(): boolean {
    return this.isOffscreenActive;
  }

  public initScene(): void {
    if (this.mainWorker) {
      this.postToMainWorker({ type: 'INIT_SCENE' });
    } else {
      const generated = buildSyntheticLiDARPointCloud();
      this.fallbackPositions = new Float32Array(generated.positions);
      this.fallbackColors = new Float32Array(generated.colors);
      this.callbacks.onMainSceneInitialized(
        generated.positions,
        generated.colors,
        generated.positions.length / 3
      );
    }
  }

  public requestFilterPointCloud(
    cuboid: CuboidParams,
    isGroundStripped: boolean,
    isObjectIsolated: boolean
  ): void {
    this.filterRequestId += 1;
    const requestId = this.filterRequestId;

    if (this.mainWorker) {
      this.postToMainWorker({
        type: 'FILTER_POINT_CLOUD',
        requestId,
        cuboid,
        isGroundStripped,
        isObjectIsolated,
      });
    } else if (this.fallbackPositions && this.fallbackColors) {
      const colors = computeFilteredColors(
        this.fallbackPositions,
        this.fallbackColors,
        cuboid,
        isGroundStripped,
        isObjectIsolated
      );
      this.callbacks.onMainPointCloudFiltered(colors);
    }
  }

  public requestSnapAndOrient(
    cuboid: CuboidParams,
    options?: { snapGround?: boolean; autoOrient?: boolean }
  ): void {
    this.snapRequestId += 1;
    const requestId = this.snapRequestId;

    if (this.mainWorker) {
      this.postToMainWorker({
        type: 'SNAP_AND_ORIENT',
        requestId,
        cuboid,
        options,
      });
    } else if (this.fallbackPositions) {
      const result = snapAndOrientCuboid(cuboid, this.fallbackPositions, options);
      this.callbacks.onMainSnapResult(result);
    }
  }

  public requestOrthoCuboidUpdate(
    cuboid: CuboidParams,
    activeColor: string,
    isGroundStripped: boolean
  ): void {
    this.orthoRequestId += 1;
    const requestId = this.orthoRequestId;

    if (this.orthoWorker && this.isOffscreenActive) {
      this.postToOrthoWorker({
        type: 'UPDATE_CUBOID',
        requestId,
        cuboid,
        activeColor,
        isGroundStripped,
      });
    } else if (this.orthoWorker) {
      this.postToOrthoWorker({
        type: 'UPDATE_ORTHO_VIEWS',
        requestId,
        cuboid,
        isGroundStripped,
      });
    } else if (this.fallbackPositions && this.fallbackColors) {
      const { roiPositions, roiColors, bounds } = computeOrthoRoiSubset(
        this.fallbackPositions,
        this.fallbackColors,
        cuboid,
        isGroundStripped
      );
      this.callbacks.onOrthoRoiMetrics(bounds);
      this.callbacks.onOrthoViewsFallbackUpdated?.(roiPositions, roiColors, bounds);
    }
  }

  public requestOrthoZoomUpdate(zoomLevels: OrthoZoomLevels): void {
    if (this.orthoWorker && this.isOffscreenActive) {
      this.postToOrthoWorker({
        type: 'UPDATE_ZOOM',
        zoomLevels,
      });
    }
  }

  public requestOrthoResize(width: number, height: number, pixelRatio: number): void {
    if (this.orthoWorker && this.isOffscreenActive) {
      this.postToOrthoWorker({
        type: 'RESIZE',
        width,
        height,
        pixelRatio,
      });
    }
  }

  /**
   * Detaches controller callbacks without spawning additional workers on remount,
   * preserving the strict 2-worker singleton pool.
   */
  public dispose(): void {
    this.isDisposed = true;
    this.isOffscreenActive = false;
  }
}
