/**
 * Worker 1: Main Three.js Scene Worker (mainSceneWorker.ts)
 * Offloads heavy main-scene point cloud generation, O(1) 2D spatial grid indexing,
 * dynamic point cloud color filtering (Ground Stripping & Object Isolation), and
 * RANSAC/GPF + PCA road snapping from the main UI / camera thread.
 */

import { snapAndOrientCuboid, type CuboidParams, type SnapResult } from '../utils/snappingAndOrientation';

export interface MainWorkerInitMessage {
  type: 'INIT_SCENE';
}

export interface MainWorkerFilterMessage {
  type: 'FILTER_POINT_CLOUD';
  requestId: number;
  cuboid: CuboidParams;
  isGroundStripped: boolean;
  isObjectIsolated: boolean;
}

export interface MainWorkerSnapMessage {
  type: 'SNAP_AND_ORIENT';
  requestId: number;
  cuboid: CuboidParams;
  options?: {
    snapGround?: boolean;
    autoOrient?: boolean;
  };
}

export type MainWorkerRequest =
  | MainWorkerInitMessage
  | MainWorkerFilterMessage
  | MainWorkerSnapMessage;

export interface MainWorkerInitializedResponse {
  type: 'SCENE_INITIALIZED';
  positions: Float32Array;
  colors: Float32Array;
  pointCount: number;
}

export interface MainWorkerFilteredResponse {
  type: 'POINT_CLOUD_FILTERED';
  requestId: number;
  colors: Float32Array;
}

export interface MainWorkerSnapResponse {
  type: 'SNAP_RESULT';
  requestId: number;
  result: SnapResult;
}

export type MainWorkerResponse =
  | MainWorkerInitializedResponse
  | MainWorkerFilteredResponse
  | MainWorkerSnapResponse;

interface WorkerGlobalScopeCompat {
  postMessage: (message: MainWorkerResponse, transfer?: Transferable[]) => void;
  onmessage: ((event: MessageEvent<MainWorkerRequest>) => void) | null;
}

const workerScope = self as unknown as WorkerGlobalScopeCompat;

let cachedPositions: Float32Array | null = null;
let cachedOriginalColors: Float32Array | null = null;
let cachedGroundStrippedColors: Float32Array | null = null;

/**
 * O(1) 2D Spatial Hash Grid for instant neighborhood queries around any (x, y) radius.
 */
class SpatialGrid2D {
  private cellSize: number;
  private buckets: Map<number, Int32Array> = new Map();

  constructor(positions: Float32Array, cellSize = 2.0) {
    this.cellSize = cellSize;
    const tempBuckets = new Map<number, number[]>();
    const count = positions.length / 3;

    for (let i = 0; i < count; i++) {
      const px = positions[i * 3];
      const py = positions[i * 3 + 1];
      const key = this.getCellKey(
        Math.floor(px / cellSize),
        Math.floor(py / cellSize)
      );
      let bucket = tempBuckets.get(key);
      if (!bucket) {
        bucket = [];
        tempBuckets.set(key, bucket);
      }
      bucket.push(i);
    }

    for (const [key, list] of tempBuckets.entries()) {
      this.buckets.set(key, new Int32Array(list));
    }
  }

  private getCellKey(cx: number, cy: number): number {
    return ((cx + 4096) << 13) ^ (cy + 4096);
  }

  public queryIndicesInRadius(x: number, y: number, radius: number): number[] {
    const minCx = Math.floor((x - radius) / this.cellSize);
    const maxCx = Math.floor((x + radius) / this.cellSize);
    const minCy = Math.floor((y - radius) / this.cellSize);
    const maxCy = Math.floor((y + radius) / this.cellSize);

    const indices: number[] = [];
    for (let cx = minCx; cx <= maxCx; cx++) {
      for (let cy = minCy; cy <= maxCy; cy++) {
        const bucket = this.buckets.get(this.getCellKey(cx, cy));
        if (bucket) {
          for (let i = 0; i < bucket.length; i++) {
            indices.push(bucket[i]);
          }
        }
      }
    }
    return indices;
  }

  public queryPointsFloat32(
    positions: Float32Array,
    x: number,
    y: number,
    radius: number
  ): Float32Array {
    const indices = this.queryIndicesInRadius(x, y, radius);
    const rSq = radius * radius;
    const pts: number[] = [];

    for (let i = 0; i < indices.length; i++) {
      const idx3 = indices[i] * 3;
      const px = positions[idx3];
      const py = positions[idx3 + 1];
      const dx = px - x;
      const dy = py - y;
      if (dx * dx + dy * dy <= rSq) {
        pts.push(px, py, positions[idx3 + 2]);
      }
    }
    return new Float32Array(pts);
  }
}

let spatialGrid: SpatialGrid2D | null = null;

/**
 * Generates the synthetic 64-beam LiDAR point cloud with ground plane, lane markings,
 * and target vehicle cluster.
 */
export function buildSyntheticLiDARPointCloud(): {
  positions: Float32Array;
  colors: Float32Array;
} {
  const pts: number[] = [];
  const colors: number[] = [];

  // Ground Plane & Road Markings
  const groundZ = -1.65;
  for (let r = 2; r < 55; r += 0.4) {
    const numTheta = Math.floor(r * 24);
    for (let t = 0; t < numTheta; t++) {
      const angle = (t / numTheta) * Math.PI * 2;
      const x = r * Math.cos(angle);
      const y = r * Math.sin(angle);
      const z = groundZ + (Math.random() - 0.5) * 0.02;

      pts.push(x, y, z);

      // Lane markings intensity (y = +-1.75, +-5.25)
      const isLane = Math.abs(Math.abs(y) - 1.75) < 0.12 || Math.abs(Math.abs(y) - 5.25) < 0.12;
      if (isLane) {
        colors.push(0.95, 0.9, 0.3); // Bright yellow lane markings
      } else {
        const depthShade = Math.max(0.15, 0.55 - r / 100);
        colors.push(depthShade * 0.5, depthShade * 0.7, depthShade);
      }
    }
  }

  // Target vehicle point cluster around x=10.0, y=1.8 (heading ~14.5 degrees)
  const vehCenterX = 10.0;
  const vehCenterY = 1.8;
  const vehYaw = (14.5 * Math.PI) / 180;
  const cosY = Math.cos(vehYaw);
  const sinY = Math.sin(vehYaw);

  const vehL = 4.4;
  const vehW = 1.9;
  const vehH = 1.5;

  for (let i = 0; i < 950; i++) {
    const sign = Math.random() > 0.5 ? 1 : -1;
    let lx: number;
    let ly: number;
    let lz: number;

    const face = Math.floor(Math.random() * 3);
    if (face === 0) {
      // Side face
      ly = (sign * vehW) / 2 + (Math.random() - 0.5) * 0.05;
      lx = (Math.random() - 0.5) * vehL;
      lz = Math.random() * vehH;
    } else if (face === 1) {
      // Front/Rear face
      lx = (sign * vehL) / 2 + (Math.random() - 0.5) * 0.05;
      ly = (Math.random() - 0.5) * vehW;
      lz = Math.random() * vehH;
    } else {
      // Roof
      lz = vehH + (Math.random() - 0.5) * 0.04;
      lx = (Math.random() - 0.5) * vehL * 0.7;
      ly = (Math.random() - 0.5) * vehW * 0.85;
    }

    const wx = vehCenterX + (lx * cosY - ly * sinY);
    const wy = vehCenterY + (lx * sinY + ly * cosY);
    const wz = groundZ + lz;

    pts.push(wx, wy, wz);
    colors.push(0.3, 0.85, 1.0);
  }

  return {
    positions: new Float32Array(pts),
    colors: new Float32Array(colors),
  };
}

/**
 * Precomputes ground-stripped colors once so toggling Ground Stripping is an instant memcpy.
 */
function buildGroundStrippedColors(
  positions: Float32Array,
  origColors: Float32Array
): Float32Array {
  const out = new Float32Array(origColors.length);
  const count = positions.length / 3;
  for (let i = 0; i < count; i++) {
    const idx = i * 3;
    if (positions[idx + 2] <= -1.55) {
      out[idx] = 0.03;
      out[idx + 1] = 0.04;
      out[idx + 2] = 0.06;
    } else {
      out[idx] = origColors[idx];
      out[idx + 1] = origColors[idx + 1];
      out[idx + 2] = origColors[idx + 2];
    }
  }
  return out;
}

/**
 * Computes filtered point cloud vertex colors for Ground Stripping and Object Isolation.
 * Uses precomputed base colors + O(1) spatial grid neighborhood lookup when Object Isolation is active.
 */
export function computeFilteredColors(
  positions: Float32Array,
  origColors: Float32Array,
  cuboid: CuboidParams,
  isGroundStripped: boolean,
  isObjectIsolated: boolean,
  groundStrippedCache?: Float32Array | null,
  grid?: SpatialGrid2D | null
): Float32Array {
  const baseColors =
    isGroundStripped && groundStrippedCache ? groundStrippedCache : origColors;

  if (!isObjectIsolated) {
    if (isGroundStripped && !groundStrippedCache) {
      return buildGroundStrippedColors(positions, origColors);
    }
    return new Float32Array(baseColors);
  }

  const outColors = new Float32Array(baseColors.length);
  const len = baseColors.length;
  for (let k = 0; k < len; k++) {
    outColors[k] = baseColors[k] * 0.15;
  }

  const cx = cuboid.x;
  const cy = cuboid.y;
  const halfL = cuboid.l / 2 + 0.35;
  const halfW = cuboid.w / 2 + 0.35;
  const minZ = cuboid.z - cuboid.h / 2 - 0.1;
  const maxZ = cuboid.z + cuboid.h / 2 + 0.2;
  const cosY = Math.cos(cuboid.yaw);
  const sinY = Math.sin(cuboid.yaw);
  const searchRadius = Math.sqrt(halfL * halfL + halfW * halfW) + 0.5;

  if (grid) {
    // O(1) spatial grid lookup for candidate object points
    const candidateIndices = grid.queryIndicesInRadius(cx, cy, searchRadius);
    for (let j = 0; j < candidateIndices.length; j++) {
      const idx = candidateIndices[j] * 3;
      const px = positions[idx];
      const py = positions[idx + 1];
      const pz = positions[idx + 2];

      if (pz < minZ || pz > maxZ) continue;
      if (isGroundStripped && pz <= -1.55) continue;

      const dx = px - cx;
      const dy = py - cy;
      const u = dx * cosY + dy * sinY;
      const v = -dx * sinY + dy * cosY;

      if (Math.abs(u) <= halfL && Math.abs(v) <= halfW) {
        outColors[idx] = 0.2;
        outColors[idx + 1] = 0.95;
        outColors[idx + 2] = 1.0;
      }
    }
    return outColors;
  }

  // Fallback linear scan if grid is not provided
  const count = positions.length / 3;
  for (let i = 0; i < count; i++) {
    const idx = i * 3;
    const px = positions[idx];
    const py = positions[idx + 1];
    const pz = positions[idx + 2];

    if (pz < minZ || pz > maxZ) continue;
    if (isGroundStripped && pz <= -1.55) continue;

    const dx = px - cx;
    const dy = py - cy;
    const u = dx * cosY + dy * sinY;
    const v = -dx * sinY + dy * cosY;

    if (Math.abs(u) <= halfL && Math.abs(v) <= halfW) {
      outColors[idx] = 0.2;
      outColors[idx + 1] = 0.95;
      outColors[idx + 2] = 1.0;
    }
  }

  return outColors;
}

workerScope.onmessage = (event: MessageEvent<MainWorkerRequest>) => {
  const msg = event.data;

  if (msg.type === 'INIT_SCENE') {
    if (!cachedPositions || !cachedOriginalColors || !spatialGrid) {
      const generated = buildSyntheticLiDARPointCloud();
      cachedPositions = new Float32Array(generated.positions);
      cachedOriginalColors = new Float32Array(generated.colors);
      cachedGroundStrippedColors = buildGroundStrippedColors(
        cachedPositions,
        cachedOriginalColors
      );
      spatialGrid = new SpatialGrid2D(cachedPositions, 2.0);
    }

    const transferPositions = new Float32Array(cachedPositions);
    const transferColors = new Float32Array(cachedOriginalColors);

    workerScope.postMessage(
      {
        type: 'SCENE_INITIALIZED',
        positions: transferPositions,
        colors: transferColors,
        pointCount: transferPositions.length / 3,
      },
      [transferPositions.buffer, transferColors.buffer]
    );
  } else if (msg.type === 'FILTER_POINT_CLOUD') {
    if (!cachedPositions || !cachedOriginalColors) return;

    const filteredColors = computeFilteredColors(
      cachedPositions,
      cachedOriginalColors,
      msg.cuboid,
      msg.isGroundStripped,
      msg.isObjectIsolated,
      cachedGroundStrippedColors,
      spatialGrid
    );

    workerScope.postMessage(
      {
        type: 'POINT_CLOUD_FILTERED',
        requestId: msg.requestId,
        colors: filteredColors,
      },
      [filteredColors.buffer]
    );
  } else if (msg.type === 'SNAP_AND_ORIENT') {
    if (!cachedPositions) return;

    const c = msg.cuboid;
    const queryRadius =
      Math.max(3.0, Math.sqrt((c.l / 2) ** 2 + (c.w / 2) ** 2) + 2.5) + 0.5;
    const localCloud = spatialGrid
      ? spatialGrid.queryPointsFloat32(cachedPositions, c.x, c.y, queryRadius)
      : cachedPositions;

    const result = snapAndOrientCuboid(c, localCloud, msg.options);
    workerScope.postMessage({
      type: 'SNAP_RESULT',
      requestId: msg.requestId,
      result,
    });
  }
};
