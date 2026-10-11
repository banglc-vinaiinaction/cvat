/**
 * 3D LiDAR Road Plane Snapping & Orientation Engine (TypeScript/WebGL)
 * Handles client-side RANSAC ground plane fitting and PCA/Minimum Area Bounding Box orientation.
 */

export interface CuboidParams {
  x: number;
  y: number;
  z: number;
  l: number;
  w: number;
  h: number;
  yaw: number;
}

export interface SnapMetrics {
  deltaZ: number;
  deltaYawDeg: number;
  groundZ: number;
  pointsCount: number;
  status: 'snapped' | 'sparse_points' | 'ground_only';
}

export interface SnapResult {
  cuboid: CuboidParams;
  previousCuboid: CuboidParams;
  cuboidArray: number[];
  previousCuboidArray: number[];
  metrics: SnapMetrics;
}

/**
 * Computes the eigenvector corresponding to the smallest eigenvalue of a 3x3 symmetric matrix
 * using the Jacobi eigenvalue algorithm.
 */
export function computeSmallestEigenvector3x3(
  c00: number, c01: number, c02: number,
  c11: number, c12: number,
  c22: number
): [number, number, number] {
  const V = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  const A = [
    [c00, c01, c02],
    [c01, c11, c12],
    [c02, c12, c22],
  ];

  for (let iter = 0; iter < 15; iter++) {
    let p = 0;
    let q = 1;
    let maxOff = Math.abs(A[0][1]);
    if (Math.abs(A[0][2]) > maxOff) { p = 0; q = 2; maxOff = Math.abs(A[0][2]); }
    if (Math.abs(A[1][2]) > maxOff) { p = 1; q = 2; maxOff = Math.abs(A[1][2]); }

    if (maxOff < 1e-9) break;

    const app = A[p][p];
    const aqq = A[q][q];
    const apq = A[p][q];

    const theta = (aqq - app) / (2 * apq);
    const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
    const c = 1 / Math.sqrt(t * t + 1);
    const s = t * c;

    A[p][p] = app - t * apq;
    A[q][q] = aqq + t * apq;
    A[p][q] = 0;
    A[q][p] = 0;

    for (let r = 0; r < 3; r++) {
      if (r !== p && r !== q) {
        const arp = A[r][p];
        const arq = A[r][q];
        A[r][p] = c * arp - s * arq;
        A[p][r] = A[r][p];
        A[r][q] = s * arp + c * arq;
        A[q][r] = A[r][q];
      }
    }

    for (let r = 0; r < 3; r++) {
      const vrp = V[r][p];
      const vrq = V[r][q];
      V[r][p] = c * vrp - s * vrq;
      V[r][q] = s * vrp + c * vrq;
    }
  }

  let minIdx = 0;
  let minVal = A[0][0];
  if (A[1][1] < minVal) { minVal = A[1][1]; minIdx = 1; }
  if (A[2][2] < minVal) { minVal = A[2][2]; minIdx = 2; }

  return [V[0][minIdx], V[1][minIdx], V[2][minIdx]];
}

/**
 * Fits a road plane Ax + By + Cz + D = 0 using Ground Plane Fitting (GPF):
 * 1. Computes Lowest Point Representative (LPR) of local neighborhood.
 * 2. Extracts candidate ground seeds within LPR threshold.
 * 3. Uses PCA / Jacobi eigensolver to compute the normal.
 * 4. Refines inliers and fits the final optimal plane deterministically.
 */
export function fitGroundPlaneGPF(
  points: Float32Array | number[][],
  options: {
    lprRatio?: number;
    seedThreshold?: number;
    distanceThreshold?: number;
    refineIterations?: number;
  } = {}
): { plane: [number, number, number, number]; inliersCount: number } {
  const {
    lprRatio = 0.25,
    seedThreshold = 0.15,
    distanceThreshold = 0.08,
    refineIterations = 1,
  } = options;

  const numPoints = Array.isArray(points) ? points.length : points.length / 3;
  if (numPoints < 3) {
    return { plane: [0, 0, 1, 1.65], inliersCount: 0 };
  }

  const getPt = (idx: number): [number, number, number] => {
    if (Array.isArray(points)) {
      return [points[idx][0], points[idx][1], points[idx][2]];
    }
    return [points[idx * 3], points[idx * 3 + 1], points[idx * 3 + 2]];
  };

  const indices = Array.from({ length: numPoints }, (_, i) => i);
  indices.sort((a, b) => getPt(a)[2] - getPt(b)[2]);

  const lprCount = Math.max(3, Math.floor(numPoints * lprRatio));
  let lprZSum = 0;
  for (let i = 0; i < lprCount; i++) {
    lprZSum += getPt(indices[i])[2];
  }
  const lprZ = lprZSum / lprCount;

  let currentInliers = indices.filter((idx) => getPt(idx)[2] <= lprZ + seedThreshold);
  if (currentInliers.length < 3) {
    currentInliers = indices.slice(0, Math.min(numPoints, 10));
  }

  let bestPlane: [number, number, number, number] = [0, 0, 1, -lprZ];
  let finalInliersCount = currentInliers.length;

  const totalPasses = 1 + Math.max(0, refineIterations);
  for (let pass = 0; pass < totalPasses; pass++) {
    const N = currentInliers.length;
    if (N < 3) break;

    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < N; i++) {
      const p = getPt(currentInliers[i]);
      cx += p[0]; cy += p[1]; cz += p[2];
    }
    cx /= N; cy /= N; cz /= N;

    let xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
    for (let i = 0; i < N; i++) {
      const p = getPt(currentInliers[i]);
      const dx = p[0] - cx, dy = p[1] - cy, dz = p[2] - cz;
      xx += dx * dx; xy += dx * dy; xz += dx * dz;
      yy += dy * dy; yz += dy * dz; zz += dz * dz;
    }

    let [normX, normY, normZ] = computeSmallestEigenvector3x3(
      xx / N, xy / N, xz / N,
      yy / N, yz / N,
      zz / N
    );

    const len = Math.sqrt(normX * normX + normY * normY + normZ * normZ);
    if (len < 1e-6) {
      normX = 0; normY = 0; normZ = 1;
    } else {
      normX /= len; normY /= len; normZ /= len;
    }

    if (normZ < 0) {
      normX = -normX; normY = -normY; normZ = -normZ;
    }

    if (normZ < 0.85) {
      normX = 0; normY = 0; normZ = 1;
    }

    const d = -(normX * cx + normY * cy + normZ * cz);
    bestPlane = [normX, normY, normZ, d];

    if (pass < totalPasses - 1) {
      const nextInliers: number[] = [];
      for (let i = 0; i < numPoints; i++) {
        const p = getPt(i);
        const dist = Math.abs(normX * p[0] + normY * p[1] + normZ * p[2] + d);
        if (dist <= distanceThreshold) {
          nextInliers.push(i);
        }
      }
      if (nextInliers.length >= 3) {
        currentInliers = nextInliers;
        finalInliersCount = nextInliers.length;
      } else {
        break;
      }
    }
  }

  return { plane: bestPlane, inliersCount: finalInliersCount };
}

/**
 * Fits a plane Ax + By + Cz + D = 0 to 3D points using RANSAC.
 */
export function fitGroundPlaneRANSAC(
  points: Float32Array | number[][],
  distanceThreshold: number = 0.08,
  maxIterations: number = 150
): { plane: [number, number, number, number]; inliersCount: number } {
  const numPoints = Array.isArray(points) ? points.length : points.length / 3;
  if (numPoints < 3) {
    return { plane: [0, 0, 1, 1.65], inliersCount: 0 };
  }

  const getPt = (idx: number): [number, number, number] => {
    if (Array.isArray(points)) {
      return [points[idx][0], points[idx][1], points[idx][2]];
    }
    return [points[idx * 3], points[idx * 3 + 1], points[idx * 3 + 2]];
  };

  let bestPlane: [number, number, number, number] = [0, 0, 1, 1.65];
  let maxInliers = -1;

  for (let iter = 0; iter < maxIterations; iter++) {
    const i1 = Math.floor(Math.random() * numPoints);
    const i2 = Math.floor(Math.random() * numPoints);
    const i3 = Math.floor(Math.random() * numPoints);
    if (i1 === i2 || i2 === i3 || i1 === i3) continue;

    const p1 = getPt(i1);
    const p2 = getPt(i2);
    const p3 = getPt(i3);

    // Normal = (p2 - p1) x (p3 - p1)
    const v1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
    const v2 = [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]];

    const nx = v1[1] * v2[2] - v1[2] * v2[1];
    const ny = v1[2] * v2[0] - v1[0] * v2[2];
    const nz = v1[0] * v2[1] - v1[1] * v2[0];

    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (len < 1e-6) continue;

    let normX = nx / len;
    let normY = ny / len;
    let normZ = nz / len;

    // Ensure upright normal
    if (normZ < 0) {
      normX = -normX;
      normY = -normY;
      normZ = -normZ;
    }

    // Road normal should be roughly vertical (tilt < 25 deg)
    if (normZ < 0.90) continue;

    const d = -(normX * p1[0] + normY * p1[1] + normZ * p1[2]);

    // Count inliers
    let inliers = 0;
    for (let j = 0; j < numPoints; j++) {
      const p = getPt(j);
      const dist = Math.abs(normX * p[0] + normY * p[1] + normZ * p[2] + d);
      if (dist < distanceThreshold) inliers++;
    }

    if (inliers > maxInliers) {
      maxInliers = inliers;
      bestPlane = [normX, normY, normZ, d];
    }
  }

  return { plane: bestPlane, inliersCount: maxInliers };
}

/**
 * Calculates ground elevation Z at coordinates (x, y) for a given plane.
 */
export function getRoadHeightAt(plane: [number, number, number, number], x: number, y: number): number {
  const [A, B, C, D] = plane;
  if (Math.abs(C) < 1e-5) return -1.65;
  return -(A * x + B * y + D) / C;
}

/**
 * Fits principal heading and tight minimum bounding box around a 2D point cluster.
 */
export function fitBoundingBox2D(
  pts: [number, number][],
  currentYaw: number = 0
): { yaw: number; length: number; width: number; center: [number, number] } {
  if (pts.length < 3) {
    return { yaw: currentYaw, length: 4.5, width: 2.0, center: [0, 0] };
  }

  // Sweep angles around current yaw (+- 45 degrees in 1.5-degree steps)
  let bestArea = Infinity;
  let bestYaw = currentYaw;
  let bestDims = [4.5, 2.0];
  let bestCenter: [number, number] = [0, 0];

  const steps = 61;
  const range = Math.PI / 4;

  for (let i = 0; i <= steps; i++) {
    const theta = currentYaw - range + (i / steps) * (2 * range);
    const cosT = Math.cos(theta);
    const sinT = Math.sin(theta);

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    for (let pIdx = 0; pIdx < pts.length; pIdx++) {
      const px = pts[pIdx][0];
      const py = pts[pIdx][1];
      // Rotate by theta
      const rx = px * cosT + py * sinT;
      const ry = -px * sinT + py * cosT;

      if (rx < minX) minX = rx;
      if (rx > maxX) maxX = rx;
      if (ry < minY) minY = ry;
      if (ry > maxY) maxY = ry;
    }

    const lenX = maxX - minX;
    const lenY = maxY - minY;
    const area = lenX * lenY;

    if (area < bestArea) {
      bestArea = area;
      let finalYaw = theta;
      let length = lenX;
      let width = lenY;

      if (lenY > lenX) {
        // Longer axis is vehicle length
        length = lenY;
        width = lenX;
        finalYaw = theta + Math.PI / 2;
      }

      const midRotX = (minX + maxX) / 2;
      const midRotY = (minY + maxY) / 2;

      // Inverse rotate back to original frame
      const worldCenterX = midRotX * cosT - midRotY * sinT;
      const worldCenterY = midRotX * sinT + midRotY * cosT;

      bestDims = [length, width];
      bestCenter = [worldCenterX, worldCenterY];
      bestYaw = Math.atan2(Math.sin(finalYaw), Math.cos(finalYaw));
    }
  }

  return {
    yaw: bestYaw,
    length: Math.max(1.2, bestDims[0]),
    width: Math.max(0.8, bestDims[1]),
    center: bestCenter,
  };
}

/**
 * Aligns vehicle heading angle (yaw) with object LiDAR points along the longitudinal axis.
 * Strictly preserves vehicle dimensions (WLH) and limits adjustment to +/- 25 degrees.
 */
export function alignYawWithPoints(
  pts2D: [number, number][],
  currentYaw: number,
  l: number,
  w: number
): number {
  if (pts2D.length < 8) {
    return currentYaw;
  }

  // 2D PCA on object points
  let meanX = 0;
  let meanY = 0;
  const N = pts2D.length;
  for (let i = 0; i < N; i++) {
    meanX += pts2D[i][0];
    meanY += pts2D[i][1];
  }
  meanX /= N;
  meanY /= N;

  let cxx = 0;
  let cxy = 0;
  let cyy = 0;
  for (let i = 0; i < N; i++) {
    const x = pts2D[i][0] - meanX;
    const y = pts2D[i][1] - meanY;
    cxx += x * x;
    cxy += x * y;
    cyy += y * y;
  }

  // Principal orientation from 2x2 covariance
  const pcaAngle = 0.5 * Math.atan2(2 * cxy, cxx - cyy);

  // If length >= width, longitudinal axis is along local X.
  // If width > length, longitudinal axis is along local Y (offset by 90 deg).
  const isXLongitudinal = l >= w;
  const baseTargetAngle = isXLongitudinal ? pcaAngle : pcaAngle - Math.PI / 2;

  // The PCA line is bidirectional (baseTargetAngle or baseTargetAngle + Math.PI).
  // Pick the direction that is closest to currentYaw.
  const normalizeAngle = (a: number): number => Math.atan2(Math.sin(a), Math.cos(a));
  const diff1 = normalizeAngle(baseTargetAngle - currentYaw);
  const diff2 = normalizeAngle(baseTargetAngle + Math.PI - currentYaw);

  const bestDiff = Math.abs(diff1) <= Math.abs(diff2) ? diff1 : diff2;

  // Only align if within reasonable tolerance (at most 25 degrees)
  const maxAllowedAdjustment = (25 * Math.PI) / 180;
  if (Math.abs(bestDiff) <= maxAllowedAdjustment) {
    return normalizeAngle(currentYaw + bestDiff);
  }

  return currentYaw;
}

/**
 * Snaps cuboid bottom flush to detected road plane and aligns orientation.
 * Note: Width, Length, and Height (WLH: l, w, h) are strictly preserved.
 */
export function snapAndOrientCuboid(
  cuboid: CuboidParams,
  pointCloud: Float32Array | number[][],
  options: {
    snapGround?: boolean;
    autoOrient?: boolean;
  } = {}
): SnapResult {
  const { snapGround = true, autoOrient = true } = options;

  const previousCuboidArray = [
    cuboid.x, cuboid.y, cuboid.z,
    0, 0, cuboid.yaw,
    cuboid.l, cuboid.w, cuboid.h,
    0, 0, 0, 0, 0, 0, 0,
  ];

  // Preserve exact dimensions (WLH)
  const newL = cuboid.l;
  const newW = cuboid.w;
  const newH = cuboid.h;

  const bottomZ = cuboid.z - cuboid.h / 2;
  const numPoints = Array.isArray(pointCloud) ? pointCloud.length : pointCloud.length / 3;
  const getPt = (idx: number): [number, number, number] => {
    if (Array.isArray(pointCloud)) {
      return [pointCloud[idx][0], pointCloud[idx][1], pointCloud[idx][2]];
    }
    return [pointCloud[idx * 3], pointCloud[idx * 3 + 1], pointCloud[idx * 3 + 2]];
  };

  const radius = Math.max(3.0, Math.sqrt((cuboid.l / 2) ** 2 + (cuboid.w / 2) ** 2) + 2.5);
  const cosYaw = Math.cos(cuboid.yaw);
  const sinYaw = Math.sin(cuboid.yaw);

  const localPoints: [number, number, number][] = [];
  const groundCandidates: [number, number, number][] = [];

  const zMin = Math.min(bottomZ - 5.0, cuboid.z - 6.0);
  const zMax = cuboid.z + cuboid.h / 2 + 1.5;

  for (let i = 0; i < numPoints; i++) {
    const p = getPt(i);
    const dx = p[0] - cuboid.x;
    const dy = p[1] - cuboid.y;
    if (dx * dx + dy * dy <= radius * radius) {
      if (p[2] >= zMin && p[2] <= zMax) {
        localPoints.push(p);

        // Check if point is outside the vehicle footprint to avoid body/wheel interference
        const u = dx * cosYaw + dy * sinYaw;
        const v = -dx * sinYaw + dy * cosYaw;
        const isOutsideVehicle = Math.abs(u) > cuboid.l / 2 + 0.10 || Math.abs(v) > cuboid.w / 2 + 0.10;

        if (isOutsideVehicle) {
          groundCandidates.push(p);
        }
      }
    }
  }

  if (localPoints.length < 5) {
    return {
      cuboid: { ...cuboid },
      previousCuboid: { ...cuboid },
      cuboidArray: [...previousCuboidArray],
      previousCuboidArray: [...previousCuboidArray],
      metrics: {
        deltaZ: 0,
        deltaYawDeg: 0,
        groundZ: bottomZ,
        pointsCount: localPoints.length,
        status: 'sparse_points',
      },
    };
  }

  // 2. Road plane detection & snapping
  let groundZ = bottomZ;
  let newZ = cuboid.z;

  if (snapGround) {
    const groundPool = groundCandidates.length >= 8 ?
      groundCandidates :
      localPoints.filter((p) => p[2] <= cuboid.z - cuboid.h * 0.15);

    if (groundPool.length >= 4) {
      const sortedZ = groundPool.map((p) => p[2]).sort((a, b) => a - b);
      const lprCount = Math.max(3, Math.floor(sortedZ.length * 0.20));
      let lprSum = 0;
      for (let k = 0; k < lprCount; k++) {
        lprSum += sortedZ[k];
      }
      const lprZ = lprSum / lprCount;

      const groundSeeds = groundPool.filter((p) => p[2] >= lprZ - 0.12 && p[2] <= lprZ + 0.25);
      const ptsForFit = groundSeeds.length >= 4 ? groundSeeds : groundPool;

      const { plane } = fitGroundPlaneGPF(ptsForFit);
      const rawPlaneZ = getRoadHeightAt(plane, cuboid.x, cuboid.y);
      const [, , nz] = plane;

      if (nz >= 0.85) {
        groundZ = rawPlaneZ;
      } else {
        groundZ = lprZ;
      }

      const groundSurfaceZ = groundZ + 0.02;

      if (Math.abs(groundSurfaceZ - bottomZ) <= 5.0) {
        groundZ = groundSurfaceZ;
      } else {
        groundZ = bottomZ + (groundSurfaceZ > bottomZ ? 5.0 : -5.0);
      }

      newZ = groundZ + cuboid.h / 2;
    }
  }

  // 3. Object points strictly above road surface
  const objectPoints2D: [number, number][] = [];

  for (const p of localPoints) {
    if (p[2] >= groundZ + 0.15 && p[2] <= groundZ + cuboid.h + 0.2) {
      const dx = p[0] - cuboid.x;
      const dy = p[1] - cuboid.y;
      const u = dx * cosYaw + dy * sinYaw;
      const v = -dx * sinYaw + dy * cosYaw;
      if (Math.abs(u) <= cuboid.l / 2 + 0.25 && Math.abs(v) <= cuboid.w / 2 + 0.25) {
        objectPoints2D.push([p[0], p[1]]);
      }
    }
  }

  let newYaw = cuboid.yaw;
  if (autoOrient && objectPoints2D.length >= 8) {
    newYaw = alignYawWithPoints(objectPoints2D, cuboid.yaw, cuboid.l, cuboid.w);
  }

  const deltaZ = newZ - cuboid.z;
  const rawDeltaYaw = (newYaw - cuboid.yaw) * (180 / Math.PI);
  const deltaYawDeg = ((rawDeltaYaw + 180) % 360) - 180;

  const resultCuboid: CuboidParams = {
    x: cuboid.x,
    y: cuboid.y,
    z: parseFloat(newZ.toFixed(2)),
    l: newL,
    w: newW,
    h: newH,
    yaw: parseFloat(newYaw.toFixed(3)),
  };

  const cuboidArray = [
    resultCuboid.x, resultCuboid.y, resultCuboid.z,
    0, 0, resultCuboid.yaw,
    resultCuboid.l, resultCuboid.w, resultCuboid.h,
    0, 0, 0, 0, 0, 0, 0,
  ];

  return {
    cuboid: resultCuboid,
    previousCuboid: { ...cuboid },
    cuboidArray,
    previousCuboidArray,
    metrics: {
      deltaZ: parseFloat(deltaZ.toFixed(2)),
      deltaYawDeg: parseFloat(deltaYawDeg.toFixed(1)),
      groundZ: parseFloat(groundZ.toFixed(2)),
      pointsCount: objectPoints2D.length,
      status: objectPoints2D.length >= 8 ? 'snapped' : 'ground_only',
    },
  };
}
