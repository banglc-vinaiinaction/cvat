// Copyright (C) 2026 CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import * as THREE from 'three';

export interface SnapMetrics {
    deltaZ: number;
    deltaYawDeg: number;
    groundZ: number;
    pointsCount: number;
}

export interface SnapResult {
    points: number[];
    previousPoints: number[];
    metrics: SnapMetrics;
}

/**
 * Computes the eigenvector corresponding to the smallest eigenvalue of a 3x3 symmetric matrix
 * using the Jacobi eigenvalue algorithm.
 */
export function computeSmallestEigenvector3x3(
    c00: number, c01: number, c02: number,
    c11: number, c12: number,
    c22: number,
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
    positions: ArrayLike<number>,
    indices: number[],
    options: {
        lprRatio?: number;
        seedThreshold?: number;
        distanceThreshold?: number;
        refineIterations?: number;
    } = {},
): [number, number, number, number] {
    const {
        lprRatio = 0.25,
        seedThreshold = 0.15,
        distanceThreshold = 0.08,
        refineIterations = 1,
    } = options;

    if (indices.length < 3) {
        return [0, 0, 1, 1.65];
    }

    // Sort indices by ascending Z
    const sorted = [...indices].sort((a, b) => positions[a * 3 + 2] - positions[b * 3 + 2]);

    // Step 1: Lowest Point Representative (LPR)
    const lprCount = Math.max(3, Math.floor(sorted.length * lprRatio));
    let lprZSum = 0;
    for (let i = 0; i < lprCount; i++) {
        lprZSum += positions[sorted[i] * 3 + 2];
    }
    const lprZ = lprZSum / lprCount;

    // Step 2: Seed points collection
    let currentInliers = sorted.filter((idx) => positions[idx * 3 + 2] <= lprZ + seedThreshold);
    if (currentInliers.length < 3) {
        currentInliers = sorted.slice(0, Math.min(sorted.length, 10));
    }

    let bestPlane: [number, number, number, number] = [0, 0, 1, -lprZ];

    // Step 3 & 4: Iterative PCA fit
    const totalPasses = 1 + Math.max(0, refineIterations);
    for (let pass = 0; pass < totalPasses; pass++) {
        const N = currentInliers.length;
        if (N < 3) break;

        // Centroid
        let cx = 0;
        let cy = 0;
        let cz = 0;
        for (let i = 0; i < N; i++) {
            const idx = currentInliers[i];
            cx += positions[idx * 3];
            cy += positions[idx * 3 + 1];
            cz += positions[idx * 3 + 2];
        }
        cx /= N;
        cy /= N;
        cz /= N;

        // Covariance matrix
        let xx = 0; let xy = 0; let xz = 0;
        let yy = 0; let yz = 0; let zz = 0;
        for (let i = 0; i < N; i++) {
            const idx = currentInliers[i];
            const dx = positions[idx * 3] - cx;
            const dy = positions[idx * 3 + 1] - cy;
            const dz = positions[idx * 3 + 2] - cz;
            xx += dx * dx;
            xy += dx * dy;
            xz += dx * dz;
            yy += dy * dy;
            yz += dy * dz;
            zz += dz * dz;
        }

        let [normX, normY, normZ] = computeSmallestEigenvector3x3(
            xx / N, xy / N, xz / N,
            yy / N, yz / N,
            zz / N,
        );

        const len = Math.sqrt(normX * normX + normY * normY + normZ * normZ);
        if (len < 1e-6) {
            normX = 0; normY = 0; normZ = 1;
        } else {
            normX /= len; normY /= len; normZ /= len;
        }

        // Ensure normal points upwards (positive Z)
        if (normZ < 0) {
            normX = -normX;
            normY = -normY;
            normZ = -normZ;
        }

        // Road normal should be roughly vertical (tilt < 30 deg, cos(30) ≈ 0.866)
        if (normZ < 0.85) {
            normX = 0;
            normY = 0;
            normZ = 1;
        }

        const d = -(normX * cx + normY * cy + normZ * cz);
        bestPlane = [normX, normY, normZ, d];

        if (pass < totalPasses - 1) {
            // Refine inliers from all local indices against current plane
            const nextInliers: number[] = [];
            for (let i = 0; i < indices.length; i++) {
                const idx = indices[i];
                const px = positions[idx * 3];
                const py = positions[idx * 3 + 1];
                const pz = positions[idx * 3 + 2];
                const dist = Math.abs(normX * px + normY * py + normZ * pz + d);
                if (dist <= distanceThreshold) {
                    nextInliers.push(idx);
                }
            }
            if (nextInliers.length >= 3) {
                currentInliers = nextInliers;
            } else {
                break;
            }
        }
    }

    return bestPlane;
}

/**
 * Fits a plane Ax + By + Cz + D = 0 to 3D points using RANSAC (alternative baseline).
 */
export function fitGroundPlaneRANSAC(
    positions: ArrayLike<number>,
    indices: number[],
    distanceThreshold: number = 0.08,
    maxIterations: number = 150,
): [number, number, number, number] {
    if (indices.length < 3) {
        return [0, 0, 1, 1.65];
    }

    let bestPlane: [number, number, number, number] = [0, 0, 1, 1.65];
    let maxInliers = -1;
    const N = indices.length;

    for (let iter = 0; iter < maxIterations; iter++) {
        const i1 = indices[Math.floor(Math.random() * N)];
        const i2 = indices[Math.floor(Math.random() * N)];
        const i3 = indices[Math.floor(Math.random() * N)];
        if (i1 === i2 || i2 === i3 || i1 === i3) continue;

        const p1x = positions[i1 * 3];
        const p1y = positions[i1 * 3 + 1];
        const p1z = positions[i1 * 3 + 2];

        const p2x = positions[i2 * 3];
        const p2y = positions[i2 * 3 + 1];
        const p2z = positions[i2 * 3 + 2];

        const p3x = positions[i3 * 3];
        const p3y = positions[i3 * 3 + 1];
        const p3z = positions[i3 * 3 + 2];

        // Normal = (p2 - p1) x (p3 - p1)
        const v1x = p2x - p1x;
        const v1y = p2y - p1y;
        const v1z = p2z - p1z;

        const v2x = p3x - p1x;
        const v2y = p3y - p1y;
        const v2z = p3z - p1z;

        const nx = v1y * v2z - v1z * v2y;
        const ny = v1z * v2x - v1x * v2z;
        const nz = v1x * v2y - v1y * v2x;

        const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
        if (len < 1e-6) continue;

        let normX = nx / len;
        let normY = ny / len;
        let normZ = nz / len;

        // Ensure normal points upwards (positive Z)
        if (normZ < 0) {
            normX = -normX;
            normY = -normY;
            normZ = -normZ;
        }

        // Road normal should be roughly vertical (tilt < 25 deg)
        if (normZ < 0.90) continue;

        const d = -(normX * p1x + normY * p1y + normZ * p1z);

        // Count inliers
        let inliers = 0;
        for (let j = 0; j < N; j++) {
            const idx = indices[j];
            const px = positions[idx * 3];
            const py = positions[idx * 3 + 1];
            const pz = positions[idx * 3 + 2];
            const dist = Math.abs(normX * px + normY * py + normZ * pz + d);
            if (dist < distanceThreshold) inliers++;
        }

        if (inliers > maxInliers) {
            maxInliers = inliers;
            bestPlane = [normX, normY, normZ, d];
        }
    }

    return bestPlane;
}

export function getRoadHeightAt(plane: [number, number, number, number], x: number, y: number): number {
    const [A, B, C, D] = plane;
    if (Math.abs(C) < 1e-5) return -1.65;
    return -(A * x + B * y + D) / C;
}

/**
 * Aligns vehicle heading angle (yaw) with object LiDAR points along the longitudinal axis.
 * Strictly preserves vehicle dimensions (WLH) and limits adjustment to +/- 25 degrees.
 */
export function alignYawWithPoints(
    pts2D: [number, number][],
    currentYaw: number,
    dx: number,
    dy: number,
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

    // If dx >= dy, longitudinal axis is along local X.
    // If dy > dx, longitudinal axis is along local Y (offset by 90 deg).
    const isXLongitudinal = dx >= dy;
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
 * Snaps a 3D cuboid flush to the road plane and aligns orientation to local LiDAR points.
 * Note: Width, Length, and Height (WLH: dx, dy, dz) are strictly preserved.
 */
export function snapCuboidToRoadAndOrient(
    cuboidPoints: number[],
    pointCloudPositions: ArrayLike<number>,
    options: {
        snapGround?: boolean;
        autoOrient?: boolean;
    } = {},
): SnapResult {
    const {
        snapGround = true, autoOrient = true,
    } = options;

    const cx = cuboidPoints[0];
    const cy = cuboidPoints[1];
    const cz = cuboidPoints[2];
    const rx = cuboidPoints[3];
    const ry = cuboidPoints[4];
    const rz = cuboidPoints[5]; // yaw
    const dx = cuboidPoints[6]; // scale X
    const dy = cuboidPoints[7]; // scale Y
    const dz = cuboidPoints[8]; // scale Z (height)

    // Preserve exact dimensions (WLH)
    const newDx = dx;
    const newDy = dy;
    const newDz = dz;

    const bottomZ = cz - dz / 2;
    const totalPoints = Math.floor(pointCloudPositions.length / 3);
    const radius = Math.sqrt((dx / 2) ** 2 + (dy / 2) ** 2) + 2.5;

    const cosRz = Math.cos(rz);
    const sinRz = Math.sin(rz);

    const localIndices: number[] = [];
    const groundCandidateIndices: number[] = [];

    for (let i = 0; i < totalPoints; i++) {
        const px = pointCloudPositions[i * 3];
        const py = pointCloudPositions[i * 3 + 1];
        const pz = pointCloudPositions[i * 3 + 2];

        const distX = px - cx;
        const distY = py - cy;
        if (distX * distX + distY * distY <= radius * radius) {
            if (pz >= bottomZ - 1.2 && pz <= cz + dz / 2 + 0.8) {
                localIndices.push(i);

                // Check distance outside vehicle footprint to exclude wheels/body
                const u = distX * cosRz + distY * sinRz;
                const v = -distX * sinRz + distY * cosRz;
                const isOutsideVehicle = Math.abs(u) > dx / 2 + 0.15 || Math.abs(v) > dy / 2 + 0.15;

                if (isOutsideVehicle && pz <= bottomZ + 0.35 && pz >= bottomZ - 0.8) {
                    groundCandidateIndices.push(i);
                }
            }
        }
    }

    if (localIndices.length < 5) {
        return {
            points: [...cuboidPoints],
            previousPoints: [...cuboidPoints],
            metrics: {
                deltaZ: 0,
                deltaYawDeg: 0,
                groundZ: bottomZ,
                pointsCount: localIndices.length,
            },
        };
    }

    // 2. Road plane detection & snapping
    let groundZ = bottomZ;
    let newZ = cz;

    if (snapGround) {
        const ptsForGround = groundCandidateIndices.length >= 8 ?
            groundCandidateIndices :
            localIndices.filter((idx) => {
                const pz = pointCloudPositions[idx * 3 + 2];
                return pz <= bottomZ + 0.25 && pz >= bottomZ - 0.8;
            });

        if (ptsForGround.length >= 4) {
            const sortedGroundZ = ptsForGround.map(
                (idx) => pointCloudPositions[idx * 3 + 2],
            ).sort((a, b) => a - b);

            // Robust 10th percentile for road elevation
            const p10Idx = Math.max(0, Math.floor(sortedGroundZ.length * 0.10));
            const robustZ = sortedGroundZ[p10Idx];

            // Fit GPF plane
            const plane = fitGroundPlaneGPF(pointCloudPositions, ptsForGround, {
                lprRatio: 0.2,
                seedThreshold: 0.12,
                distanceThreshold: 0.06,
            });

            const rawPlaneZ = getRoadHeightAt(plane, cx, cy);
            const [, , nz] = plane;
            if (nz >= 0.95 && Math.abs(rawPlaneZ - robustZ) <= 0.15) {
                groundZ = rawPlaneZ;
            } else {
                groundZ = robustZ;
            }

            // Clamp max adjustment to avoid unexpected jumping
            if (Math.abs(groundZ - bottomZ) > 1.2) {
                groundZ = bottomZ + (groundZ > bottomZ ? 1.2 : -1.2);
            }

            newZ = groundZ + dz / 2;
        }
    }

    // 3. Object points strictly above road surface
    const objectPoints2D: [number, number][] = [];

    for (const idx of localIndices) {
        const px = pointCloudPositions[idx * 3];
        const py = pointCloudPositions[idx * 3 + 1];
        const pz = pointCloudPositions[idx * 3 + 2];

        // Points above ground surface and within vehicle height
        if (pz >= groundZ + 0.15 && pz <= groundZ + dz + 0.2) {
            const distX = px - cx;
            const distY = py - cy;
            const u = distX * cosRz + distY * sinRz;
            const v = -distX * sinRz + distY * cosRz;
            if (Math.abs(u) <= dx / 2 + 0.25 && Math.abs(v) <= dy / 2 + 0.25) {
                objectPoints2D.push([px, py]);
            }
        }
    }

    let newYaw = rz;
    if (autoOrient && objectPoints2D.length >= 8) {
        newYaw = alignYawWithPoints(objectPoints2D, rz, dx, dy);
    }

    const deltaZ = newZ - cz;
    const rawDeltaYaw = (newYaw - rz) * (180 / Math.PI);
    const deltaYawDeg = ((rawDeltaYaw + 180) % 360) - 180;

    const resultPoints = [
        parseFloat(cx.toFixed(3)),
        parseFloat(cy.toFixed(3)),
        parseFloat(newZ.toFixed(3)),
        rx,
        ry,
        parseFloat(newYaw.toFixed(4)),
        newDx,
        newDy,
        newDz,
        0, 0, 0, 0, 0, 0, 0,
    ];

    return {
        points: resultPoints,
        previousPoints: [...cuboidPoints],
        metrics: {
            deltaZ: parseFloat(deltaZ.toFixed(3)),
            deltaYawDeg: parseFloat(deltaYawDeg.toFixed(1)),
            groundZ: parseFloat(groundZ.toFixed(3)),
            pointsCount: objectPoints2D.length,
        },
    };
}
