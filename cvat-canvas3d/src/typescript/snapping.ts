// Copyright (C) 2026 CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import * as THREE from 'three';
import { getClassDimensionPrior, isPedestrianClass, CLASS_DIMENSION_PRIORS } from './consts';

export interface SnapMetrics {
    deltaZ: number;
    deltaYawDeg: number;
    groundZ: number;
    pointsCount: number;
    deltaHeight?: number;
    deltaWidth?: number;
    deltaLength?: number;
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
    // eslint-disable-next-line prefer-destructuring
    if (A[1][1] < minVal) { minVal = A[1][1]; minIdx = 1; }
    // eslint-disable-next-line prefer-destructuring
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
        lprRatio = 0.15,
        seedThreshold = 0.10,
        distanceThreshold = 0.08,
        refineIterations = 1,
    } = options;

    if (indices.length < 3) {
        let estZ = 0.0;
        if (positions.length >= 3) {
            const sampleCount = Math.min(100, Math.floor(positions.length / 3));
            let sumZ = 0;
            for (let i = 0; i < sampleCount; i++) sumZ += positions[i * 3 + 2];
            estZ = sumZ / sampleCount;
        }
        return [0, 0, 1, -estZ];
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
        let estZ = 0.0;
        if (positions.length >= 3) {
            const sampleCount = Math.min(100, Math.floor(positions.length / 3));
            let sumZ = 0;
            for (let i = 0; i < sampleCount; i++) sumZ += positions[i * 3 + 2];
            estZ = sumZ / sampleCount;
        }
        return [0, 0, 1, -estZ];
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
 * Heading direction is strictly preserved and never flips (requires heading dot product > 0.7 relative to currentYaw).
 */
export function alignYawWithPoints(
    pts2D: [number, number][],
    currentYaw: number,
    dx: number,
    dy: number,
    motionYaw?: number | null,
): number {
    const normalizeAngle = (a: number): number => Math.atan2(Math.sin(a), Math.cos(a));

    if (pts2D.length < 8) {
        if (typeof motionYaw === 'number' && Math.cos(motionYaw - currentYaw) > 0.85) {
            return normalizeAngle(motionYaw);
        }
        return currentYaw;
    }

    // Work strictly in the vehicle's local frame relative to currentYaw:
    // u: longitudinal axis (along vehicle length / heading)
    // v: lateral axis (along vehicle width)
    const cosY = Math.cos(currentYaw);
    const sinY = Math.sin(currentYaw);
    let meanU = 0;
    let meanV = 0;
    const N = pts2D.length;

    for (let i = 0; i < N; i++) {
        const px = pts2D[i][0];
        const py = pts2D[i][1];
        meanU += px * cosY + py * sinY;
        meanV += -px * sinY + py * cosY;
    }
    meanU /= N;
    meanV /= N;

    let cuu = 0;
    let cuv = 0;
    let cvv = 0;
    for (let i = 0; i < N; i++) {
        const px = pts2D[i][0];
        const py = pts2D[i][1];
        const du = (px * cosY + py * sinY) - meanU;
        const dv = (-px * sinY + py * cosY) - meanV;
        cuu += du * du;
        cuv += du * dv;
        cvv += dv * dv;
    }

    // Guard against partial returns: If variance along length (cuu) is not significantly
    // greater than width (cvv), the points are either on a bumper or isotropic cluster.
    // Under no circumstances should bumper points twist the vehicle 90 degrees or flip 180 degrees.
    if (cuu <= cvv * 1.5) {
        return currentYaw;
    }

    // Principal orientation in vehicle local frame:
    const localDeltaAngle = 0.5 * Math.atan2(2 * cuv, cuu - cvv);
    const maxAllowedAdjustment = (5 * Math.PI) / 180; // +/- 5 degrees max subtle refinement

    if (Math.abs(localDeltaAngle) <= maxAllowedAdjustment) {
        return normalizeAngle(currentYaw + localDeltaAngle);
    }

    return currentYaw;
}

/**
 * Snaps a 3D cuboid flush to the road plane and aligns orientation to local LiDAR points.
 * Applies class-specific dimension priors (pedestrian, two-wheeler, heavy vehicle) and
 * resolves heading using trajectory velocity vectors when available.
 */
export function snapCuboidToRoadAndOrient(
    cuboidPoints: number[],
    pointCloudPositions: ArrayLike<number>,
    options: {
        snapGround?: boolean;
        autoOrient?: boolean;
        snapGeometry?: boolean;
        labelName?: string;
        motionYaw?: number | null;
        groundClearance?: number;
    } = {},
): SnapResult {
    const {
        snapGround = true, autoOrient = true, snapGeometry = true, groundClearance = 0.04,
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

    // Dimensions remain strictly constant during snap and orient (LWH invariant)
    const newDx = dx;
    const newDy = dy;
    const newDz = dz;
    let finalCx = cx;
    let finalCy = cy;

    const bottomZ = cz - dz / 2;
    const totalPoints = Math.floor(pointCloudPositions.length / 3);
    const radius = Math.max(3.0, Math.sqrt((dx / 2) ** 2 + (dy / 2) ** 2) + 2.5);
    const r2 = radius * radius;

    const cosRz = Math.cos(rz);
    const sinRz = Math.sin(rz);

    const localIndices: number[] = [];
    const groundCandidateIndices: number[] = [];

    // Allow searching down beneath the cuboid or up if cuboid was placed sunken
    const zMin = Math.min(bottomZ - 5.0, cz - 6.0);
    const zMax = Math.max(cz + dz / 2 + 2.0, bottomZ + dz + 2.0);
    const zMaxGround = Math.max(bottomZ + 0.60, cz + dz * 0.25);
    const halfDxPad = dx / 2 + 0.10;
    const halfDyPad = dy / 2 + 0.10;

    for (let i = 0; i < totalPoints; i++) {
        const i3 = i * 3;
        const distX = pointCloudPositions[i3] - cx;
        if (distX > radius || distX < -radius) continue;
        const distY = pointCloudPositions[i3 + 1] - cy;
        if (distY > radius || distY < -radius) continue;
        const pz = pointCloudPositions[i3 + 2];
        if (pz < zMin || pz > zMax) continue;

        if (distX * distX + distY * distY <= r2) {
            localIndices.push(i);

            // Check distance outside vehicle footprint to exclude wheels/body
            const u = distX * cosRz + distY * sinRz;
            const v = -distX * sinRz + distY * cosRz;
            const isOutsideVehicle = Math.abs(u) > halfDxPad || Math.abs(v) > halfDyPad;

            if (isOutsideVehicle && pz <= zMaxGround) {
                groundCandidateIndices.push(i);
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
    let detectedPlane: [number, number, number, number] | null = null;

    if (snapGround) {
        const groundPool = groundCandidateIndices.length >= 8 ?
            groundCandidateIndices :
            localIndices.filter((idx) => {
                const pz = pointCloudPositions[idx * 3 + 2];
                return pz <= zMaxGround;
            });

        if (groundPool.length >= 8) {
            const sortedGroundZ = groundPool.map(
                (idx) => pointCloudPositions[idx * 3 + 2],
            ).sort((a, b) => a - b);

            // Compute Lowest Point Representative (LPR) of local ground cluster
            const lprCount = Math.max(3, Math.floor(sortedGroundZ.length * 0.15));
            let lprSum = 0;
            for (let k = 0; k < lprCount; k++) {
                lprSum += sortedGroundZ[k];
            }
            const lprZ = lprSum / lprCount;

            // Allow ground detection even if cuboid bottom is sunken beneath the ground
            if (lprZ <= bottomZ + 3.0) {
                // Extract candidate ground seeds tightly around LPR
                const groundSeeds = groundPool.filter((idx) => {
                    const pz = pointCloudPositions[idx * 3 + 2];
                    return pz >= lprZ - 0.12 && pz <= lprZ + 0.20;
                });

                // Fit GPF plane on ground seeds
                const ptsForFit = groundSeeds.length >= 4 ? groundSeeds : groundPool;
                const plane = fitGroundPlaneGPF(pointCloudPositions, ptsForFit, {
                    lprRatio: 0.25,
                    seedThreshold: 0.15,
                    distanceThreshold: 0.08,
                });

                detectedPlane = plane;
                const rawPlaneZ = getRoadHeightAt(plane, cx, cy);
                const [, , nz] = plane;

                if (nz >= 0.85) {
                    groundZ = Math.max(lprZ - 0.20, Math.min(lprZ + 0.35, rawPlaneZ));
                } else {
                    groundZ = lprZ;
                }

                // Road surface upper envelope adjustment:
                // Automotive LiDAR ground points have vertical beam scatter (~2-3cm).
                // Use the 90th percentile of consensus ground seeds.
                const seedZValues = (groundSeeds.length >= 3 ? groundSeeds : groundPool)
                    .map((idx) => pointCloudPositions[idx * 3 + 2])
                    .sort((a, b) => a - b);
                const p90Idx = Math.min(seedZValues.length - 1, Math.floor(seedZValues.length * 0.90));
                const groundP90Z = seedZValues[p90Idx];
                const baseGroundZ = Math.max(groundZ, groundP90Z) + groundClearance;

                // Restrict ground snapping adjustment to realistic range, but allow pulling sunken boxes up
                if (Math.abs(baseGroundZ - bottomZ) <= 0.45) {
                    groundZ = baseGroundZ;
                } else if (baseGroundZ > bottomZ) {
                    groundZ = baseGroundZ;
                } else {
                    groundZ = bottomZ - 0.25;
                }

                newZ = groundZ + dz / 2;
            }
        }
    }

    // 3. Object points strictly above road surface
    const objectPoints: { px: number; py: number; pz: number }[] = [];
    const objectPoints2D: [number, number][] = [];

    const searchMarginX = Math.max(0.35, dx * 0.1);
    const searchMarginY = Math.max(0.35, dy * 0.1);

    for (const idx of localIndices) {
        const px = pointCloudPositions[idx * 3];
        const py = pointCloudPositions[idx * 3 + 1];
        const pz = pointCloudPositions[idx * 3 + 2];

        // Points above ground surface and within vehicle height + margin
        if (pz >= groundZ + 0.10 && pz <= groundZ + Math.max(dz, 1.6) + 0.5) {
            const distX = px - cx;
            const distY = py - cy;
            const u = distX * cosRz + distY * sinRz;
            const v = -distX * sinRz + distY * cosRz;
            if (Math.abs(u) <= dx / 2 + searchMarginX && Math.abs(v) <= dy / 2 + searchMarginY) {
                objectPoints.push({ px, py, pz });
                objectPoints2D.push([px, py]);
            }
        }
    }

    let newYaw = rz;
    if (autoOrient && objectPoints2D.length >= 8) {
        const candidateYaw = alignYawWithPoints(objectPoints2D, rz, dx, dy, options.motionYaw);
        if (Math.cos(candidateYaw - rz) > 0.7) {
            newYaw = candidateYaw;
        } else {
            newYaw = rz;
        }
    } else {
        newYaw = rz;
    }

    // Auto-tighten dimensions & center on vehicle points if sufficient points exist
    if (snapGeometry && objectPoints.length >= 10) {
        const cosYaw = Math.cos(newYaw);
        const sinYaw = Math.sin(newYaw);

        const uValues: number[] = [];
        const vValues: number[] = [];
        const zValues: number[] = [];

        for (const pt of objectPoints) {
            const distX = pt.px - cx;
            const distY = pt.py - cy;
            const u = distX * cosYaw + distY * sinYaw;
            const v = -distX * sinYaw + distY * cosYaw;
            uValues.push(u);
            vValues.push(v);
            zValues.push(pt.pz);
        }

        uValues.sort((a, b) => a - b);
        vValues.sort((a, b) => a - b);
        zValues.sort((a, b) => a - b);

        const nPts = objectPoints.length;
        const p05Idx = Math.floor(nPts * 0.05);
        const p95Idx = Math.min(nPts - 1, Math.floor(nPts * 0.95));
        const p98Idx = Math.min(nPts - 1, Math.floor(nPts * 0.98));

        const uMin = uValues[p05Idx];
        const uMax = uValues[p95Idx];
        const vMin = vValues[p05Idx];
        const vMax = vValues[p95Idx];
        const zTop = zValues[p98Idx];

        // Lateral centering (box dimensions dx, dy, dz remain strictly constant):
        if (vMin < -0.15 && vMax > 0.15) {
            const shiftV = (vMin + vMax) / 2;
            const clampedShiftV = Math.max(-0.35, Math.min(0.35, shiftV));
            finalCx += -clampedShiftV * sinYaw;
            finalCy += clampedShiftV * cosYaw;
        }

        // Longitudinal centering (box dimensions dx, dy, dz remain strictly constant):
        if (uMax - uMin >= 2.0) {
            const shiftU = (uMin + uMax) / 2;
            const clampedShiftU = Math.max(-0.4, Math.min(0.4, shiftU));
            finalCx += clampedShiftU * cosYaw;
            finalCy += clampedShiftU * sinYaw;
        }
    }

    // 4. Upright Posture for Pedestrians (dimensions remain strictly constant)
    const isPedestrian = isPedestrianClass(options.labelName);

    // 4. Sectional Terrain Profiling: Angle cuboid between highest average point vs lowest average point and everything in-between
    let finalRx = 0.0;
    let finalRy = 0.0;
    const finalRz = parseFloat(newYaw.toFixed(4));

    if (!isPedestrian && localIndices.length >= 8) {
        const cosYaw = Math.cos(finalRz);
        const sinYaw = Math.sin(finalRz);
        const halfL = newDx / 2;
        const halfW = newDy / 2;
        const marginX = Math.max(0.35, newDx * 0.10);
        const marginY = Math.max(0.35, newDy * 0.10);

        // Ground candidates in footprint
        interface LocalGroundPt { u: number; v: number; z: number; }
        const footprintGround: LocalGroundPt[] = [];

        const poolIndices = groundCandidateIndices.length >= 8 ? groundCandidateIndices : localIndices;
        for (let k = 0; k < poolIndices.length; k++) {
            const idx = poolIndices[k];
            const px = pointCloudPositions[idx * 3];
            const py = pointCloudPositions[idx * 3 + 1];
            const pz = pointCloudPositions[idx * 3 + 2];
            const distX = px - finalCx;
            const distY = py - finalCy;
            const u = distX * cosYaw + distY * sinYaw;
            const v = -distX * sinYaw + distY * cosYaw;

            const currentBaseZ = groundZ > bottomZ ? groundZ : bottomZ;
            if (Math.abs(u) <= halfL + marginX && Math.abs(v) <= halfW + marginY) {
                if (pz <= currentBaseZ + 0.35 && pz >= currentBaseZ - 0.70) {
                    footprintGround.push({ u, v, z: pz });
                }
            }
        }

        if (footprintGround.length >= 10) {
            // Find Lowest Point Representative
            const sortedFootprintZ = footprintGround.map((pt) => pt.z).sort((a, b) => a - b);
            const lprSeedCount = Math.max(3, Math.floor(sortedFootprintZ.length * 0.15));
            let lprSum = 0;
            for (let i = 0; i < lprSeedCount; i++) lprSum += sortedFootprintZ[i];
            const lprZ = lprSum / lprSeedCount;

            // Restrict to surface points tightly around LPR to exclude high obstacles
            const surfacePts = footprintGround.filter((pt) => pt.z >= lprZ - 0.15 && pt.z <= lprZ + 0.22);

            // 1. Longitudinal slices along vehicle length (Rear -> Center -> Front)
            const numBinsX = 5;
            const stepX = (halfL * 2) / numBinsX;
            const binCentersX: number[] = [];
            const binAvgZX: number[] = [];

            for (let b = 0; b < numBinsX; b++) {
                const xMin = -halfL + b * stepX - 0.10;
                const xMax = -halfL + (b + 1) * stepX + 0.10;
                let sumZ = 0;
                let count = 0;
                for (let i = 0; i < surfacePts.length; i++) {
                    const pt = surfacePts[i];
                    if (pt.u >= xMin && pt.u <= xMax) {
                        sumZ += pt.z;
                        count++;
                    }
                }
                if (count >= 2) {
                    binCentersX.push(-halfL + (b + 0.5) * stepX);
                    binAvgZX.push(sumZ / count);
                }
            }

            // 2. Lateral slices across vehicle width (Right -> Center -> Left)
            const numBinsY = 3;
            const stepY = (halfW * 2) / numBinsY;
            const binCentersY: number[] = [];
            const binAvgZY: number[] = [];

            for (let b = 0; b < numBinsY; b++) {
                const yMin = -halfW + b * stepY - 0.10;
                const yMax = -halfW + (b + 1) * stepY + 0.10;
                let sumZ = 0;
                let count = 0;
                for (let i = 0; i < surfacePts.length; i++) {
                    const pt = surfacePts[i];
                    if (pt.v >= yMin && pt.v <= yMax) {
                        sumZ += pt.z;
                        count++;
                    }
                }
                if (count >= 2) {
                    binCentersY.push(-halfW + (b + 0.5) * stepY);
                    binAvgZY.push(sumZ / count);
                }
            }

            // 3. Slope regression between highest average point, lowest average point, and everything in-between
            let slopeX = 0;
            if (binCentersX.length >= 2) {
                let meanX = 0; let meanZ = 0;
                for (let i = 0; i < binCentersX.length; i++) {
                    meanX += binCentersX[i];
                    meanZ += binAvgZX[i];
                }
                meanX /= binCentersX.length;
                meanZ /= binCentersX.length;

                let varX = 0; let covXZ = 0;
                for (let i = 0; i < binCentersX.length; i++) {
                    const dx = binCentersX[i] - meanX;
                    varX += dx * dx;
                    covXZ += dx * (binAvgZX[i] - meanZ);
                }
                if (varX > 1e-4) {
                    slopeX = covXZ / varX;
                }
            }

            let slopeY = 0;
            if (binCentersY.length >= 2) {
                let meanY = 0; let meanZ = 0;
                for (let i = 0; i < binCentersY.length; i++) {
                    meanY += binCentersY[i];
                    meanZ += binAvgZY[i];
                }
                meanY /= binCentersY.length;
                meanZ /= binCentersY.length;

                let varY = 0; let covYZ = 0;
                for (let i = 0; i < binCentersY.length; i++) {
                    const dy = binCentersY[i] - meanY;
                    varY += dy * dy;
                    covYZ += dy * (binAvgZY[i] - meanZ);
                }
                if (varY > 1e-4) {
                    slopeY = covYZ / varY;
                }
            }

            // Clamp slopes to realistic roadway grades (+- 10 deg = ~0.176 rad/m)
            const maxSlope = 0.176;
            slopeX = Math.max(-maxSlope, Math.min(maxSlope, slopeX));
            slopeY = Math.max(-maxSlope, Math.min(maxSlope, slopeY));

            // Average ground height across all observed bins (everything in-between)
            const allBinZ = [...binAvgZX, ...binAvgZY];
            let meanGroundZ = lprZ;
            if (allBinZ.length > 0) {
                let sumZ = 0;
                for (let i = 0; i < allBinZ.length; i++) sumZ += allBinZ[i];
                meanGroundZ = sumZ / allBinZ.length;
            }

            // Balanced in-between elevation: base rests at meanGroundZ + groundClearance
            const baseZ = meanGroundZ + groundClearance;
            newZ = baseZ + newDz / 2;

            // Restrict vertical shift to realistic bound relative to cz (allow pulling sunken boxes up to 3.0m)
            newZ = Math.max(cz - 0.50, Math.min(cz + 3.0, newZ));

            // Convert longitudinal tilt (thetaLong) and lateral tilt (thetaLat) to 3D Euler angles (XYZ)
            const thetaLong = Math.atan(slopeX);
            const thetaLat = Math.atan(slopeY);

            if (Math.abs(thetaLong) > 1e-4 || Math.abs(thetaLat) > 1e-4) {
                const fwdLocal = new THREE.Vector3(Math.cos(thetaLong), 0, Math.sin(thetaLong));
                const leftLocal = new THREE.Vector3(0, Math.cos(thetaLat), Math.sin(thetaLat));
                const upLocal = new THREE.Vector3().crossVectors(fwdLocal, leftLocal).normalize();
                const reLeftLocal = new THREE.Vector3().crossVectors(upLocal, fwdLocal).normalize();
                const reFwdLocal = new THREE.Vector3().crossVectors(reLeftLocal, upLocal).normalize();

                const rotBasis = new THREE.Matrix4().makeBasis(reFwdLocal, reLeftLocal, upLocal);
                const rotYaw = new THREE.Matrix4().makeRotationZ(finalRz);
                const rotWorld = new THREE.Matrix4().multiplyMatrices(rotYaw, rotBasis);

                const euler = new THREE.Euler().setFromRotationMatrix(rotWorld, 'XYZ');
                finalRx = parseFloat(euler.x.toFixed(4));
                finalRy = parseFloat(euler.y.toFixed(4));
            }
        }
    }

    const deltaZ = newZ - cz;
    const rawDeltaYaw = (finalRz - rz) * (180 / Math.PI);
    const deltaYawDeg = ((rawDeltaYaw + 180) % 360) - 180;

    const resultPoints = [
        parseFloat(finalCx.toFixed(3)),
        parseFloat(finalCy.toFixed(3)),
        parseFloat(newZ.toFixed(3)),
        finalRx,
        finalRy,
        parseFloat(finalRz.toFixed(4)),
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
            deltaHeight: 0,
            deltaWidth: 0,
            deltaLength: 0,
        },
    };
}

/**
 * Projects a horizontal yaw heading onto the 3D road surface normal to compute pitch (rx) and roll (ry).
 */
export function alignEulerToRoadPlane(
    yaw: number,
    detectedPlane: [number, number, number, number] | null,
): { rx: number; ry: number; rz: number } {
    if (!detectedPlane || detectedPlane[2] < 0.80) {
        return { rx: 0, ry: 0, rz: yaw };
    }

    const [nA, nB, nC] = detectedPlane;
    const norm = new THREE.Vector3(nA, nB, nC).normalize();
    const fHoriz = new THREE.Vector3(Math.cos(yaw), Math.sin(yaw), 0);
    const dot = fHoriz.dot(norm);
    const xLocal = fHoriz.clone().sub(norm.clone().multiplyScalar(dot));

    if (xLocal.lengthSq() > 1e-6) {
        xLocal.normalize();
        const yLocal = new THREE.Vector3().crossVectors(norm, xLocal).normalize();
        const zLocal = norm;

        const basisMatrix = new THREE.Matrix4().makeBasis(xLocal, yLocal, zLocal);
        const euler = new THREE.Euler().setFromRotationMatrix(basisMatrix, 'XYZ');

        const maxTilt = (20 * Math.PI) / 180;
        if (Math.abs(euler.x) <= maxTilt && Math.abs(euler.y) <= maxTilt) {
            return {
                rx: parseFloat(euler.x.toFixed(4)),
                ry: parseFloat(euler.y.toFixed(4)),
                rz: parseFloat(yaw.toFixed(4)),
            };
        }
    }

    return { rx: 0, ry: 0, rz: yaw };
}

/**
 * Estimates the local road plane near (x, y, zApprox) within radius R.
 * Returns [nA, nB, nC, nD] if valid road plane with normal.z >= 0.80, or null.
 */
export function estimateLocalRoadPlane(
    x: number,
    y: number,
    zApprox: number,
    positions: ArrayLike<number>,
    radius: number = 2.5,
): [number, number, number, number] | null {
    if (!positions || positions.length < 9) return null;

    const r2 = radius * radius;
    const totalPoints = Math.floor(positions.length / 3);
    const localIndices: number[] = [];
    const zMin = zApprox - 3.5;
    const zMax = zApprox + 3.5;

    for (let i = 0; i < totalPoints; i++) {
        const px = positions[i * 3];
        if (Math.abs(px - x) > radius) continue;
        const py = positions[i * 3 + 1];
        if (Math.abs(py - y) > radius) continue;
        const pz = positions[i * 3 + 2];
        if (pz < zMin || pz > zMax) continue;

        if ((px - x) ** 2 + (py - y) ** 2 <= r2) {
            localIndices.push(i);
            if (localIndices.length >= 250) break;
        }
    }

    if (localIndices.length < 4) return null;

    const sortedIndices = [...localIndices].sort(
        (a, b) => positions[a * 3 + 2] - positions[b * 3 + 2],
    );
    const lprCount = Math.max(3, Math.floor(sortedIndices.length * 0.25));
    let lprSum = 0;
    for (let k = 0; k < lprCount; k++) {
        lprSum += positions[sortedIndices[k] * 3 + 2];
    }
    const lprZ = lprSum / lprCount;

    const groundSeeds = sortedIndices.filter((idx) => {
        const pz = positions[idx * 3 + 2];
        return pz >= lprZ - 0.15 && pz <= lprZ + 0.25;
    });

    const ptsForFit = groundSeeds.length >= 4 ?
        groundSeeds :
        sortedIndices.slice(0, Math.min(sortedIndices.length, 20));

    const plane = fitGroundPlaneGPF(positions, ptsForFit, {
        lprRatio: 0.25,
        seedThreshold: 0.15,
        distanceThreshold: 0.08,
    });

    if (plane[2] >= 0.80) {
        return plane;
    }

    return null;
}

export interface ClusterPoint {
    px: number;
    py: number;
    pz: number;
}

/**
 * Extracts candidate object cluster points within a cylindrical ROI around `center`,
 * filtering out ground/pavement points below `localRoadZ + minHeightAboveGround`.
 */
export function extractClusterPoints(
    center: [number, number, number],
    positions: ArrayLike<number>,
    options: {
        radius?: number;
        zGround?: number;
        detectedPlane?: [number, number, number, number] | null;
        minHeightAboveGround?: number;
        maxHeightAboveGround?: number;
    } = {},
): ClusterPoint[] {
    const [cx, cy, cz] = center;
    const {
        radius = 2.5,
        zGround = cz,
        detectedPlane = null,
        minHeightAboveGround = 0.15,
        maxHeightAboveGround = 2.6,
    } = options;

    if (!positions || positions.length < 9) return [];

    const r2 = radius * radius;
    const totalPoints = Math.floor(positions.length / 3);
    const zMin = cz - 4.0;
    const zMax = cz + 4.0;
    const clusterPoints: ClusterPoint[] = [];

    for (let i = 0; i < totalPoints; i++) {
        const px = positions[i * 3];
        if (Math.abs(px - cx) > radius) continue;
        const py = positions[i * 3 + 1];
        if (Math.abs(py - cy) > radius) continue;
        const pz = positions[i * 3 + 2];
        if (pz < zMin || pz > zMax) continue;

        if ((px - cx) ** 2 + (py - cy) ** 2 <= r2) {
            const localRoadZ = detectedPlane ? getRoadHeightAt(detectedPlane, px, py) : zGround;
            if (pz >= localRoadZ + minHeightAboveGround && pz <= localRoadZ + maxHeightAboveGround) {
                clusterPoints.push({ px, py, pz });
            }
        }
    }

    return clusterPoints;
}

export interface ClusterYawResult {
    yaw: number;
    centroidX: number;
    centroidY: number;
}

/**
 * Computes 2D centroid and PCA orientation for a set of cluster points.
 * Returns null if points count is below minPoints (default 8).
 */
export function solveClusterYaw(
    points: { px: number; py: number }[],
    isXLongitudinal: boolean = true,
    minPoints: number = 8,
): ClusterYawResult | null {
    if (!points || points.length < minPoints) {
        return null;
    }

    let sumX = 0;
    let sumY = 0;
    for (let i = 0; i < points.length; i++) {
        sumX += points[i].px;
        sumY += points[i].py;
    }
    const centroidX = sumX / points.length;
    const centroidY = sumY / points.length;

    let cxx = 0;
    let cxy = 0;
    let cyy = 0;
    for (let i = 0; i < points.length; i++) {
        const x = points[i].px - centroidX;
        const y = points[i].py - centroidY;
        cxx += x * x;
        cxy += x * y;
        cyy += y * y;
    }

    const pcaAngle = 0.5 * Math.atan2(2 * cxy, cxx - cyy);
    let yaw = isXLongitudinal ? pcaAngle : pcaAngle - Math.PI / 2;
    yaw = Math.atan2(Math.sin(yaw), Math.cos(yaw));

    return {
        yaw,
        centroidX,
        centroidY,
    };
}

export interface PrecomputedGroundGridData {
    cellSize: number;
    minX: number;
    minY: number;
    maxX: number;
    maxY: number;
    cols: number;
    rows: number;
    grid: Float32Array;
}

/**
 * Coarse 2D ground elevation spatial grid / heightmap cache.
 * Populated once when a point cloud frame loads, enabling O(1) road elevation
 * and road plane normal lookups during mousemove/rendering without linearly
 * scanning the point cloud array every frame.
 */
export class GroundElevationGrid {
    public readonly cellSize: number;
    public readonly minX: number;
    public readonly minY: number;
    public readonly maxX: number;
    public readonly maxY: number;
    public readonly cols: number;
    public readonly rows: number;
    private readonly grid: Float32Array;

    public static fromPrecomputed(data: PrecomputedGroundGridData): GroundElevationGrid {
        const instance = Object.create(GroundElevationGrid.prototype) as GroundElevationGrid;
        (instance as any).cellSize = data.cellSize;
        (instance as any).minX = data.minX;
        (instance as any).minY = data.minY;
        (instance as any).maxX = data.maxX;
        (instance as any).maxY = data.maxY;
        (instance as any).cols = data.cols;
        (instance as any).rows = data.rows;
        (instance as any).grid = data.grid;
        return instance;
    }

    constructor(positions: ArrayLike<number>, cellSize: number = 0.5) {
        this.cellSize = cellSize > 0 ? cellSize : 0.5;

        if (!positions || positions.length < 9) {
            this.minX = 0;
            this.minY = 0;
            this.maxX = 0;
            this.maxY = 0;
            this.cols = 0;
            this.rows = 0;
            this.grid = new Float32Array(0);
            return;
        }

        const totalPoints = Math.floor(positions.length / 3);
        let minX = Infinity;
        let maxX = -Infinity;
        let minY = Infinity;
        let maxY = -Infinity;

        for (let i = 0; i < totalPoints; i++) {
            const px = positions[i * 3];
            const py = positions[i * 3 + 1];
            const pz = positions[i * 3 + 2];
            if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz)) continue;
            if (px < minX) minX = px;
            if (px > maxX) maxX = px;
            if (py < minY) minY = py;
            if (py > maxY) maxY = py;
        }

        if (!Number.isFinite(minX) || !Number.isFinite(minY)) {
            this.minX = 0;
            this.minY = 0;
            this.maxX = 0;
            this.maxY = 0;
            this.cols = 0;
            this.rows = 0;
            this.grid = new Float32Array(0);
            return;
        }

        let effectiveCellSize = this.cellSize;
        const width = maxX - minX;
        const height = maxY - minY;
        let c = Math.ceil(width / effectiveCellSize) + 1;
        let r = Math.ceil(height / effectiveCellSize) + 1;

        if (c * r > 1_000_000) {
            effectiveCellSize = Math.max(width, height) / 1000;
            c = Math.ceil(width / effectiveCellSize) + 1;
            r = Math.ceil(height / effectiveCellSize) + 1;
        }

        this.cellSize = effectiveCellSize;
        this.minX = minX;
        this.minY = minY;
        this.maxX = maxX;
        this.maxY = maxY;
        this.cols = c;
        this.rows = r;

        const invCellSize = 1.0 / effectiveCellSize;
        const grid = new Float32Array(c * r);
        grid.fill(Infinity);

        for (let i = 0; i < totalPoints; i++) {
            const i3 = i * 3;
            const px = positions[i3];
            const py = positions[i3 + 1];
            const pz = positions[i3 + 2];
            if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz)) continue;

            const col = Math.floor((px - minX) * invCellSize);
            const row = Math.floor((py - minY) * invCellSize);
            if (col >= 0 && col < c && row >= 0 && row < r) {
                const idx = row * c + col;
                if (pz < grid[idx]) {
                    grid[idx] = pz;
                }
            }
        }

        this.grid = grid;
    }

    /**
     * Looks up road/ground elevation at (x, y) in O(1) time.
     * If the cell is empty (e.g. gap between LiDAR rings), searches expanding neighbor cells.
     */
    public getElevation(x: number, y: number, maxSearchRadiusCells: number = 2): number | null {
        if (this.cols === 0 || this.rows === 0) return null;

        const col = Math.floor((x - this.minX) / this.cellSize);
        const row = Math.floor((y - this.minY) / this.cellSize);

        if (col >= 0 && col < this.cols && row >= 0 && row < this.rows) {
            const val = this.grid[row * this.cols + col];
            if (val !== Infinity) return val;
        }

        for (let rad = 1; rad <= maxSearchRadiusCells; rad++) {
            for (let dr = -rad; dr <= rad; dr++) {
                for (let dc = -rad; dc <= rad; dc++) {
                    const r = row + dr;
                    const c = col + dc;
                    if (r >= 0 && r < this.rows && c >= 0 && c < this.cols) {
                        const val = this.grid[r * this.cols + c];
                        if (val !== Infinity) return val;
                    }
                }
            }
        }

        return null;
    }

    /**
     * Estimates local road plane [nA, nB, nC, nD] at (x, y) using heightmap finite differences in O(1).
     * Returns plane equation coefficients [nA, nB, nC, nD] where nA*x + nB*y + nC*z + nD = 0,
     * or null if normal is not sufficiently upward-pointing (nC >= 0.80).
     */
    public getRoadPlane(x: number, y: number): [number, number, number, number] | null {
        const z0 = this.getElevation(x, y);
        if (z0 === null) return null;

        const step = Math.max(this.cellSize, 1.0);
        const zRight = this.getElevation(x + step, y);
        const zLeft = this.getElevation(x - step, y);
        const zTop = this.getElevation(x, y + step);
        const zBottom = this.getElevation(x, y - step);

        let dzDx = 0;
        if (zRight !== null && zLeft !== null) {
            dzDx = (zRight - zLeft) / (2 * step);
        } else if (zRight !== null) {
            dzDx = (zRight - z0) / step;
        } else if (zLeft !== null) {
            dzDx = (z0 - zLeft) / step;
        }

        let dzDy = 0;
        if (zTop !== null && zBottom !== null) {
            dzDy = (zTop - zBottom) / (2 * step);
        } else if (zTop !== null) {
            dzDy = (zTop - z0) / step;
        } else if (zBottom !== null) {
            dzDy = (z0 - zBottom) / step;
        }

        const nx = -dzDx;
        const ny = -dzDy;
        const nz = 1.0;
        const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
        if (len < 1e-6) return null;

        const normX = nx / len;
        const normY = ny / len;
        const normZ = nz / len;

        if (normZ < 0.80) return null;

        const d = -(normX * x + normY * y + normZ * z0);
        return [normX, normY, normZ, d];
    }

    private static readonly scratchPlane = new THREE.Plane();

    /**
     * Intersects a 3D camera ray with the ground elevation heightmap / road plane in O(1) time,
     * eliminating O(N) brute-force raycasting against 150,000 point cloud vertices during Mode.DRAW.
     */
    public intersectRay(
        ray: THREE.Ray,
        fallbackPlane: [number, number, number, number] = [0, 0, 1, 1.65],
        maxDistance: number = 150,
    ): THREE.Vector3 | null {
        let [A, B, C, D] = fallbackPlane;
        const len = Math.sqrt(A * A + B * B + C * C);
        if (len > 1e-6) {
            A /= len;
            B /= len;
            C /= len;
            D /= len;
        } else {
            A = 0;
            B = 0;
            C = 1;
            D = 1.65;
        }

        const plane = GroundElevationGrid.scratchPlane;
        plane.normal.set(A, B, C);
        plane.constant = D;
        const hit = ray.intersectPlane(plane, new THREE.Vector3());

        if (hit && ray.origin.distanceTo(hit) <= maxDistance) {
            for (let iter = 0; iter < 2; iter++) {
                const roadZ = this.getElevation(hit.x, hit.y, 2);
                if (roadZ === null || Math.abs(roadZ - hit.z) > 5.0) break;
                if (Math.abs(ray.direction.z) > 0.03) {
                    const t = (roadZ - ray.origin.z) / ray.direction.z;
                    if (t >= 0.5 && t <= maxDistance) {
                        hit.set(
                            ray.origin.x + ray.direction.x * t,
                            ray.origin.y + ray.direction.y * t,
                            roadZ,
                        );
                    } else {
                        hit.z = roadZ;
                    }
                } else {
                    hit.z = roadZ;
                }
            }
            return hit;
        }

        // Spatial heightmap ray-march fallback when ray is aimed near/above the horizon
        if (this.cols > 0 && this.rows > 0) {
            for (let t = 2.0; t <= 80.0; t += 1.0) {
                const rx = ray.origin.x + ray.direction.x * t;
                const ry = ray.origin.y + ray.direction.y * t;
                const rz = ray.origin.z + ray.direction.z * t;
                const elev = this.getElevation(rx, ry, 1);
                if (elev !== null && rz <= elev + 0.5) {
                    return new THREE.Vector3(rx, ry, elev);
                }
            }
        }

        return null;
    }
}

/**
 * Fits a 3D bounding box cluster around a clicked point.
 * Estimates local ground surface, filters out pavement, and if >= 8 vehicle points exist,
 * computes centroid & 2D PCA orientation aligned to the road plane normal.
 */
export function fitClusterAtPoint(
    clickedPoint: [number, number, number],
    positions: ArrayLike<number>,
    options: {
        labelName?: string;
        label?: any;
        attributes?: any;
        defaultDims?: [number, number, number];
        snapGround?: boolean;
        autoOrient?: boolean;
    } = {},
): { points: number[] } {
    const [cx, cy, cz] = clickedPoint;
    const {
        labelName,
        label,
        attributes,
        defaultDims,
        snapGround = true,
        autoOrient = true,
    } = options;
    const [dx, dy, dz] = defaultDims || getClassDimensionPrior(label || labelName, {
        attributes,
        label,
        defaultDims,
    });
    const isPedestrian = isPedestrianClass(label || labelName);

    if (!positions || positions.length < 9) {
        return {
            points: [
                cx, cy, cz + dz / 2,
                0, 0, 0,
                dx, dy, dz,
                0, 0, 0, 0, 0, 0, 0,
            ],
        };
    }

    const radius = 2.5;
    const detectedPlane = snapGround ? estimateLocalRoadPlane(cx, cy, cz, positions, radius) : null;
    const zGround = detectedPlane ? getRoadHeightAt(detectedPlane, cx, cy) : cz;

    const clusterPoints = extractClusterPoints([cx, cy, cz], positions, {
        radius,
        zGround,
        detectedPlane,
        minHeightAboveGround: 0.15,
        maxHeightAboveGround: Math.max(dz, 1.8) + 0.8,
    });

    const isXLongitudinal = dx >= dy;
    const yawResult = autoOrient ? solveClusterYaw(clusterPoints, isXLongitudinal) : null;

    let finalDx = dx;
    let finalDy = dy;
    let finalDz = dz;

    if (!defaultDims && clusterPoints.length >= 8 && yawResult) {
        const lower = (labelName || '').toLowerCase().trim();
        const hasCustomPrior = lower && (
            CLASS_DIMENSION_PRIORS[lower] ||
            (attributes && Object.keys(attributes).length > 0)
        );
        if (!hasCustomPrior) {
            const cosYaw = Math.cos(yawResult.yaw);
            const sinYaw = Math.sin(yawResult.yaw);
            const uVals: number[] = [];
            const vVals: number[] = [];
            const zVals: number[] = [];
            for (const pt of clusterPoints) {
                const du = (pt.px - yawResult.centroidX) * cosYaw + (pt.py - yawResult.centroidY) * sinYaw;
                const dv = -(pt.px - yawResult.centroidX) * sinYaw + (pt.py - yawResult.centroidY) * cosYaw;
                uVals.push(du);
                vVals.push(dv);
                zVals.push(pt.pz);
            }
            uVals.sort((a, b) => a - b);
            vVals.sort((a, b) => a - b);
            zVals.sort((a, b) => a - b);
            const n = clusterPoints.length;
            const uSpan = uVals[Math.min(n - 1, Math.floor(n * 0.95))] - uVals[Math.floor(n * 0.05)] + 0.15;
            const vSpan = vVals[Math.min(n - 1, Math.floor(n * 0.95))] - vVals[Math.floor(n * 0.05)] + 0.10;
            const zSpan = zVals[Math.min(n - 1, Math.floor(n * 0.98))] - zGround;
            if (uSpan >= 0.5 && uSpan <= 15.0) finalDx = parseFloat(uSpan.toFixed(3));
            if (vSpan >= 0.4 && vSpan <= 4.0) finalDy = parseFloat(vSpan.toFixed(3));
            if (zSpan >= 0.4 && zSpan <= 5.0) finalDz = parseFloat(zSpan.toFixed(3));
        }
    }

    const finalX = yawResult ? yawResult.centroidX : cx;
    const finalY = yawResult ? yawResult.centroidY : cy;
    const yaw = (yawResult && !isPedestrian) ? yawResult.yaw : 0;

    let rx = 0;
    let ry = 0;
    let rz = 0;

    if (isPedestrian || !snapGround || !detectedPlane) {
        rx = 0;
        ry = 0;
        rz = yaw;
    } else {
        const aligned = alignEulerToRoadPlane(yaw, detectedPlane);
        rx = aligned.rx;
        ry = aligned.ry;
        rz = aligned.rz;
    }

    const roadZAtCentroid = (snapGround && detectedPlane) ?
        getRoadHeightAt(detectedPlane, finalX, finalY) : zGround;
    const finalZ = (snapGround ? roadZAtCentroid : cz) + finalDz / 2;

    return {
        points: [
            parseFloat(finalX.toFixed(3)),
            parseFloat(finalY.toFixed(3)),
            parseFloat(finalZ.toFixed(3)),
            rx,
            ry,
            parseFloat(rz.toFixed(4)),
            finalDx,
            finalDy,
            finalDz,
            0, 0, 0, 0, 0, 0, 0,
        ],
    };
}

export interface AnchorFaceConfig {
    anchorX?: 'front' | 'rear' | 'center'; // Longitudinal axis (X): +X = front bumper, -X = rear bumper
    anchorY?: 'left' | 'right' | 'center'; // Lateral axis (Y): +Y = left side, -Y = right side
    anchorZ?: 'bottom' | 'top' | 'center'; // Vertical axis (Z): +Z = roof, -Z = ground bottom
}

/**
 * Resizes a 3D cuboid while pinning a specified face (e.g. rear bumper, ground bottom) in place.
 * Shifts centroid in world space by rotating local anchor delta via object orientation Euler.
 *
 * In CVAT 3D coordinate convention:
 * - Longitudinal axis (X, dx): +X is Front, -X is Rear.
 * - Lateral axis (Y, dy): +Y is Left, -Y is Right.
 * - Vertical axis (Z, dz): +Z is Top, -Z is Bottom.
 */
export function resizeCuboidWithAnchor(
    points: number[],
    newDx: number,
    newDy: number,
    newDz: number,
    anchor: AnchorFaceConfig = { anchorX: 'rear', anchorZ: 'bottom' },
): number[] {
    if (!points || points.length < 9) return points;
    const [x, y, z, rx, ry, rz, oldDx, oldDy, oldDz] = points;
    const deltaLocal = new THREE.Vector3(0, 0, 0);

    // Longitudinal axis (X, dx)
    if (anchor.anchorX === 'rear') deltaLocal.x = (newDx - oldDx) / 2;
    else if (anchor.anchorX === 'front') deltaLocal.x = -(newDx - oldDx) / 2;

    // Lateral axis (Y, dy)
    if (anchor.anchorY === 'right') deltaLocal.y = (newDy - oldDy) / 2;
    else if (anchor.anchorY === 'left') deltaLocal.y = -(newDy - oldDy) / 2;

    // Vertical axis (Z, dz)
    if (anchor.anchorZ === 'bottom') deltaLocal.z = (newDz - oldDz) / 2;
    else if (anchor.anchorZ === 'top') deltaLocal.z = -(newDz - oldDz) / 2;

    const euler = new THREE.Euler(rx, ry, rz, 'XYZ');
    deltaLocal.applyEuler(euler);

    const result = [...points];
    result[0] = parseFloat((x + deltaLocal.x).toFixed(4));
    result[1] = parseFloat((y + deltaLocal.y).toFixed(4));
    result[2] = parseFloat((z + deltaLocal.z).toFixed(4));
    result[6] = parseFloat(newDx.toFixed(4));
    result[7] = parseFloat(newDy.toFixed(4));
    result[8] = parseFloat(newDz.toFixed(4));

    return result;
}

/**
 * Computes a robust global ground plane Ax + By + Cz + D = 0 for the point cloud scene using GPF.
 * Samples points within a central ground ROI, extracts ground seeds, and fits the optimal plane.
 */
export function computeSceneGroundPlane(
    positions: ArrayLike<number>,
    maxRadius: number = 35.0,
): [number, number, number, number] {
    if (!positions || positions.length < 9) {
        return [0, 0, 1, 1.65];
    }

    const totalPoints = Math.floor(positions.length / 3);
    const candidateIndices: number[] = [];
    const r2 = maxRadius * maxRadius;

    // Stride to sample at most ~5,000 points uniformly for fast & accurate fit
    const stride = Math.max(1, Math.floor(totalPoints / 5000));
    for (let i = 0; i < totalPoints; i += stride) {
        const px = positions[i * 3];
        const py = positions[i * 3 + 1];
        const pz = positions[i * 3 + 2];
        if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz)) continue;
        if (px * px + py * py <= r2) {
            candidateIndices.push(i);
        }
    }

    if (candidateIndices.length < 10) {
        for (let i = 0; i < totalPoints; i += stride) {
            const pz = positions[i * 3 + 2];
            if (Number.isFinite(pz)) candidateIndices.push(i);
        }
    }

    return fitGroundPlaneGPF(positions, candidateIndices, {
        lprRatio: 0.2,
        seedThreshold: 0.15,
        distanceThreshold: 0.10,
        refineIterations: 2,
    });
}
