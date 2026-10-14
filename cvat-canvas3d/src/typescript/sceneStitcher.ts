// Copyright (C) 2026 CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import * as THREE from 'three';

export interface FramePointData {
    frameNumber: number;
    positions: Float32Array;
    colors?: Float32Array | null;
    transform?: number[]; // 4x4 relative matrix to current frame if available
}

export interface StitchingOptions {
    voxelSize?: number; // Voxel downsampling size in meters, default 0.05 (5cm)
    enableSOR?: boolean; // Statistical Outlier Removal
    sorNeighborRadius?: number; // Radius to check neighbor density (meters)
    sorMinNeighbors?: number; // Minimum neighbor count to retain point
}

function nextPowerOfTwo(n: number): number {
    let p = 1024;
    while (p < n) {
        p *= 2;
    }
    return p;
}

/**
 * Fast Voxel Grid Centroid Downsampling & Smoothing in TypeScript/WebGL
 * Groups points falling into the same 3D voxel cell and computes their arithmetic centroid.
 * Flattens planar road surfaces, removes duplicate sensor hits, and bounds memory for 60 FPS Three.js rendering.
 * Uses a flat TypedArray open-addressed spatial hash table with zero string or object allocations.
 */
export function voxelDownsampleAndSmooth(
    positions: Float32Array,
    colors: Float32Array | null,
    voxelSize: number = 0.05,
): { positions: Float32Array; colors: Float32Array | null } {
    if (!positions || positions.length < 9 || voxelSize <= 0) {
        return { positions, colors };
    }

    const numPoints = Math.floor(positions.length / 3);
    const hasColors = Boolean(colors && colors.length >= positions.length);

    // Find bounding box for spatial index normalization
    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    for (let i = 0; i < positions.length; i += 3) {
        const x = positions[i];
        const y = positions[i + 1];
        const z = positions[i + 2];
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (z < minZ) minZ = z;
    }

    if (!Number.isFinite(minX)) {
        return { positions, colors };
    }

    const invV = 1.0 / voxelSize;
    const capacity = nextPowerOfTwo(numPoints * 2);
    const mask = capacity - 1;

    const slotMap = new Int32Array(capacity);
    slotMap.fill(-1);

    const voxelVx = new Int32Array(numPoints);
    const voxelVy = new Int32Array(numPoints);
    const voxelVz = new Int32Array(numPoints);
    const counts = new Uint32Array(numPoints);
    const sumX = new Float64Array(numPoints);
    const sumY = new Float64Array(numPoints);
    const sumZ = new Float64Array(numPoints);
    const sumR = hasColors ? new Float32Array(numPoints) : null;
    const sumG = hasColors ? new Float32Array(numPoints) : null;
    const sumB = hasColors ? new Float32Array(numPoints) : null;

    let uniqueCount = 0;

    for (let i = 0; i < numPoints; i++) {
        const i3 = i * 3;
        const x = positions[i3];
        const y = positions[i3 + 1];
        const z = positions[i3 + 2];
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;

        const vx = Math.floor((x - minX) * invV) | 0;
        const vy = Math.floor((y - minY) * invV) | 0;
        const vz = Math.floor((z - minZ) * invV) | 0;

        let slot = ((vx * 73856093) ^ (vy * 19349663) ^ (vz * 83492791)) & mask;
        let vIdx = slotMap[slot];
        while (vIdx !== -1) {
            if (voxelVx[vIdx] === vx && voxelVy[vIdx] === vy && voxelVz[vIdx] === vz) {
                break;
            }
            slot = (slot + 1) & mask;
            vIdx = slotMap[slot];
        }

        if (vIdx === -1) {
            vIdx = uniqueCount++;
            slotMap[slot] = vIdx;
            voxelVx[vIdx] = vx;
            voxelVy[vIdx] = vy;
            voxelVz[vIdx] = vz;
        }

        counts[vIdx]++;
        sumX[vIdx] += x;
        sumY[vIdx] += y;
        sumZ[vIdx] += z;

        if (hasColors && colors && sumR && sumG && sumB) {
            sumR[vIdx] += colors[i3];
            sumG[vIdx] += colors[i3 + 1];
            sumB[vIdx] += colors[i3 + 2];
        }
    }

    const outPositions = new Float32Array(uniqueCount * 3);
    const outColors = hasColors ? new Float32Array(uniqueCount * 3) : null;

    for (let idx = 0; idx < uniqueCount; idx++) {
        const invC = 1.0 / counts[idx];
        const i3 = idx * 3;
        outPositions[i3] = sumX[idx] * invC;
        outPositions[i3 + 1] = sumY[idx] * invC;
        outPositions[i3 + 2] = sumZ[idx] * invC;

        if (outColors && sumR && sumG && sumB) {
            outColors[i3] = sumR[idx] * invC;
            outColors[i3 + 1] = sumG[idx] * invC;
            outColors[i3 + 2] = sumB[idx] * invC;
        }
    }

    return { positions: outPositions, colors: outColors };
}

/**
 * Fast Statistical Outlier Removal (SOR) / Density Filter
 * Removes isolated noise points that have fewer than minNeighbors within neighborRadius.
 * Uses a flat TypedArray open-addressed spatial hash grid and intrusive index linked list.
 */
export function filterStatisticalOutliers(
    positions: Float32Array,
    colors: Float32Array | null,
    neighborRadius: number = 0.35,
    minNeighbors: number = 3,
): { positions: Float32Array; colors: Float32Array | null } {
    if (!positions || positions.length < 30 || neighborRadius <= 0) {
        return { positions, colors };
    }

    const numPoints = Math.floor(positions.length / 3);
    const hasColors = Boolean(colors && colors.length >= positions.length);
    const invCell = 1.0 / neighborRadius;

    const capacity = nextPowerOfTwo(numPoints * 2);
    const mask = capacity - 1;

    const cellCx = new Int32Array(capacity);
    const cellCy = new Int32Array(capacity);
    const cellCz = new Int32Array(capacity);
    const cellHead = new Int32Array(capacity);
    cellHead.fill(-1);

    const pointNext = new Int32Array(numPoints);
    const pointCellSlot = new Int32Array(numPoints);
    const pointCx = new Int32Array(numPoints);
    const pointCy = new Int32Array(numPoints);
    const pointCz = new Int32Array(numPoints);

    for (let i = 0; i < numPoints; i++) {
        const i3 = i * 3;
        const x = positions[i3];
        const y = positions[i3 + 1];
        const z = positions[i3 + 2];
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
            pointCellSlot[i] = -1;
            pointNext[i] = -1;
            continue;
        }

        const cx = Math.floor(x * invCell) | 0;
        const cy = Math.floor(y * invCell) | 0;
        const cz = Math.floor(z * invCell) | 0;
        pointCx[i] = cx;
        pointCy[i] = cy;
        pointCz[i] = cz;

        let slot = ((cx * 73856093) ^ (cy * 19349663) ^ (cz * 83492791)) & mask;
        let head = cellHead[slot];
        while (head !== -1) {
            if (cellCx[slot] === cx && cellCy[slot] === cy && cellCz[slot] === cz) {
                break;
            }
            slot = (slot + 1) & mask;
            head = cellHead[slot];
        }

        if (head === -1) {
            cellCx[slot] = cx;
            cellCy[slot] = cy;
            cellCz[slot] = cz;
        }

        pointNext[i] = head;
        cellHead[slot] = i;
        pointCellSlot[i] = slot;
    }

    const validIndices = new Int32Array(numPoints);
    let validCount = 0;
    const r2 = neighborRadius * neighborRadius;

    for (let i = 0; i < numPoints; i++) {
        const selfSlot = pointCellSlot[i];
        if (selfSlot === -1) continue;

        const i3 = i * 3;
        const x = positions[i3];
        const y = positions[i3 + 1];
        const z = positions[i3 + 2];

        let neighborCount = 0;

        // 1. Check points in the point's own cell first (O(1) slot access, >95% early exit)
        let pIdx = cellHead[selfSlot];
        while (pIdx !== -1) {
            if (pIdx !== i) {
                const p3 = pIdx * 3;
                const dX = positions[p3] - x;
                const dY = positions[p3 + 1] - y;
                const dZ = positions[p3 + 2] - z;
                if (dX * dX + dY * dY + dZ * dZ <= r2) {
                    neighborCount++;
                    if (neighborCount >= minNeighbors) break;
                }
            }
            pIdx = pointNext[pIdx];
        }

        // 2. Only if needed, search the 26 adjacent cells in the 3x3x3 neighborhood
        if (neighborCount < minNeighbors) {
            const cx = pointCx[i];
            const cy = pointCy[i];
            const cz = pointCz[i];

            for (let dx = -1; dx <= 1; dx++) {
                const tx = (cx + dx) | 0;
                const hX = tx * 73856093;
                for (let dy = -1; dy <= 1; dy++) {
                    const ty = (cy + dy) | 0;
                    const hXY = hX ^ (ty * 19349663);
                    for (let dz = -1; dz <= 1; dz++) {
                        if (dx === 0 && dy === 0 && dz === 0) continue;
                        const tz = (cz + dz) | 0;
                        let slot = (hXY ^ (tz * 83492791)) & mask;
                        let cellPt = cellHead[slot];
                        while (cellPt !== -1) {
                            if (cellCx[slot] === tx && cellCy[slot] === ty && cellCz[slot] === tz) {
                                while (cellPt !== -1) {
                                    const p3 = cellPt * 3;
                                    const dX = positions[p3] - x;
                                    const dY = positions[p3 + 1] - y;
                                    const dZ = positions[p3 + 2] - z;
                                    if (dX * dX + dY * dY + dZ * dZ <= r2) {
                                        neighborCount++;
                                        if (neighborCount >= minNeighbors) break;
                                    }
                                    cellPt = pointNext[cellPt];
                                }
                                break;
                            }
                            slot = (slot + 1) & mask;
                            cellPt = cellHead[slot];
                        }
                        if (neighborCount >= minNeighbors) break;
                    }
                    if (neighborCount >= minNeighbors) break;
                }
                if (neighborCount >= minNeighbors) break;
            }
        }

        if (neighborCount >= minNeighbors) {
            validIndices[validCount++] = i;
        }
    }

    const outPositions = new Float32Array(validCount * 3);
    const outColors = hasColors ? new Float32Array(validCount * 3) : null;

    for (let idx = 0; idx < validCount; idx++) {
        const src3 = validIndices[idx] * 3;
        const dst3 = idx * 3;
        outPositions[dst3] = positions[src3];
        outPositions[dst3 + 1] = positions[src3 + 1];
        outPositions[dst3 + 2] = positions[src3 + 2];

        if (outColors && colors) {
            outColors[dst3] = colors[src3];
            outColors[dst3 + 1] = colors[src3 + 1];
            outColors[dst3 + 2] = colors[src3 + 2];
        }
    }

    return { positions: outPositions, colors: outColors };
}

/**
 * Apply 4x4 matrix transformation to positions into an output buffer at dstOffset
 */
function transformFloat32PointsInto(
    positions: Float32Array,
    matrix4: number[],
    out: Float32Array,
    dstOffset: number,
): void {
    const m = matrix4;
    const m00 = m[0];
    const m10 = m[1];
    const m20 = m[2];
    const m01 = m[4];
    const m11 = m[5];
    const m21 = m[6];
    const m02 = m[8];
    const m12 = m[9];
    const m22 = m[10];
    const m03 = m[12];
    const m13 = m[13];
    const m23 = m[14];

    for (let i = 0; i < positions.length; i += 3) {
        const x = positions[i];
        const y = positions[i + 1];
        const z = positions[i + 2];
        const dst = dstOffset + i;

        out[dst] = m00 * x + m01 * y + m02 * z + m03;
        out[dst + 1] = m10 * x + m11 * y + m12 * z + m13;
        out[dst + 2] = m20 * x + m21 * y + m22 * z + m23;
    }
}

/**
 * Apply 4x4 matrix transformation to positions
 */
export function transformFloat32Points(positions: Float32Array, matrix4: number[]): Float32Array {
    const out = new Float32Array(positions.length);
    transformFloat32PointsInto(positions, matrix4, out, 0);
    return out;
}

/**
 * Stitches and smooths multiple frames into a unified high-density point cloud
 */
export function stitchAndSmoothPointClouds(
    currentFrame: { positions: Float32Array; colors?: Float32Array | null },
    neighborFrames: FramePointData[],
    options: StitchingOptions = {},
): THREE.BufferGeometry {
    const voxelSize = options.voxelSize ?? 0.05;
    const enableSOR = options.enableSOR ?? true;

    // 1. Compute total points and allocate unified merged buffer directly
    let totalPoints = Math.floor(currentFrame.positions.length / 3);

    for (const neighbor of neighborFrames) {
        totalPoints += Math.floor(neighbor.positions.length / 3);
    }

    const mergedPositions = new Float32Array(totalPoints * 3);
    const mergedColors = new Float32Array(totalPoints * 3);

    let offset = 0;
    const currentLen = Math.floor(currentFrame.positions.length / 3) * 3;
    mergedPositions.set(currentFrame.positions.subarray(0, currentLen), offset);
    if (currentFrame.colors && currentFrame.colors.length >= currentLen) {
        mergedColors.set(currentFrame.colors.subarray(0, currentLen), offset);
    } else {
        mergedColors.fill(1.0, offset, offset + currentLen);
    }
    offset += currentLen;

    for (const neighbor of neighborFrames) {
        const len = Math.floor(neighbor.positions.length / 3) * 3;
        const posSub = neighbor.positions.subarray(0, len);
        if (neighbor.transform && neighbor.transform.length === 16) {
            transformFloat32PointsInto(posSub, neighbor.transform, mergedPositions, offset);
        } else {
            mergedPositions.set(posSub, offset);
        }
        if (neighbor.colors && neighbor.colors.length >= len) {
            mergedColors.set(neighbor.colors.subarray(0, len), offset);
        } else {
            mergedColors.fill(1.0, offset, offset + len);
        }
        offset += len;
    }

    // 2. Multi-level Smoothing: Voxel Grid Centroid Downsampling
    const voxelResult = voxelDownsampleAndSmooth(mergedPositions, mergedColors, voxelSize);

    // 3. Multi-level Smoothing: Statistical Outlier Removal
    let finalPositions = voxelResult.positions;
    let finalColors = voxelResult.colors;

    if (enableSOR) {
        const sorResult = filterStatisticalOutliers(
            finalPositions,
            finalColors,
            options.sorNeighborRadius ?? 0.35,
            options.sorMinNeighbors ?? 3,
        );
        finalPositions = sorResult.positions;
        finalColors = sorResult.colors;
    }

    // 4. Construct THREE.BufferGeometry
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(finalPositions, 3));
    if (finalColors) {
        geo.setAttribute('color', new THREE.BufferAttribute(finalColors, 3));
    }
    geo.computeBoundingBox();
    geo.computeBoundingSphere();

    return geo;
}
