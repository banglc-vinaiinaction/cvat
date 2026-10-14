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

/**
 * Fast Voxel Grid Centroid Downsampling & Smoothing in TypeScript/WebGL
 * Groups points falling into the same 3D voxel cell and computes their arithmetic centroid.
 * Flattens planar road surfaces, removes duplicate sensor hits, and bounds memory for 60 FPS Three.js rendering.
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
        if (positions[i] < minX) minX = positions[i];
        if (positions[i + 1] < minY) minY = positions[i + 1];
        if (positions[i + 2] < minZ) minZ = positions[i + 2];
    }

    const invV = 1.0 / voxelSize;
    // Map voxel string key -> accumulator
    // Key format: `${vx}_${vy}_${vz}`
    const voxelMap = new Map<string, {
        count: number;
        sumX: number;
        sumY: number;
        sumZ: number;
        sumR: number;
        sumG: number;
        sumB: number;
    }>();

    for (let i = 0; i < numPoints; i++) {
        const i3 = i * 3;
        const x = positions[i3];
        const y = positions[i3 + 1];
        const z = positions[i3 + 2];

        const vx = Math.floor((x - minX) * invV);
        const vy = Math.floor((y - minY) * invV);
        const vz = Math.floor((z - minZ) * invV);
        const key = `${vx}_${vy}_${vz}`;

        let entry = voxelMap.get(key);
        if (!entry) {
            entry = {
                count: 0,
                sumX: 0,
                sumY: 0,
                sumZ: 0,
                sumR: 0,
                sumG: 0,
                sumB: 0,
            };
            voxelMap.set(key, entry);
        }

        entry.count++;
        entry.sumX += x;
        entry.sumY += y;
        entry.sumZ += z;

        if (hasColors && colors) {
            entry.sumR += colors[i3];
            entry.sumG += colors[i3 + 1];
            entry.sumB += colors[i3 + 2];
        }
    }

    const outCount = voxelMap.size;
    const outPositions = new Float32Array(outCount * 3);
    const outColors = hasColors ? new Float32Array(outCount * 3) : null;

    let idx = 0;
    for (const entry of voxelMap.values()) {
        const invC = 1.0 / entry.count;
        const i3 = idx * 3;
        outPositions[i3] = entry.sumX * invC;
        outPositions[i3 + 1] = entry.sumY * invC;
        outPositions[i3 + 2] = entry.sumZ * invC;

        if (outColors) {
            outColors[i3] = entry.sumR * invC;
            outColors[i3 + 1] = entry.sumG * invC;
            outColors[i3 + 2] = entry.sumB * invC;
        }
        idx++;
    }

    return { positions: outPositions, colors: outColors };
}

/**
 * Fast Statistical Outlier Removal (SOR) / Density Filter
 * Removes isolated noise points that have fewer than minNeighbors within neighborRadius.
 */
export function filterStatisticalOutliers(
    positions: Float32Array,
    colors: Float32Array | null,
    neighborRadius: number = 0.35,
    minNeighbors: number = 3,
): { positions: Float32Array; colors: Float32Array | null } {
    if (!positions || positions.length < 30) {
        return { positions, colors };
    }

    const numPoints = Math.floor(positions.length / 3);
    const hasColors = Boolean(colors && colors.length >= positions.length);
    const cellSize = neighborRadius;
    const invCell = 1.0 / cellSize;

    // Coarse spatial hash
    const grid = new Map<string, number[]>();
    for (let i = 0; i < numPoints; i++) {
        const i3 = i * 3;
        const cx = Math.floor(positions[i3] * invCell);
        const cy = Math.floor(positions[i3 + 1] * invCell);
        const cz = Math.floor(positions[i3 + 2] * invCell);
        const key = `${cx}_${cy}_${cz}`;

        let list = grid.get(key);
        if (!list) {
            list = [];
            grid.set(key, list);
        }
        list.push(i);
    }

    const validIndices: number[] = [];
    const r2 = neighborRadius * neighborRadius;

    for (let i = 0; i < numPoints; i++) {
        const i3 = i * 3;
        const x = positions[i3];
        const y = positions[i3 + 1];
        const z = positions[i3 + 2];

        const cx = Math.floor(x * invCell);
        const cy = Math.floor(y * invCell);
        const cz = Math.floor(z * invCell);

        let neighborCount = 0;

        // Search adjacent 3x3x3 cells
        for (let dx = -1; dx <= 1; dx++) {
            for (let dy = -1; dy <= 1; dy++) {
                for (let dz = -1; dz <= 1; dz++) {
                    const cellKey = `${cx + dx}_${cy + dy}_${cz + dz}`;
                    const cellPoints = grid.get(cellKey);
                    if (cellPoints) {
                        for (const pIdx of cellPoints) {
                            if (pIdx === i) continue;
                            const p3 = pIdx * 3;
                            const dX = positions[p3] - x;
                            const dY = positions[p3 + 1] - y;
                            const dZ = positions[p3 + 2] - z;
                            if (dX * dX + dY * dY + dZ * dZ <= r2) {
                                neighborCount++;
                                if (neighborCount >= minNeighbors) break;
                            }
                        }
                    }
                    if (neighborCount >= minNeighbors) break;
                }
                if (neighborCount >= minNeighbors) break;
            }
            if (neighborCount >= minNeighbors) break;
        }

        if (neighborCount >= minNeighbors) {
            validIndices.push(i);
        }
    }

    const outCount = validIndices.length;
    const outPositions = new Float32Array(outCount * 3);
    const outColors = hasColors ? new Float32Array(outCount * 3) : null;

    for (let idx = 0; idx < outCount; idx++) {
        const srcI = validIndices[idx];
        const src3 = srcI * 3;
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
 * Apply 4x4 matrix transformation to positions in-place
 */
export function transformFloat32Points(positions: Float32Array, matrix4: number[]): Float32Array {
    const out = new Float32Array(positions.length);
    const m = matrix4;
    // Row-major or column-major support: Three.js Matrix4 elements are column-major
    const isColumnMajor = true;
    const m00 = isColumnMajor ? m[0] : m[0];
    const m10 = isColumnMajor ? m[1] : m[4];
    const m20 = isColumnMajor ? m[2] : m[8];
    const m01 = isColumnMajor ? m[4] : m[1];
    const m11 = isColumnMajor ? m[5] : m[5];
    const m21 = isColumnMajor ? m[6] : m[9];
    const m02 = isColumnMajor ? m[8] : m[2];
    const m12 = isColumnMajor ? m[9] : m[6];
    const m22 = isColumnMajor ? m[10] : m[10];
    const m03 = isColumnMajor ? m[12] : m[3];
    const m13 = isColumnMajor ? m[13] : m[7];
    const m23 = isColumnMajor ? m[14] : m[11];

    for (let i = 0; i < positions.length; i += 3) {
        const x = positions[i];
        const y = positions[i + 1];
        const z = positions[i + 2];

        out[i] = m00 * x + m01 * y + m02 * z + m03;
        out[i + 1] = m10 * x + m11 * y + m12 * z + m13;
        out[i + 2] = m20 * x + m21 * y + m22 * z + m23;
    }
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

    // 1. Gather all points
    const positionChunks: Float32Array[] = [currentFrame.positions];
    const colorChunks: Float32Array[] = currentFrame.colors ? [currentFrame.colors] : [];
    let totalPoints = Math.floor(currentFrame.positions.length / 3);

    for (const neighbor of neighborFrames) {
        let transformedPositions = neighbor.positions;
        if (neighbor.transform && neighbor.transform.length === 16) {
            transformedPositions = transformFloat32Points(neighbor.positions, neighbor.transform);
        }
        positionChunks.push(transformedPositions);
        if (neighbor.colors) {
            colorChunks.push(neighbor.colors);
        }
        totalPoints += Math.floor(neighbor.positions.length / 3);
    }

    // Allocate unified merged buffer
    const mergedPositions = new Float32Array(totalPoints * 3);
    const hasColors = colorChunks.length === positionChunks.length;
    const mergedColors = hasColors ? new Float32Array(totalPoints * 3) : null;

    let offset = 0;
    for (let c = 0; c < positionChunks.length; c++) {
        mergedPositions.set(positionChunks[c], offset);
        if (mergedColors && colorChunks[c]) {
            mergedColors.set(colorChunks[c], offset);
        }
        offset += positionChunks[c].length;
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
