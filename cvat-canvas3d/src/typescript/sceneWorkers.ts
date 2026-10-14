// Copyright (C) 2026 CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import {
    PrecomputedGroundGridData,
    GroundElevationGrid,
    computeSceneGroundPlane,
} from './snapping';

export interface MainSceneInitResult {
    bbox: [number, number, number, number, number, number];
    groundGrid: PrecomputedGroundGridData;
    globalGroundPlane: [number, number, number, number];
}

export interface OrthoROIResult {
    positions: Float32Array;
    colors: Float32Array;
}

const MAIN_SCENE_WORKER_SOURCE = `
function computeSmallestEigenvector3x3(c00, c01, c02, c11, c12, c22) {
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

function fitGroundPlaneGPF(positions, indices) {
    const lprRatio = 0.2;
    const seedThreshold = 0.15;
    const distanceThreshold = 0.10;
    const refineIterations = 2;

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

    const sorted = indices.slice().sort((a, b) => positions[a * 3 + 2] - positions[b * 3 + 2]);
    const lprCount = Math.max(3, Math.floor(sorted.length * lprRatio));
    let lprZSum = 0;
    for (let i = 0; i < lprCount; i++) {
        lprZSum += positions[sorted[i] * 3 + 2];
    }
    const lprZ = lprZSum / lprCount;

    let currentInliers = sorted.filter((idx) => positions[idx * 3 + 2] <= lprZ + seedThreshold);
    if (currentInliers.length < 3) {
        currentInliers = sorted.slice(0, Math.min(sorted.length, 10));
    }

    let bestPlane = [0, 0, 1, -lprZ];
    const totalPasses = 1 + refineIterations;
    for (let pass = 0; pass < totalPasses; pass++) {
        const N = currentInliers.length;
        if (N < 3) break;

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

        let xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
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
            zz / N
        );

        const len = Math.sqrt(normX * normX + normY * normY + normZ * normZ);
        if (len < 1e-6) {
            normX = 0; normY = 0; normZ = 1;
        } else {
            normX /= len; normY /= len; normZ /= len;
        }

        if (normZ < 0) {
            normX = -normX;
            normY = -normY;
            normZ = -normZ;
        }

        if (normZ < 0.85) {
            normX = 0;
            normY = 0;
            normZ = 1;
        }

        const d = -(normX * cx + normY * cy + normZ * cz);
        bestPlane = [normX, normY, normZ, d];

        if (pass < totalPasses - 1) {
            const nextInliers = [];
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

function computeSceneGroundPlaneInWorker(positions, maxRadius) {
    if (!positions || positions.length < 9) {
        return [0, 0, 1, 1.65];
    }
    const totalPoints = Math.floor(positions.length / 3);
    const candidateIndices = [];
    const r2 = maxRadius * maxRadius;
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

    return fitGroundPlaneGPF(positions, candidateIndices);
}

self.onmessage = function (e) {
    const data = e.data;
    if (!data || data.type !== 'initScene') return;

    const id = data.id;
    const positions = data.positions;
    const requestedCellSize = data.cellSize > 0 ? data.cellSize : 0.5;

    if (!positions || positions.length < 9) {
        const emptyGrid = new Float32Array(0);
        self.postMessage({
            type: 'sceneInitialized',
            id,
            bbox: [0, 0, 0, 0, 0, 0],
            groundGrid: {
                cellSize: requestedCellSize,
                minX: 0,
                minY: 0,
                maxX: 0,
                maxY: 0,
                cols: 0,
                rows: 0,
                gridBuffer: emptyGrid.buffer,
            },
            globalGroundPlane: [0, 0, 1, 1.65],
        }, [emptyGrid.buffer]);
        return;
    }

    const totalPoints = Math.floor(positions.length / 3);
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

    for (let i = 0; i < totalPoints; i++) {
        const px = positions[i * 3];
        const py = positions[i * 3 + 1];
        const pz = positions[i * 3 + 2];
        if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz)) continue;
        if (px < minX) minX = px;
        if (px > maxX) maxX = px;
        if (py < minY) minY = py;
        if (py > maxY) maxY = py;
        if (pz < minZ) minZ = pz;
        if (pz > maxZ) maxZ = pz;
    }

    if (!Number.isFinite(minX) || !Number.isFinite(minY)) {
        const emptyGrid = new Float32Array(0);
        self.postMessage({
            type: 'sceneInitialized',
            id,
            bbox: [0, 0, 0, 0, 0, 0],
            groundGrid: {
                cellSize: requestedCellSize,
                minX: 0,
                minY: 0,
                maxX: 0,
                maxY: 0,
                cols: 0,
                rows: 0,
                gridBuffer: emptyGrid.buffer,
            },
            globalGroundPlane: [0, 0, 1, 1.65],
        }, [emptyGrid.buffer]);
        return;
    }

    let effectiveCellSize = requestedCellSize;
    const width = maxX - minX;
    const height = maxY - minY;
    let cols = Math.ceil(width / effectiveCellSize) + 1;
    let rows = Math.ceil(height / effectiveCellSize) + 1;

    if (cols * rows > 1000000) {
        effectiveCellSize = Math.max(width, height) / 1000;
        cols = Math.ceil(width / effectiveCellSize) + 1;
        rows = Math.ceil(height / effectiveCellSize) + 1;
    }

    const grid = new Float32Array(cols * rows);
    grid.fill(Infinity);

    for (let i = 0; i < totalPoints; i++) {
        const px = positions[i * 3];
        const py = positions[i * 3 + 1];
        const pz = positions[i * 3 + 2];
        if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz)) continue;

        const col = Math.floor((px - minX) / effectiveCellSize);
        const row = Math.floor((py - minY) / effectiveCellSize);
        if (col >= 0 && col < cols && row >= 0 && row < rows) {
            const idx = row * cols + col;
            if (pz < grid[idx]) {
                grid[idx] = pz;
            }
        }
    }

    const globalGroundPlane = computeSceneGroundPlaneInWorker(positions, 35.0);

    self.postMessage({
        type: 'sceneInitialized',
        id,
        bbox: [minX, minY, minZ, maxX, maxY, maxZ],
        groundGrid: {
            cellSize: effectiveCellSize,
            minX,
            minY,
            maxX,
            maxY,
            cols,
            rows,
            gridBuffer: grid.buffer,
        },
        globalGroundPlane,
    }, [grid.buffer]);
};
`;

const ORTHO_VIEWS_WORKER_SOURCE = `
let cachedPositions = null;
let cachedColors = null;
let scratchIndices = null;

self.onmessage = function (e) {
    const data = e.data;
    if (!data) return;

    if (data.type === 'initPoints') {
        cachedPositions = data.positions || null;
        cachedColors = data.colors || null;
        if (cachedPositions && cachedPositions.length > 0) {
            const count = Math.floor(cachedPositions.length / 3);
            if (!scratchIndices || scratchIndices.length < count) {
                scratchIndices = new Uint32Array(count);
            }
        }
        return;
    }

    if (data.type === 'clear') {
        cachedPositions = null;
        cachedColors = null;
        return;
    }

    if (data.type === 'computeOrthoROI') {
        const { id, cx, cy, cz, radius } = data;
        if (!cachedPositions || cachedPositions.length === 0) {
            const emptyPos = new Float32Array(0);
            const emptyCol = new Float32Array(0);
            self.postMessage({
                type: 'orthoROIResult',
                id,
                positionsBuffer: emptyPos.buffer,
                colorsBuffer: emptyCol.buffer,
                count: 0,
            }, [emptyPos.buffer, emptyCol.buffer]);
            return;
        }

        const count = Math.floor(cachedPositions.length / 3);
        if (!scratchIndices || scratchIndices.length < count) {
            scratchIndices = new Uint32Array(count);
        }

        const radiusSq = radius * radius;
        let outCount = 0;

        for (let i = 0; i < count; i++) {
            const idx = i * 3;
            const dx = cachedPositions[idx] - cx;
            const dy = cachedPositions[idx + 1] - cy;
            const dz = cachedPositions[idx + 2] - cz;
            if (dx * dx + dy * dy + dz * dz <= radiusSq) {
                scratchIndices[outCount++] = i;
            }
        }

        const roiPositions = new Float32Array(outCount * 3);
        const roiColors = new Float32Array(outCount * 3);
        const hasColors = Boolean(cachedColors && cachedColors.length >= cachedPositions.length);

        for (let j = 0; j < outCount; j++) {
            const srcIdx = scratchIndices[j] * 3;
            const dstIdx = j * 3;
            roiPositions[dstIdx] = cachedPositions[srcIdx];
            roiPositions[dstIdx + 1] = cachedPositions[srcIdx + 1];
            roiPositions[dstIdx + 2] = cachedPositions[srcIdx + 2];

            if (hasColors) {
                roiColors[dstIdx] = cachedColors[srcIdx];
                roiColors[dstIdx + 1] = cachedColors[srcIdx + 1];
                roiColors[dstIdx + 2] = cachedColors[srcIdx + 2];
            } else {
                roiColors[dstIdx] = 1;
                roiColors[dstIdx + 1] = 1;
                roiColors[dstIdx + 2] = 1;
            }
        }

        self.postMessage({
            type: 'orthoROIResult',
            id,
            positionsBuffer: roiPositions.buffer,
            colorsBuffer: roiColors.buffer,
            count: outCount,
        }, [roiPositions.buffer, roiColors.buffer]);
    }
};
`;

function createInlineWorker(source: string): Worker | null {
    if (typeof window === 'undefined' || typeof Worker === 'undefined' || typeof Blob === 'undefined' || typeof URL === 'undefined') {
        return null;
    }
    try {
        const blob = new Blob([source], { type: 'application/javascript' });
        const url = URL.createObjectURL(blob);
        const worker = new Worker(url);
        URL.revokeObjectURL(url);
        return worker;
    } catch {
        return null;
    }
}

/**
 * Worker 1: Dedicated background worker for the main Three.js perspective scene.
 * Offloads spatial ground elevation grid construction, bounding box calculation,
 * and global road plane GPF estimation from the main UI thread.
 */
export class MainSceneWorker {
    private worker: Worker | null;
    private latestRequestId: number = 0;
    private pendingCallback: ((result: MainSceneInitResult) => void) | null = null;

    constructor() {
        this.worker = createInlineWorker(MAIN_SCENE_WORKER_SOURCE);
        if (this.worker) {
            this.worker.onmessage = (e: MessageEvent) => {
                const data = e.data;
                if (!data || data.type !== 'sceneInitialized') return;
                if (data.id !== this.latestRequestId || !this.pendingCallback) return;

                const { bbox, groundGrid, globalGroundPlane } = data;
                const gridArray = new Float32Array(groundGrid.gridBuffer);
                const callback = this.pendingCallback;
                this.pendingCallback = null;

                callback({
                    bbox,
                    groundGrid: {
                        cellSize: groundGrid.cellSize,
                        minX: groundGrid.minX,
                        minY: groundGrid.minY,
                        maxX: groundGrid.maxX,
                        maxY: groundGrid.maxY,
                        cols: groundGrid.cols,
                        rows: groundGrid.rows,
                        grid: gridArray,
                    },
                    globalGroundPlane,
                });
            };
        }
    }

    public initScene(
        positions: ArrayLike<number>,
        onResult: (result: MainSceneInitResult) => void,
        cellSize: number = 0.5,
    ): void {
        const id = ++this.latestRequestId;
        this.pendingCallback = onResult;

        if (this.worker) {
            const copy = new Float32Array(positions.length);
            copy.set(positions);
            this.worker.postMessage(
                {
                    type: 'initScene',
                    id,
                    positions: copy,
                    cellSize,
                },
                [copy.buffer],
            );
            return;
        }

        // Synchronous fallback when Web Worker is unavailable
        const gridInstance = new GroundElevationGrid(positions, cellSize);
        const globalGroundPlane = computeSceneGroundPlane(positions, 35.0);
        if (id === this.latestRequestId && this.pendingCallback) {
            const callback = this.pendingCallback;
            this.pendingCallback = null;
            callback({
                bbox: [gridInstance.minX, gridInstance.minY, 0, gridInstance.maxX, gridInstance.maxY, 0],
                groundGrid: {
                    cellSize: gridInstance.cellSize,
                    minX: gridInstance.minX,
                    minY: gridInstance.minY,
                    maxX: gridInstance.maxX,
                    maxY: gridInstance.maxY,
                    cols: gridInstance.cols,
                    rows: gridInstance.rows,
                    grid: (gridInstance as any).grid,
                },
                globalGroundPlane,
            });
        }
    }

    public clear(): void {
        this.latestRequestId++;
        this.pendingCallback = null;
    }

    public dispose(): void {
        this.clear();
        if (this.worker) {
            this.worker.terminate();
            this.worker = null;
        }
    }
}

/**
 * Worker 2: Dedicated background worker for the top, side, and front orthographic views.
 * Caches point cloud coordinates & colors and computes compact spherical ROI vertex buffers
 * off the main thread, returning Float32Array position and color buffers via zero-copy Transferables.
 */
export class OrthoViewsWorker {
    private worker: Worker | null;
    private latestRequestId: number = 0;
    private pendingCallback: ((result: OrthoROIResult) => void) | null = null;
    private fallbackPositions: Float32Array | null = null;
    private fallbackColors: Float32Array | null = null;

    constructor() {
        this.worker = createInlineWorker(ORTHO_VIEWS_WORKER_SOURCE);
        if (this.worker) {
            this.worker.onmessage = (e: MessageEvent) => {
                const data = e.data;
                if (!data || data.type !== 'orthoROIResult') return;
                if (data.id !== this.latestRequestId || !this.pendingCallback) return;

                const positions = new Float32Array(data.positionsBuffer, 0, data.count * 3);
                const colors = new Float32Array(data.colorsBuffer, 0, data.count * 3);
                const callback = this.pendingCallback;
                this.pendingCallback = null;
                callback({ positions, colors });
            };
        }
    }

    public initPoints(positions: ArrayLike<number>, colors?: ArrayLike<number> | null): void {
        this.latestRequestId++;
        this.pendingCallback = null;

        if (this.worker) {
            const posCopy = new Float32Array(positions.length);
            posCopy.set(positions);
            const transferList: Transferable[] = [posCopy.buffer];
            let colCopy: Float32Array | null = null;
            if (colors && colors.length >= positions.length) {
                colCopy = new Float32Array(colors.length);
                colCopy.set(colors);
                transferList.push(colCopy.buffer);
            }
            this.worker.postMessage(
                {
                    type: 'initPoints',
                    positions: posCopy,
                    colors: colCopy,
                },
                transferList,
            );
            return;
        }

        const posCopy = new Float32Array(positions.length);
        posCopy.set(positions);
        this.fallbackPositions = posCopy;
        if (colors && colors.length >= positions.length) {
            const colCopy = new Float32Array(colors.length);
            colCopy.set(colors);
            this.fallbackColors = colCopy;
        } else {
            this.fallbackColors = null;
        }
    }

    public computeROI(
        cx: number,
        cy: number,
        cz: number,
        radius: number,
        onResult: (result: OrthoROIResult) => void,
    ): void {
        const id = ++this.latestRequestId;
        this.pendingCallback = onResult;

        if (this.worker) {
            this.worker.postMessage({
                type: 'computeOrthoROI',
                id,
                cx,
                cy,
                cz,
                radius,
            });
            return;
        }

        // Synchronous fallback when Web Worker is unavailable
        const positions = this.fallbackPositions;
        const colors = this.fallbackColors;
        if (!positions || positions.length === 0) {
            onResult({
                positions: new Float32Array(0),
                colors: new Float32Array(0),
            });
            return;
        }
        const count = Math.floor(positions.length / 3);
        const radiusSq = radius * radius;
        const indices: number[] = [];
        for (let i = 0; i < count; i++) {
            const idx = i * 3;
            const dx = positions[idx] - cx;
            const dy = positions[idx + 1] - cy;
            const dz = positions[idx + 2] - cz;
            if (dx * dx + dy * dy + dz * dz <= radiusSq) {
                indices.push(i);
            }
        }
        const outCount = indices.length;
        const roiPositions = new Float32Array(outCount * 3);
        const roiColors = new Float32Array(outCount * 3);
        const hasColors = Boolean(colors && colors.length >= positions.length);
        for (let j = 0; j < outCount; j++) {
            const srcIdx = indices[j] * 3;
            const dstIdx = j * 3;
            roiPositions[dstIdx] = positions[srcIdx];
            roiPositions[dstIdx + 1] = positions[srcIdx + 1];
            roiPositions[dstIdx + 2] = positions[srcIdx + 2];
            if (hasColors && colors) {
                roiColors[dstIdx] = colors[srcIdx];
                roiColors[dstIdx + 1] = colors[srcIdx + 1];
                roiColors[dstIdx + 2] = colors[srcIdx + 2];
            } else {
                roiColors[dstIdx] = 1;
                roiColors[dstIdx + 1] = 1;
                roiColors[dstIdx + 2] = 1;
            }
        }
        if (id === this.latestRequestId && this.pendingCallback) {
            this.pendingCallback = null;
            onResult({ positions: roiPositions, colors: roiColors });
        }
    }

    public clear(): void {
        this.latestRequestId++;
        this.pendingCallback = null;
        this.fallbackPositions = null;
        this.fallbackColors = null;
        if (this.worker) {
            this.worker.postMessage({ type: 'clear' });
        }
    }

    public dispose(): void {
        this.clear();
        if (this.worker) {
            this.worker.terminate();
            this.worker = null;
        }
    }
}
