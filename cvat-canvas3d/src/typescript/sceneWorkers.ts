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

export interface MainSceneParseResult extends MainSceneInitResult {
    positions: Float32Array;
    colors: Float32Array | null;
    normals: Float32Array | null;
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

function srgbToLinear(c) {
    return (c < 0.04045) ? c * 0.0773993808 : Math.pow(c * 0.9478672986 + 0.0521327014, 2.4);
}

function decompressLZF(inData, outLength) {
    const inLength = inData.length;
    const outData = new Uint8Array(outLength);
    let inPtr = 0;
    let outPtr = 0;
    let ctrl;
    let len;
    let ref;
    do {
        ctrl = inData[inPtr++];
        if (ctrl < (1 << 5)) {
            ctrl++;
            if (outPtr + ctrl > outLength) throw new Error('Output buffer is not large enough');
            if (inPtr + ctrl > inLength) throw new Error('Invalid compressed data');
            do {
                outData[outPtr++] = inData[inPtr++];
            } while (--ctrl);
        } else {
            len = ctrl >> 5;
            ref = outPtr - ((ctrl & 0x1f) << 8) - 1;
            if (inPtr >= inLength) throw new Error('Invalid compressed data');
            if (len === 7) {
                len += inData[inPtr++];
                if (inPtr >= inLength) throw new Error('Invalid compressed data');
            }
            ref -= inData[inPtr++];
            if (outPtr + len + 2 > outLength) throw new Error('Output buffer is not large enough');
            if (ref < 0 || ref >= outPtr) throw new Error('Invalid compressed data');
            do {
                outData[outPtr++] = outData[ref++];
            } while (--len + 2);
        }
    } while (inPtr < inLength);
    return outData;
}

function getDataViewValue(dataview, offset, type, size) {
    switch (type) {
        case 'F':
            return size === 8 ? dataview.getFloat64(offset, true) : dataview.getFloat32(offset, true);
        case 'I':
            if (size === 1) return dataview.getInt8(offset);
            if (size === 2) return dataview.getInt16(offset, true);
            return dataview.getInt32(offset, true);
        case 'U':
            if (size === 1) return dataview.getUint8(offset);
            if (size === 2) return dataview.getUint16(offset, true);
            return dataview.getUint32(offset, true);
        default:
            return dataview.getFloat32(offset, true);
    }
}

function parsePCDHeader(binaryData) {
    const PCDheader = {};
    const buffer = new Uint8Array(binaryData);
    let data = '', line = '', i = 0, end = false;
    const max = buffer.length;

    while (i < max && end === false) {
        const char = String.fromCharCode(buffer[i++]);
        if (char === '\\n' || char === '\\r') {
            if (line.trim().toLowerCase().startsWith('data')) {
                end = true;
            }
            line = '';
        } else {
            line += char;
        }
        data += char;
    }

    const result1 = data.search(/[\\r\\n]DATA\\s(\\S*)\\s/i);
    const result2 = /[\\r\\n]DATA\\s(\\S*)\\s/i.exec(data.slice(result1 - 1));

    PCDheader.data = result2[1];
    PCDheader.headerLen = result2[0].length + result1;
    PCDheader.str = data.slice(0, PCDheader.headerLen).replace(/#.*/gi, '');

    PCDheader.version = /^VERSION (.*)/im.exec(PCDheader.str);
    PCDheader.fields = /^FIELDS (.*)/im.exec(PCDheader.str);
    PCDheader.size = /^SIZE (.*)/im.exec(PCDheader.str);
    PCDheader.type = /^TYPE (.*)/im.exec(PCDheader.str);
    PCDheader.count = /^COUNT (.*)/im.exec(PCDheader.str);
    PCDheader.width = /^WIDTH (.*)/im.exec(PCDheader.str);
    PCDheader.height = /^HEIGHT (.*)/im.exec(PCDheader.str);
    PCDheader.viewpoint = /^VIEWPOINT (.*)/im.exec(PCDheader.str);
    PCDheader.points = /^POINTS (.*)/im.exec(PCDheader.str);

    if (PCDheader.version !== null) PCDheader.version = parseFloat(PCDheader.version[1]);
    PCDheader.fields = (PCDheader.fields !== null) ? PCDheader.fields[1].split(' ') : [];
    if (PCDheader.type !== null) PCDheader.type = PCDheader.type[1].split(' ');
    if (PCDheader.width !== null) PCDheader.width = parseInt(PCDheader.width[1], 10);
    if (PCDheader.height !== null) PCDheader.height = parseInt(PCDheader.height[1], 10);
    if (PCDheader.viewpoint !== null) PCDheader.viewpoint = PCDheader.viewpoint[1];
    if (PCDheader.points !== null) PCDheader.points = parseInt(PCDheader.points[1], 10);
    if (PCDheader.points === null) PCDheader.points = PCDheader.width * PCDheader.height;

    if (PCDheader.size !== null) {
        PCDheader.size = PCDheader.size[1].split(' ').map((x) => parseInt(x, 10));
    }
    if (PCDheader.count !== null) {
        PCDheader.count = PCDheader.count[1].split(' ').map((x) => parseInt(x, 10));
    } else {
        PCDheader.count = [];
        for (let j = 0, l = PCDheader.fields.length; j < l; j++) {
            PCDheader.count.push(1);
        }
    }

    PCDheader.offset = {};
    let sizeSum = 0;
    for (let j = 0, l = PCDheader.fields.length; j < l; j++) {
        if (PCDheader.data === 'ascii') {
            PCDheader.offset[PCDheader.fields[j]] = j;
        } else {
            PCDheader.offset[PCDheader.fields[j]] = sizeSum;
            sizeSum += PCDheader.size[j] * PCDheader.count[j];
        }
    }
    PCDheader.rowSize = sizeSum;
    return PCDheader;
}

function parsePCDInWorker(buffer) {
    const PCDheader = parsePCDHeader(buffer);
    const offset = PCDheader.offset;
    const hasRGB = offset.rgb !== undefined;
    const hasNormal = offset.normal_x !== undefined;

    let maxPoints = PCDheader.points || 0;
    let asciiLines = null;
    if (PCDheader.data === 'ascii') {
        const textData = new TextDecoder().decode(buffer);
        const pcdData = textData.slice(PCDheader.headerLen);
        asciiLines = pcdData.split('\\n');
        maxPoints = asciiLines.length;
    }

    const rawPositions = new Float32Array(maxPoints * 3);
    const rawColors = hasRGB ? new Float32Array(maxPoints * 3) : null;
    const rawNormals = hasNormal ? new Float32Array(maxPoints * 3) : null;

    let validCount = 0;
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

    const xIndex = PCDheader.fields.indexOf('x');
    const yIndex = PCDheader.fields.indexOf('y');
    const zIndex = PCDheader.fields.indexOf('z');
    const xType = xIndex >= 0 ? PCDheader.type[xIndex] : 'F';
    const yType = yIndex >= 0 ? PCDheader.type[yIndex] : 'F';
    const zType = zIndex >= 0 ? PCDheader.type[zIndex] : 'F';
    const xSize = xIndex >= 0 ? PCDheader.size[xIndex] : 4;
    const ySize = yIndex >= 0 ? PCDheader.size[yIndex] : 4;
    const zSize = zIndex >= 0 ? PCDheader.size[zIndex] : 4;
    const isStandardFloat32XYZ = (
        xType === 'F' && xSize === 4 &&
        yType === 'F' && ySize === 4 &&
        zType === 'F' && zSize === 4
    );

    const nxIndex = hasNormal ? PCDheader.fields.indexOf('normal_x') : -1;
    const nyIndex = hasNormal ? PCDheader.fields.indexOf('normal_y') : -1;
    const nzIndex = hasNormal ? PCDheader.fields.indexOf('normal_z') : -1;
    const nxType = nxIndex >= 0 ? PCDheader.type[nxIndex] : 'F';
    const nyType = nyIndex >= 0 ? PCDheader.type[nyIndex] : 'F';
    const nzType = nzIndex >= 0 ? PCDheader.type[nzIndex] : 'F';
    const nxSize = nxIndex >= 0 ? PCDheader.size[nxIndex] : 4;
    const nySize = nyIndex >= 0 ? PCDheader.size[nyIndex] : 4;
    const nzSize = nzIndex >= 0 ? PCDheader.size[nzIndex] : 4;

    if (PCDheader.data === 'ascii' && asciiLines) {
        const rgbFieldIndex = hasRGB ? PCDheader.fields.indexOf('rgb') : -1;
        const rgbType = rgbFieldIndex >= 0 ? PCDheader.type[rgbFieldIndex] : 'U';
        const farr = new Float32Array(1);
        const iarr = new Int32Array(farr.buffer);

        for (let i = 0, l = asciiLines.length; i < l; i++) {
            const lineStr = asciiLines[i];
            if (!lineStr) continue;
            const line = lineStr.split(' ');
            if (offset.x === undefined) continue;

            const px = parseFloat(line[offset.x]);
            const py = parseFloat(line[offset.y]);
            const pz = parseFloat(line[offset.z]);
            if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz)) continue;

            const outIdx = validCount * 3;
            rawPositions[outIdx] = px;
            rawPositions[outIdx + 1] = py;
            rawPositions[outIdx + 2] = pz;

            if (px < minX) minX = px;
            if (px > maxX) maxX = px;
            if (py < minY) minY = py;
            if (py > maxY) maxY = py;
            if (pz < minZ) minZ = pz;
            if (pz > maxZ) maxZ = pz;

            if (hasRGB) {
                const flt = parseFloat(line[offset.rgb]);
                let rgb = flt;
                if (rgbType === 'F') {
                    farr[0] = flt;
                    rgb = iarr[0];
                }
                const r = ((rgb >> 16) & 0x0000ff) / 255.0;
                const g = ((rgb >> 8) & 0x0000ff) / 255.0;
                const b = (rgb & 0x0000ff) / 255.0;
                rawColors[outIdx] = srgbToLinear(r);
                rawColors[outIdx + 1] = srgbToLinear(g);
                rawColors[outIdx + 2] = srgbToLinear(b);
            }

            if (hasNormal && rawNormals) {
                const nx = parseFloat(line[offset.normal_x]);
                const ny = parseFloat(line[offset.normal_y]);
                const nz = parseFloat(line[offset.normal_z]);
                rawNormals[outIdx] = Number.isFinite(nx) ? nx : 0;
                rawNormals[outIdx + 1] = Number.isFinite(ny) ? ny : 0;
                rawNormals[outIdx + 2] = Number.isFinite(nz) ? nz : 0;
            }

            validCount++;
        }
    } else if (PCDheader.data === 'binary_compressed') {
        const sizes = new Uint32Array(buffer.slice(PCDheader.headerLen, PCDheader.headerLen + 8));
        const compressedSize = sizes[0];
        const decompressedSize = sizes[1];
        const decompressed = decompressLZF(
            new Uint8Array(buffer, PCDheader.headerLen + 8, compressedSize),
            decompressedSize
        );
        const dataview = new DataView(decompressed.buffer);
        const pts = PCDheader.points;
        const rgbIndex = hasRGB ? PCDheader.fields.indexOf('rgb') : -1;
        const rgbSize = rgbIndex >= 0 ? PCDheader.size[rgbIndex] : 4;
        const xBase = pts * offset.x;
        const yBase = pts * offset.y;
        const zBase = pts * offset.z;
        const rgbBase = hasRGB ? pts * offset.rgb : 0;
        const nxBase = hasNormal ? pts * offset.normal_x : 0;
        const nyBase = hasNormal ? pts * offset.normal_y : 0;
        const nzBase = hasNormal ? pts * offset.normal_z : 0;

        for (let i = 0; i < pts; i++) {
            if (offset.x === undefined) break;
            const px = isStandardFloat32XYZ ?
                dataview.getFloat32(xBase + 4 * i, true) :
                getDataViewValue(dataview, xBase + xSize * i, xType, xSize);
            const py = isStandardFloat32XYZ ?
                dataview.getFloat32(yBase + 4 * i, true) :
                getDataViewValue(dataview, yBase + ySize * i, yType, ySize);
            const pz = isStandardFloat32XYZ ?
                dataview.getFloat32(zBase + 4 * i, true) :
                getDataViewValue(dataview, zBase + zSize * i, zType, zSize);

            if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz)) continue;

            const outIdx = validCount * 3;
            rawPositions[outIdx] = px;
            rawPositions[outIdx + 1] = py;
            rawPositions[outIdx + 2] = pz;

            if (px < minX) minX = px;
            if (px > maxX) maxX = px;
            if (py < minY) minY = py;
            if (py > maxY) maxY = py;
            if (pz < minZ) minZ = pz;
            if (pz > maxZ) maxZ = pz;

            if (hasRGB) {
                const rgbOff = rgbBase + rgbSize * i;
                const r = dataview.getUint8(rgbOff + 2) / 255.0;
                const g = dataview.getUint8(rgbOff + 1) / 255.0;
                const b = dataview.getUint8(rgbOff + 0) / 255.0;
                rawColors[outIdx] = srgbToLinear(r);
                rawColors[outIdx + 1] = srgbToLinear(g);
                rawColors[outIdx + 2] = srgbToLinear(b);
            }

            if (hasNormal && rawNormals) {
                const nx = getDataViewValue(dataview, nxBase + nxSize * i, nxType, nxSize);
                const ny = getDataViewValue(dataview, nyBase + nySize * i, nyType, nySize);
                const nz = getDataViewValue(dataview, nzBase + nzSize * i, nzType, nzSize);
                rawNormals[outIdx] = Number.isFinite(nx) ? nx : 0;
                rawNormals[outIdx + 1] = Number.isFinite(ny) ? ny : 0;
                rawNormals[outIdx + 2] = Number.isFinite(nz) ? nz : 0;
            }

            validCount++;
        }
    } else if (PCDheader.data === 'binary') {
        const dataview = new DataView(buffer, PCDheader.headerLen);
        const pts = PCDheader.points;
        const rowSize = PCDheader.rowSize;
        const offX = offset.x;
        const offY = offset.y;
        const offZ = offset.z;
        const offRGB = offset.rgb;
        const offNX = offset.normal_x;
        const offNY = offset.normal_y;
        const offNZ = offset.normal_z;

        for (let i = 0, row = 0; i < pts; i++, row += rowSize) {
            if (offX === undefined) break;
            const px = isStandardFloat32XYZ ?
                dataview.getFloat32(row + offX, true) :
                getDataViewValue(dataview, row + offX, xType, xSize);
            const py = isStandardFloat32XYZ ?
                dataview.getFloat32(row + offY, true) :
                getDataViewValue(dataview, row + offY, yType, ySize);
            const pz = isStandardFloat32XYZ ?
                dataview.getFloat32(row + offZ, true) :
                getDataViewValue(dataview, row + offZ, zType, zSize);

            if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz)) continue;

            const outIdx = validCount * 3;
            rawPositions[outIdx] = px;
            rawPositions[outIdx + 1] = py;
            rawPositions[outIdx + 2] = pz;

            if (px < minX) minX = px;
            if (px > maxX) maxX = px;
            if (py < minY) minY = py;
            if (py > maxY) maxY = py;
            if (pz < minZ) minZ = pz;
            if (pz > maxZ) maxZ = pz;

            if (hasRGB) {
                const r = dataview.getUint8(row + offRGB + 2) / 255.0;
                const g = dataview.getUint8(row + offRGB + 1) / 255.0;
                const b = dataview.getUint8(row + offRGB + 0) / 255.0;
                rawColors[outIdx] = srgbToLinear(r);
                rawColors[outIdx + 1] = srgbToLinear(g);
                rawColors[outIdx + 2] = srgbToLinear(b);
            }

            if (hasNormal && rawNormals) {
                const nx = getDataViewValue(dataview, row + offNX, nxType, nxSize);
                const ny = getDataViewValue(dataview, row + offNY, nyType, nySize);
                const nz = getDataViewValue(dataview, row + offNZ, nzType, nzSize);
                rawNormals[outIdx] = Number.isFinite(nx) ? nx : 0;
                rawNormals[outIdx + 1] = Number.isFinite(ny) ? ny : 0;
                rawNormals[outIdx + 2] = Number.isFinite(nz) ? nz : 0;
            }

            validCount++;
        }
    }

    const positions = validCount === maxPoints ? rawPositions : rawPositions.slice(0, validCount * 3);
    const colors = rawColors ?
        (validCount === maxPoints ? rawColors : rawColors.slice(0, validCount * 3)) :
        null;
    const normals = rawNormals ?
        (validCount === maxPoints ? rawNormals : rawNormals.slice(0, validCount * 3)) :
        null;

    return {
        positions,
        colors,
        normals,
        minX,
        minY,
        minZ,
        maxX,
        maxY,
        maxZ,
    };
}

function buildSceneDataFromPositions(positions, requestedCellSize, preBBox) {
    if (!positions || positions.length < 9) {
        const emptyGrid = new Float32Array(0);
        return {
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
        };
    }

    const totalPoints = Math.floor(positions.length / 3);
    let minX = preBBox ? preBBox[0] : Infinity;
    let minY = preBBox ? preBBox[1] : Infinity;
    let minZ = preBBox ? preBBox[2] : Infinity;
    let maxX = preBBox ? preBBox[3] : -Infinity;
    let maxY = preBBox ? preBBox[4] : -Infinity;
    let maxZ = preBBox ? preBBox[5] : -Infinity;

    if (!preBBox) {
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
    }

    if (!Number.isFinite(minX) || !Number.isFinite(minY)) {
        const emptyGrid = new Float32Array(0);
        return {
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
        };
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

    return {
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
    };
}

self.onmessage = function (e) {
    const data = e.data;
    if (!data) return;

    if (data.type === 'parseAndInitScene') {
        const id = data.id;
        const requestedCellSize = data.cellSize > 0 ? data.cellSize : 0.5;
        const parsed = parsePCDInWorker(data.buffer);
        const sceneData = buildSceneDataFromPositions(
            parsed.positions,
            requestedCellSize,
            [parsed.minX, parsed.minY, parsed.minZ, parsed.maxX, parsed.maxY, parsed.maxZ]
        );
        const transferList = [
            parsed.positions.buffer,
            sceneData.groundGrid.gridBuffer,
        ];
        if (parsed.colors) {
            transferList.push(parsed.colors.buffer);
        }
        if (parsed.normals) {
            transferList.push(parsed.normals.buffer);
        }
        self.postMessage({
            type: 'sceneParsedAndInitialized',
            id,
            positionsBuffer: parsed.positions.buffer,
            colorsBuffer: parsed.colors ? parsed.colors.buffer : null,
            normalsBuffer: parsed.normals ? parsed.normals.buffer : null,
            bbox: sceneData.bbox,
            groundGrid: sceneData.groundGrid,
            globalGroundPlane: sceneData.globalGroundPlane,
        }, transferList);
        return;
    }

    if (data.type === 'initScene') {
        const id = data.id;
        const positions = data.positions;
        const requestedCellSize = data.cellSize > 0 ? data.cellSize : 0.5;
        const sceneData = buildSceneDataFromPositions(positions, requestedCellSize, null);
        self.postMessage({
            type: 'sceneInitialized',
            id,
            bbox: sceneData.bbox,
            groundGrid: sceneData.groundGrid,
            globalGroundPlane: sceneData.globalGroundPlane,
        }, [sceneData.groundGrid.gridBuffer]);
    }
};
`;

function createInlineWorker(source: string, name: string): Worker | null {
    if (typeof window === 'undefined' || typeof Worker === 'undefined' || typeof Blob === 'undefined' || typeof URL === 'undefined') {
        return null;
    }
    try {
        const blob = new Blob([source], { type: 'application/javascript' });
        const url = URL.createObjectURL(blob);
        const worker = new Worker(url, { name });
        URL.revokeObjectURL(url);
        return worker;
    } catch {
        return null;
    }
}

// Strict module-level singleton worker: guarantees at most 1 scene worker exists across the entire browser tab
// even if Canvas3d / Canvas3dViewImpl is instantiated multiple times.
let sharedMainWorker: Worker | null = null;
let sharedMainRequestId: number = 0;

function getSharedMainWorker(): Worker | null {
    if (!sharedMainWorker) {
        sharedMainWorker = createInlineWorker(MAIN_SCENE_WORKER_SOURCE, 'CVAT3D-MainSceneWorker');
    }
    return sharedMainWorker;
}

export function areSceneWorkersActive(): { mainWorkerActive: boolean; orthoWorkerActive: boolean } {
    return {
        mainWorkerActive: sharedMainWorker !== null,
        orthoWorkerActive: false,
    };
}

/**
 * Worker 1: Dedicated singleton background worker for the main Three.js perspective scene.
 * Offloads PCD ArrayBuffer parsing, NaN filtering, spatial ground elevation grid construction,
 * bounding box calculation, and global road plane GPF estimation from the main UI thread.
 */
export class MainSceneWorker {
    private worker: Worker | null = null;
    private pendingCallback: ((result: MainSceneInitResult) => void) | null = null;
    private pendingParseCallback: ((result: MainSceneParseResult) => void) | null = null;

    constructor() {
        this.worker = null;
    }

    private bindWorkerHandler(): void {
        if (!this.worker) return;
        this.worker.onmessage = (e: MessageEvent) => {
            const data = e.data;
            if (!data || data.id !== sharedMainRequestId) return;

            if (data.type === 'sceneParsedAndInitialized' && this.pendingParseCallback) {
                const {
                    positionsBuffer, colorsBuffer, normalsBuffer, bbox, groundGrid, globalGroundPlane,
                } = data;
                const positions = new Float32Array(positionsBuffer);
                const colors = colorsBuffer ? new Float32Array(colorsBuffer) : null;
                const normals = normalsBuffer ? new Float32Array(normalsBuffer) : null;
                const gridArray = new Float32Array(groundGrid.gridBuffer);
                const callback = this.pendingParseCallback;
                this.pendingParseCallback = null;

                callback({
                    positions,
                    colors,
                    normals,
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
                return;
            }

            if (data.type === 'sceneInitialized' && this.pendingCallback) {
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
            }
        };
    }

    public parseAndInitScene(
        buffer: ArrayBuffer,
        onResult: (result: MainSceneParseResult) => void,
        cellSize: number = 0.5,
        fallbackParser?: (buf: ArrayBuffer) => { positions: Float32Array; colors: Float32Array | null; normals: Float32Array | null },
    ): void {
        const id = ++sharedMainRequestId;
        this.pendingParseCallback = onResult;
        this.pendingCallback = null;

        if (!this.worker) {
            this.worker = getSharedMainWorker();
        }

        if (this.worker) {
            this.bindWorkerHandler();
            this.worker.postMessage(
                {
                    type: 'parseAndInitScene',
                    id,
                    buffer,
                    cellSize,
                },
                [buffer],
            );
            return;
        }

        // Synchronous fallback when Web Worker is unavailable
        if (fallbackParser) {
            const { positions, colors, normals } = fallbackParser(buffer);
            const gridInstance = new GroundElevationGrid(positions, cellSize);
            const globalGroundPlane = computeSceneGroundPlane(positions, 35.0);
            if (id === sharedMainRequestId && this.pendingParseCallback) {
                const callback = this.pendingParseCallback;
                this.pendingParseCallback = null;
                callback({
                    positions,
                    colors,
                    normals,
                    bbox: [gridInstance.minX, gridInstance.minY, -2, gridInstance.maxX, gridInstance.maxY, 5],
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
    }

    public initScene(
        positions: ArrayLike<number>,
        onResult: (result: MainSceneInitResult) => void,
        cellSize: number = 0.5,
    ): void {
        const id = ++sharedMainRequestId;
        this.pendingCallback = onResult;
        this.pendingParseCallback = null;

        if (!this.worker) {
            this.worker = getSharedMainWorker();
        }

        if (this.worker) {
            this.bindWorkerHandler();
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
        if (id === sharedMainRequestId && this.pendingCallback) {
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
        sharedMainRequestId++;
        this.pendingCallback = null;
        this.pendingParseCallback = null;
    }
}
