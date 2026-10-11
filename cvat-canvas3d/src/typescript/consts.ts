// Copyright (C) 2021-2022 Intel Corporation
// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

const BASE_GRID_WIDTH = 2;
const MOVEMENT_FACTOR = 200;
const DOLLY_FACTOR = 5;
const MAX_DISTANCE = 100;
const MIN_DISTANCE = 0.3;
const ORTHO_FRUSTUM_HEIGHT = 1;
const ROTATION_HELPER_OFFSET = 0.75;
const CAMERA_REFERENCE = 'camRef';
const CUBOID_EDGE_NAME = 'edges';
const ROTATION_HELPER_NAME = '2DRotationHelper';
const PLANE_ROTATION_HELPER = 'planeRotationHelper';
const RESIZE_HELPER_NAME = '2DResizeHelper';
const FOV_DEFAULT = 1;
const SIDE_VIEWS_MAX_ZOOM = 20;
const SIDE_VIEWS_MIN_ZOOM = 0.01;
const DEFAULT_GROUP_COLOR = '#e0e0e0';
const DEFAULT_OUTLINE_COLOR = '#000000';
const GROUPING_COLOR = '#8b008b';
const MERGING_COLOR = '#0000ff';
const SPLITTING_COLOR = '#1e90ff';
const BASE_POINT_SIZE = 4;

export const DEFAULT_DIMENSION_PRIOR: [number, number, number] = [4.5, 2.0, 1.6];

export const CLASS_DIMENSION_PRIORS: Record<string, [number, number, number]> = {
    vehicles: [4.8, 1.9, 1.6],
    vehicle: [4.8, 1.9, 1.6],
    car: [4.7, 1.85, 1.5],
    suv: [4.8, 1.95, 1.7],
    van: [5.2, 2.0, 2.0],
    truck: [10.5, 2.5, 3.2],
    bus: [12.0, 2.55, 3.4],
    motorcycle: [2.1, 0.8, 1.2],
    bicycle: [1.8, 0.6, 1.1],
    pedestrian: [0.6, 0.6, 1.75],
    person: [0.6, 0.6, 1.75],
};

export function parseDimensionsString(val: string): [number, number, number] | null {
    if (!val || typeof val !== 'string') return null;
    try {
        const parsed = JSON.parse(val);
        if (Array.isArray(parsed) && parsed.length >= 3) {
            const [x, y, z] = parsed.map(Number);
            if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) && x > 0 && y > 0 && z > 0) {
                return [x, y, z];
            }
        } else if (typeof parsed === 'object' && parsed !== null) {
            const p = parsed as Record<string, unknown>;
            const x = Number(p.length ?? p.x ?? p.dx);
            const y = Number(p.width ?? p.y ?? p.dy);
            const z = Number(p.height ?? p.z ?? p.dz);
            if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) && x > 0 && y > 0 && z > 0) {
                return [x, y, z];
            }
        }
    } catch {
        // Not JSON formatted, fallback to delimiter splitting
    }

    const clean = val.replace(/[[\](){}]/g, '').trim();
    const parts = clean.split(/[\s,;xX]+/).map(Number).filter((n) => Number.isFinite(n) && n > 0);
    if (parts.length >= 3) {
        return [parts[0], parts[1], parts[2]];
    }
    return null;
}

export function extractDimensionsFromAttributes(
    attrs: unknown,
    stateAttrs?: Record<string, unknown> | null,
): [number, number, number] | null {
    if (!attrs) return null;

    if (Array.isArray(attrs)) {
        for (const attr of attrs) {
            const name = String(attr?.name || '').toLowerCase().trim();
            if (
                name === 'dimensions' ||
                name === 'dimension' ||
                name === 'size' ||
                name === 'default_dimension' ||
                name === 'default_dimensions' ||
                name === 'default_dims' ||
                name === 'extent' ||
                name === 'cuboid_size' ||
                name === 'bbox_size'
            ) {
                const val = (stateAttrs && (stateAttrs[attr.id] ?? stateAttrs[attr.name])) ??
                    attr.defaultValue ??
                    (Array.isArray(attr.values) && attr.values.length ? attr.values[0] : null);
                if (typeof val === 'string') {
                    const parsed = parseDimensionsString(val);
                    if (parsed) return parsed;
                } else if (Array.isArray(val) && val.length >= 3) {
                    const [x, y, z] = val.map(Number);
                    if (x > 0 && y > 0 && z > 0) return [x, y, z];
                }
            }
        }

        let length: number | null = null;
        let width: number | null = null;
        let height: number | null = null;

        for (const attr of attrs) {
            const name = String(attr?.name || '').toLowerCase().trim();
            const rawVal = (stateAttrs && (stateAttrs[attr.id] ?? stateAttrs[attr.name])) ??
                attr.defaultValue ??
                (Array.isArray(attr.values) && attr.values.length ? attr.values[0] : null);
            const val = Number(rawVal);

            if (Number.isFinite(val) && val > 0) {
                if (name === 'length' || name === 'len' || name === 'dx' || name === 'dim_x' || name === 'size_x') {
                    length = val;
                } else if (name === 'width' || name === 'wid' || name === 'dy' || name === 'dim_y' || name === 'size_y') {
                    width = val;
                } else if (name === 'height' || name === 'hgt' || name === 'dz' || name === 'dim_z' || name === 'size_z') {
                    height = val;
                }
            }
        }

        if (length !== null && width !== null && height !== null) {
            return [length, width, height];
        }
    } else if (typeof attrs === 'object' && attrs !== null) {
        for (const [key, val] of Object.entries(attrs as Record<string, unknown>)) {
            const lowerKey = key.toLowerCase().trim();
            if (
                lowerKey === 'dimensions' ||
                lowerKey === 'dimension' ||
                lowerKey === 'size' ||
                lowerKey === 'default_dimension' ||
                lowerKey === 'default_dimensions' ||
                lowerKey === 'extent'
            ) {
                if (typeof val === 'string') {
                    const parsed = parseDimensionsString(val);
                    if (parsed) return parsed;
                } else if (Array.isArray(val) && val.length >= 3) {
                    const [x, y, z] = val.map(Number);
                    if (x > 0 && y > 0 && z > 0) return [x, y, z];
                }
            }
        }
    }

    return null;
}

export function parseGeometryBounds(bounds: unknown): [number, number, number] | null {
    if (!bounds) return null;
    if (Array.isArray(bounds) && bounds.length >= 3) {
        const [x, y, z] = bounds.map(Number);
        if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) && x > 0 && y > 0 && z > 0) {
            return [x, y, z];
        }
    }
    if (typeof bounds === 'object' && bounds !== null) {
        const b = bounds as Record<string, any>;
        if (b.min && b.max && typeof b.min === 'object' && typeof b.max === 'object') {
            const dx = Math.abs(Number(b.max.x) - Number(b.min.x));
            const dy = Math.abs(Number(b.max.y) - Number(b.min.y));
            const dz = Math.abs(Number(b.max.z) - Number(b.min.z));
            if (Number.isFinite(dx) && Number.isFinite(dy) && Number.isFinite(dz) && dx > 0 && dy > 0 && dz > 0) {
                return [
                    parseFloat(Math.max(0.2, Math.min(50, dx)).toFixed(3)),
                    parseFloat(Math.max(0.2, Math.min(50, dy)).toFixed(3)),
                    parseFloat(Math.max(0.2, Math.min(50, dz)).toFixed(3)),
                ];
            }
        }
        if ('x' in b && 'y' in b && 'z' in b) {
            const x = Math.abs(Number(b.x));
            const y = Math.abs(Number(b.y));
            const z = Math.abs(Number(b.z));
            if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) && x > 0 && y > 0 && z > 0) {
                return [
                    parseFloat(Math.max(0.2, Math.min(50, x)).toFixed(3)),
                    parseFloat(Math.max(0.2, Math.min(50, y)).toFixed(3)),
                    parseFloat(Math.max(0.2, Math.min(50, z)).toFixed(3)),
                ];
            }
        }
    }
    return null;
}

export interface LabelAttributeDefinition {
    id?: number;
    name?: string;
    defaultValue?: string;
    values?: string[];
}

export interface DimensionOptions {
    label?: { name?: string; attributes?: LabelAttributeDefinition[] };
    attributes?: Record<string, unknown> | LabelAttributeDefinition[];
    task?: { attributes?: LabelAttributeDefinition[] };
    geometryBounds?: [number, number, number] |
        { x: number; y: number; z: number } |
        { min: { x: number; y: number; z: number }; max: { x: number; y: number; z: number } };
    defaultDims?: [number, number, number];
    fallbackDimensions?: [number, number, number];
}

export function getClassDimensionPrior(
    labelOrName?: string | {
        name?: string;
        attributes?: unknown;
        label?: { name?: string; attributes?: unknown };
    } | null,
    fallback?: [number, number, number] | DimensionOptions | null,
    geometryBounds?: unknown,
): [number, number, number] {
    const rawName = typeof labelOrName === 'string' ?
        labelOrName :
        ((labelOrName as any)?.name || (labelOrName as any)?.label?.name || '');
    const lower = rawName.toLowerCase().trim();

    // 1. Case-insensitive matching against class dimension priors
    if (lower) {
        // Direct key lookup
        for (const [key, dims] of Object.entries(CLASS_DIMENSION_PRIORS)) {
            if (key.toLowerCase().trim() === lower) {
                return dims;
            }
        }
        // Substring / token matching (prioritizing longer, more specific keys)
        const sortedEntries = Object.entries(CLASS_DIMENSION_PRIORS)
            .sort(([a], [b]) => b.length - a.length);
        for (const [key, dims] of sortedEntries) {
            const lowerKey = key.toLowerCase().trim();
            if (lower.includes(lowerKey) || lowerKey.includes(lower)) {
                return dims;
            }
        }
    }

    // 2. Fallback to task/label default attributes
    const optionsObj = (fallback && !Array.isArray(fallback)) ? (fallback as DimensionOptions) : null;
    const labelAttrs = (labelOrName as any)?.attributes ||
        (labelOrName as any)?.label?.attributes ||
        optionsObj?.label?.attributes;
    const stateAttrs = (optionsObj?.attributes as Record<string, unknown>) ||
        ((labelOrName as any)?.state?.attributes as Record<string, unknown>);

    const parsedAttrDims = extractDimensionsFromAttributes(labelAttrs, stateAttrs) ||
        extractDimensionsFromAttributes(optionsObj?.attributes) ||
        extractDimensionsFromAttributes(optionsObj?.task?.attributes);

    if (parsedAttrDims) {
        return parsedAttrDims;
    }

    // 3. Fallback to geometry bounding heuristics
    const geom = optionsObj?.geometryBounds ?? geometryBounds;
    const parsedGeom = parseGeometryBounds(geom);
    if (parsedGeom) {
        return parsedGeom;
    }

    // 4. Fallback to caller-specified fallback array or defaultDims
    if (Array.isArray(fallback) && fallback.length >= 3) {
        const [x, y, z] = fallback.map(Number);
        if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) && x > 0 && y > 0 && z > 0) {
            return [x, y, z];
        }
    }
    if (optionsObj?.defaultDims && Array.isArray(optionsObj.defaultDims) && optionsObj.defaultDims.length >= 3) {
        const [x, y, z] = optionsObj.defaultDims.map(Number);
        if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) && x > 0 && y > 0 && z > 0) {
            return [x, y, z];
        }
    }
    if (optionsObj?.fallbackDimensions &&
        Array.isArray(optionsObj.fallbackDimensions) &&
        optionsObj.fallbackDimensions.length >= 3) {
        const [x, y, z] = optionsObj.fallbackDimensions.map(Number);
        if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) && x > 0 && y > 0 && z > 0) {
            return [x, y, z];
        }
    }

    // 5. Final fallback to default dimension prior
    return DEFAULT_DIMENSION_PRIOR;
}

export function isPedestrianClass(labelOrName?: unknown): boolean {
    const rawName = typeof labelOrName === 'string' ?
        labelOrName :
        ((labelOrName as any)?.name || (labelOrName as any)?.label?.name || '');
    const lower = rawName.toLowerCase().trim();
    return lower.includes('pedestrian') || lower.includes('person') ||
        lower.includes('human') || lower.includes('child');
}

export default {
    BASE_GRID_WIDTH,
    MOVEMENT_FACTOR,
    DOLLY_FACTOR,
    MAX_DISTANCE,
    MIN_DISTANCE,
    ORTHO_FRUSTUM_HEIGHT,
    ROTATION_HELPER_OFFSET,
    CAMERA_REFERENCE,
    CUBOID_EDGE_NAME,
    ROTATION_HELPER_NAME,
    PLANE_ROTATION_HELPER,
    RESIZE_HELPER_NAME,
    FOV_DEFAULT,
    SIDE_VIEWS_MAX_ZOOM,
    SIDE_VIEWS_MIN_ZOOM,
    DEFAULT_GROUP_COLOR,
    DEFAULT_OUTLINE_COLOR,
    GROUPING_COLOR,
    MERGING_COLOR,
    SPLITTING_COLOR,
    BASE_POINT_SIZE,
    CLASS_DIMENSION_PRIORS,
    DEFAULT_DIMENSION_PRIOR,
    getClassDimensionPrior,
    isPedestrianClass,
};
