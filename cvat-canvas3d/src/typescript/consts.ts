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

export function getClassDimensionPrior(labelName?: string): [number, number, number] {
    if (!labelName) {
        return DEFAULT_DIMENSION_PRIOR;
    }
    const lower = labelName.toLowerCase().trim();
    if (CLASS_DIMENSION_PRIORS[lower]) {
        return CLASS_DIMENSION_PRIORS[lower];
    }
    for (const [key, dims] of Object.entries(CLASS_DIMENSION_PRIORS)) {
        if (lower.includes(key)) {
            return dims;
        }
    }
    return DEFAULT_DIMENSION_PRIOR;
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
};
