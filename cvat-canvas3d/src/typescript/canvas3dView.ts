// Copyright (C) 2021-2022 Intel Corporation
// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import * as THREE from 'three';
import { PCDLoader } from 'three/examples/jsm/loaders/PCDLoader';
import CameraControls from 'camera-controls';
import { Canvas3dController } from './canvas3dController';
import { Listener, Master } from './master';
import consts, { getClassDimensionPrior, isPedestrianClass } from './consts';
import {
    Canvas3dModel, DrawData, Mode, Planes, UpdateReasons, ViewType,
} from './canvas3dModel';
import {
    createRotationHelper, removeRotationHelper,
    createResizeHelper, removeResizeHelper,
    createCuboidEdges, removeCuboidEdges, CuboidModel, makeCornerPointsMatrix,
} from './cuboid';
import { ObjectState, ObjectType } from '.';
import { disposeScene } from './utils';
import {
    snapCuboidToRoadAndOrient, fitClusterAtPoint,
    alignEulerToRoadPlane, GroundElevationGrid,
    computeSceneGroundPlane, estimateLocalRoadPlane,
} from './snapping';
import { MainSceneWorker, areSceneWorkersActive } from './sceneWorkers';
import { detectGpuCapabilities, GpuCapabilities } from './gpuDetector';

export interface Canvas3dPerformanceStats {
    fps: number;
    frameTimeMs: number;
    renderTimeMs: number;
    pointCount: number;
    orthoRoiPoints: number;
    cuboidCount: number;
    gpuTier: string;
    gpuRenderer: string;
    renderingMethod: string;
    mainWorkerActive: boolean;
}

export interface Canvas3dView {
    html(): ViewsDOM;
    render(): void;
    requestRender(frameCount?: number): void;
    keyControls(keys: KeyboardEvent): void;
    focusObjectByClientId(clientID: number, animate?: boolean): void;
    snapToRoad(clientID?: number | null, options?: { snapGround?: boolean; autoOrient?: boolean; snapGeometry?: boolean; }): void;
    revertSnap(clientID?: number | null): boolean;
    getPreviousCuboid(clientID?: number | null): number[] | null;
    invertHeading(clientID?: number | null): void;
    getActiveElement(): number | null;
    setCuboidDimensions(clientID: number | null, dimensions: [number, number, number]): void;
    nudgeCuboid(
        dxLocal: number,
        dyLocal: number,
        dzWorld: number,
        dYawRad: number,
        clientID?: number | null,
    ): boolean;
    toggleObjectIsolation(clientID?: number | null): boolean;
    isObjectIsolatedState(): boolean;
    toggleGroundStripping(elevationOffset?: number): boolean;
    isGroundStrippedState(): boolean;
    snapTrackToRoad(
        jobInstance: any,
        clientID?: number | null,
        onProgress?: (message: string, percent: number) => void,
    ): Promise<{ processedFrames: number }>;
    getPerformanceStats(): Canvas3dPerformanceStats;
    togglePerformanceHud(enabled?: boolean): boolean;
    destroy(): void;
}

export enum CameraAction {
    ZOOM_IN = 'KeyI',
    MOVE_UP = 'KeyU',
    MOVE_DOWN = 'KeyO',
    MOVE_LEFT = 'KeyJ',
    ZOOM_OUT = 'KeyK',
    MOVE_RIGHT = 'KeyL',
    TILT_UP = 'ArrowUp',
    TILT_DOWN = 'ArrowDown',
    ROTATE_RIGHT = 'ArrowRight',
    ROTATE_LEFT = 'ArrowLeft',
}

export type Views = {
    [key in ViewType]: RenderView;
};

export type ViewsDOM = {
    [key in ViewType]: HTMLCanvasElement;
};

export interface RenderView {
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera?: THREE.PerspectiveCamera | THREE.OrthographicCamera;
    controls?: CameraControls;
    rayCaster?: {
        renderer: THREE.Raycaster;
        mouseVector: THREE.Vector2;
    };
}

interface DrawnObjectData {
    serverID: number | null;
    clientID: number;
    labelID: number;
    labelColor: string;
    points: number[];
    groupID: number | null;
    groupColor: string;
    color: string;
    occluded: boolean;
    outside: boolean;
    hidden: boolean;
    pinned: boolean;
    lock: boolean;
    updated: number;
}

interface SideViewsZoomMemory {
    serverID: number | null;
    [ViewType.TOP]: number | null;
    [ViewType.SIDE]: number | null;
    [ViewType.FRONT]: number | null;
}

const BOTTOM_VIEWS = [
    ViewType.TOP,
    ViewType.SIDE,
    ViewType.FRONT,
];

const ALL_VIEWS = [...BOTTOM_VIEWS, ViewType.PERSPECTIVE];

function drawnDataFromState(state: ObjectState): DrawnObjectData {
    return {
        serverID: state.serverID,
        clientID: state.clientID,
        labelID: state.label.id,
        labelColor: state.label.color,
        groupID: state.group?.id || null,
        groupColor: state.group?.color || '#ffffff',
        points: [...state.points],
        color: state.color,
        hidden: state.hidden,
        lock: state.lock,
        occluded: state.occluded,
        outside: state.outside,
        pinned: state.pinned,
        updated: state.updated,
    };
}

export class Canvas3dViewImpl implements Canvas3dView, Listener {
    private controller: Canvas3dController;
    private gpuCapabilities: GpuCapabilities;
    private mainRenderer: THREE.WebGLRenderer;
    private mainSceneWorker: MainSceneWorker;
    private onDocumentMouseUp: () => void;
    private views: Views;
    private clock: THREE.Clock;
    private speed: number;
    private cube: CuboidModel;
    private isPerspectiveBeingDragged: boolean;
    private activatedElementID: number | null;
    private isCtrlDown: boolean;
    private stateToBeSplitted: ObjectState | null;
    private statesToBeGrouped: ObjectState[];
    private statesToBeMerged: ObjectState[];
    private sceneBBox: THREE.Box3;
    private sideViewsZoomMemory: Record<number, SideViewsZoomMemory>;
    private model: Canvas3dModel & Master;
    private drawnObjects: Record<number, {
        data: DrawnObjectData;
        cuboid: CuboidModel;
    }>;
    private hoverNeedsUpdate: boolean;
    private needsPerspectiveRenderCount: number = 2;
    private needsOrthoRenderCount: number = 2;
    private viewSizes: Record<ViewType, { width: number; height: number }> = {
        [ViewType.PERSPECTIVE]: { width: 0, height: 0 },
        [ViewType.TOP]: { width: 0, height: 0 },
        [ViewType.SIDE]: { width: 0, height: 0 },
        [ViewType.FRONT]: { width: 0, height: 0 },
    };
    private pointCloudPoints: any = null;
    private basePointCloudGeometry: THREE.BufferGeometry | null = null;
    private filteredPointCloudGeometry: THREE.BufferGeometry | null = null;
    private lastIsolationToggleTime: number = 0;
    private groundGrid: GroundElevationGrid | null = null;
    private previousCuboids: Map<number, number[][]> = new Map();
    private isObjectIsolated: boolean = false;
    private isolatedClientID: number | null = null;
    private isGroundStripped: boolean = false;
    private groundStrippingOffset: number = 0.12;
    private groundClippingPlane: THREE.Plane | null = null;
    private cachedGlobalGroundPlane: [number, number, number, number] | null = null;
    private pointCloudMeshes: THREE.Points[] = [];
    private hiddenByIsolation: Set<number> = new Set();
    private orthoPointsGeometry: THREE.BufferGeometry | null = null;
    private orthoRoiSphereUniform: { value: THREE.Vector4 } = { value: new THREE.Vector4(0, 0, 0, 0) };
    private lastRoiCenter: THREE.Vector3 | null = null;
    private scenePlanes: Record<string, THREE.Mesh | null> = {
        [ViewType.TOP]: null,
        [ViewType.SIDE]: null,
        [ViewType.FRONT]: null,
    };
    private geometryCache: Map<number, {
        geometry: THREE.BufferGeometry;
        strippedGeometry?: THREE.BufferGeometry | null;
        groundGrid: GroundElevationGrid;
        globalGroundPlane: [number, number, number, number];
    }> = new Map();
    private perfFps: number = 60;
    private perfFrameTimeMs: number = 16.7;
    private perfRenderTimeMs: number = 0.5;
    private perfPointCount: number = 0;
    private perfOrthoRoiPoints: number = 0;
    private perfHudEnabled: boolean = false;
    private perfHudElement: HTMLDivElement | null = null;
    private perfHudLastUpdate: number = 0;

    public getPerformanceStats(): Canvas3dPerformanceStats {
        const { mainWorkerActive } = areSceneWorkersActive();
        let gpuTierLabel = 'iGPU';
        let renderingMethodLabel = 'iGPU Balanced (4-View WebGL2 + Shader ROI)';
        if (this.gpuCapabilities.tier === 'dgpu') {
            gpuTierLabel = 'dGPU';
            renderingMethodLabel = 'dGPU 4-View Direct WebGL2 + Shader ROI';
        } else if (this.gpuCapabilities.tier === 'software') {
            gpuTierLabel = 'Software';
            renderingMethodLabel = 'Software Low-Overhead';
        }
        const pointCount = this.perfPointCount ||
            (this.pointCloudPoints?.geometry?.attributes?.position?.count ?? 0);
        return {
            fps: Math.round(this.perfFps),
            frameTimeMs: Number(this.perfFrameTimeMs.toFixed(2)),
            renderTimeMs: Number(this.perfRenderTimeMs.toFixed(2)),
            pointCount,
            orthoRoiPoints: this.perfOrthoRoiPoints,
            cuboidCount: Object.keys(this.drawnObjects).length,
            gpuTier: gpuTierLabel,
            gpuRenderer: this.gpuCapabilities.renderer,
            renderingMethod: renderingMethodLabel,
            mainWorkerActive,
        };
    }

    public togglePerformanceHud(enabled?: boolean): boolean {
        this.perfHudEnabled = typeof enabled === 'boolean' ? enabled : !this.perfHudEnabled;
        if (!this.perfHudEnabled) {
            if (this.perfHudElement) {
                this.perfHudElement.style.display = 'none';
            }
            return false;
        }
        this.updatePerformanceHud();
        return true;
    }

    private updatePerformanceHud(): void {
        if (!this.perfHudEnabled || typeof window === 'undefined') return;
        const container = this.views?.perspective?.renderer?.domElement?.parentElement;
        if (!container) return;

        if (!this.perfHudElement) {
            const el = window.document.createElement('div');
            el.className = 'cvat-3d-performance-hud';
            el.style.cssText = [
                'pointer-events: none',
                'position: absolute',
                'top: 10px',
                'left: 10px',
                'z-index: 20',
                'background: rgba(15, 23, 42, 0.85)',
                'color: #e2e8f0',
                'border: 1px solid rgba(56, 189, 248, 0.35)',
                'border-radius: 6px',
                'padding: 8px 10px',
                'font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
                'font-size: 11px',
                'line-height: 1.45',
                'box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4)',
                'white-space: pre',
            ].join('; ');
            this.perfHudElement = el;
        }

        if (this.perfHudElement.parentElement !== container) {
            const computedPos = window.getComputedStyle(container).position;
            if (!computedPos || computedPos === 'static') {
                container.style.position = 'relative';
            }
            container.appendChild(this.perfHudElement);
        }

        this.perfHudElement.style.display = 'block';
        const stats = this.getPerformanceStats();
        const w1 = stats.mainWorkerActive ? 'Active (Singleton)' : 'Standby (Lazy)';
        this.perfHudElement.textContent = [
            `GPU Tier    : ${stats.gpuTier} (${stats.gpuRenderer})`,
            `Render Mode : ${stats.renderingMethod}`,
            `Scene Worker: MainSceneWorker [${w1}] | VRAM Cache: ${this.geometryCache.size}/64 frames`,
            `FPS / Frame : ${stats.fps} FPS (${stats.frameTimeMs.toFixed(1)} ms | GPU draw: ${stats.renderTimeMs.toFixed(2)} ms)`,
            `Points      : ${stats.pointCount.toLocaleString()} main | ${stats.orthoRoiPoints.toLocaleString()} ortho ROI | ${stats.cuboidCount} cuboids`,
        ].join('\n');
    }

    public requestRender(frameCount: number = 2): void {
        this.needsPerspectiveRenderCount = Math.max(this.needsPerspectiveRenderCount, frameCount);
        this.needsOrthoRenderCount = Math.max(this.needsOrthoRenderCount, frameCount);
    }

    public requestPerspectiveRender(frameCount: number = 2): void {
        this.needsPerspectiveRenderCount = Math.max(this.needsPerspectiveRenderCount, frameCount);
    }

    public requestOrthoRender(frameCount: number = 2): void {
        this.needsOrthoRenderCount = Math.max(this.needsOrthoRenderCount, frameCount);
    }

    public focusObjectByClientId(clientID: number, animate: boolean = true): void {
        const cuboidModel = this.drawnObjects[clientID];
        if (cuboidModel) {
            this.fitObject(cuboidModel.cuboid, animate);
            this.requestRender();
        }
    }

    public getActiveElement(): number | null {
        if (this.activatedElementID !== null) return this.activatedElementID;
        if (this.model.data.activeElement.clientID !== null) return +this.model.data.activeElement.clientID;
        return null;
    }

    private getBasePointCloudPositions(): Float32Array | null {
        const geo = this.basePointCloudGeometry || this.pointCloudPoints?.geometry;
        return (geo?.attributes?.position?.array as Float32Array) || null;
    }

    public snapToRoad(clientID?: number | null, options?: { snapGround?: boolean; autoOrient?: boolean; snapGeometry?: boolean; }): void {
        let targetClientID: number | null = typeof clientID === 'number' ? clientID : null;
        if (targetClientID === null) {
            targetClientID = this.getActiveElement();
        }

        if (targetClientID === null || !(targetClientID in this.drawnObjects)) {
            return;
        }

        this.model.data.activeElement.clientID = `${targetClientID}`;
        this.activatedElementID = targetClientID;

        const positions = this.getBasePointCloudPositions();
        if (!positions) {
            return;
        }
        const { cuboid, data } = this.drawnObjects[targetClientID];
        if (data.lock) return;

        const state = this.model.objects.find((obj: ObjectState) => obj.clientID === targetClientID);
        if (!state) return;

        const currentPoints = [
            cuboid.perspective.position.x,
            cuboid.perspective.position.y,
            cuboid.perspective.position.z,
            cuboid.perspective.rotation.x,
            cuboid.perspective.rotation.y,
            cuboid.perspective.rotation.z,
            cuboid.perspective.scale.x,
            cuboid.perspective.scale.y,
            cuboid.perspective.scale.z,
            0, 0, 0, 0, 0, 0, 0,
        ];

        const previousPoints = [...currentPoints];
        if (!this.previousCuboids.has(targetClientID)) {
            this.previousCuboids.set(targetClientID, []);
        }
        this.previousCuboids.get(targetClientID)!.push(previousPoints);

        const snapMode = this.model.data.configuration.snapMode ?? 'ground';
        const defaultFlags = snapMode === 'ground' ?
            { snapGround: true, autoOrient: false, snapGeometry: false } :
            { snapGround: true, autoOrient: true, snapGeometry: true };

        const snapResult = snapCuboidToRoadAndOrient(currentPoints, positions, {
            labelName: state.label?.name,
            ...defaultFlags,
            ...(options || {}),
        });
        const [x, y, z, rx, ry, rz, dx, dy, dz] = snapResult.points;

        cuboid.setPosition(x, y, z);
        cuboid.setRotation(rx, ry, rz);
        cuboid.setScale(dx, dy, dz);
        data.points = [...snapResult.points];

        this.translateReferencePlane(new THREE.Vector3(x, y, z));
        try {
            this.rotatePlane(null as any, null as any);
        } catch (_) {
            // ignore if not all views are initialized
        }

        this.updateResizeHelperPos();
        this.updateRotationHelperPos();
        this.detachCamera();
        BOTTOM_VIEWS.forEach((view) => this.updateHelperPointsSize(view));
        this.updateCameraFrustumPlane();

        this.dispatchEvent(
            new CustomEvent('canvas.edited', {
                bubbles: false,
                cancelable: true,
                detail: {
                    state,
                    points: [...snapResult.points],
                    previousPoints: [...previousPoints],
                },
            }),
        );

        this.dispatchEvent(
            new CustomEvent('canvas.snapped', {
                bubbles: false,
                cancelable: true,
                detail: {
                    state,
                    points: [...snapResult.points],
                    previousPoints: [...previousPoints],
                    metrics: snapResult.metrics,
                },
            }),
        );

        this.requestRender();
        this.render();
    }

    public revertSnap(clientID?: number | null): boolean {
        let targetClientID: number | null = typeof clientID === 'number' ? clientID : null;
        if (targetClientID === null) {
            targetClientID = this.getActiveElement();
        }

        if (targetClientID === null || !this.previousCuboids.has(targetClientID)) {
            return false;
        }

        this.model.data.activeElement.clientID = `${targetClientID}`;
        this.activatedElementID = targetClientID;

        const stack = this.previousCuboids.get(targetClientID)!;
        if (stack.length === 0) {
            this.previousCuboids.delete(targetClientID);
            return false;
        }

        const prevPoints = stack.pop()!;
        if (stack.length === 0) {
            this.previousCuboids.delete(targetClientID);
        }

        const { cuboid, data } = this.drawnObjects[targetClientID] || {};
        if (!cuboid || data?.lock) return false;

        const state = this.model.objects.find((obj: ObjectState) => obj.clientID === targetClientID);
        if (!state) return false;

        const snappedPoints = [
            cuboid.perspective.position.x,
            cuboid.perspective.position.y,
            cuboid.perspective.position.z,
            cuboid.perspective.rotation.x,
            cuboid.perspective.rotation.y,
            cuboid.perspective.rotation.z,
            cuboid.perspective.scale.x,
            cuboid.perspective.scale.y,
            cuboid.perspective.scale.z,
            0, 0, 0, 0, 0, 0, 0,
        ];

        cuboid.setPosition(prevPoints[0], prevPoints[1], prevPoints[2]);
        cuboid.setRotation(prevPoints[3], prevPoints[4], prevPoints[5]);
        cuboid.setScale(prevPoints[6], prevPoints[7], prevPoints[8]);
        data.points = [...prevPoints];

        this.translateReferencePlane(new THREE.Vector3(prevPoints[0], prevPoints[1], prevPoints[2]));
        try {
            this.rotatePlane(null as any, null as any);
        } catch (_) {
            // ignore if not all views initialized
        }

        this.updateResizeHelperPos();
        this.updateRotationHelperPos();
        this.detachCamera();
        BOTTOM_VIEWS.forEach((view) => this.updateHelperPointsSize(view));
        this.updateCameraFrustumPlane();

        this.dispatchEvent(
            new CustomEvent('canvas.edited', {
                bubbles: false,
                cancelable: true,
                detail: {
                    state,
                    points: [...prevPoints],
                    previousPoints: [...snappedPoints],
                },
            }),
        );

        this.dispatchEvent(
            new CustomEvent('canvas.reverted', {
                bubbles: false,
                cancelable: true,
                detail: {
                    state,
                    points: [...prevPoints],
                    previousPoints: [...snappedPoints],
                },
            }),
        );

        this.requestRender();
        this.render();
        return true;
    }

    public getPreviousCuboid(clientID?: number | null): number[] | null {
        let targetClientID: number | null = typeof clientID === 'number' ? clientID : null;
        if (targetClientID === null) {
            if (this.activatedElementID !== null) {
                targetClientID = this.activatedElementID;
            } else if (this.model.data.activeElement.clientID !== null) {
                targetClientID = +this.model.data.activeElement.clientID;
            }
        }

        if (targetClientID !== null && this.previousCuboids.has(targetClientID)) {
            const stack = this.previousCuboids.get(targetClientID)!;
            if (stack.length > 0) {
                return [...stack[stack.length - 1]];
            }
        }
        return null;
    }

    public invertHeading(clientID?: number | null): void {
        let targetClientID: number | null = typeof clientID === 'number' ? clientID : null;
        if (targetClientID === null) {
            targetClientID = this.getActiveElement();
        }

        if (targetClientID === null || !(targetClientID in this.drawnObjects)) {
            return;
        }

        this.model.data.activeElement.clientID = `${targetClientID}`;
        this.activatedElementID = targetClientID;

        const { cuboid, data } = this.drawnObjects[targetClientID];
        if (data.lock) return;

        const state = this.model.objects.find((obj: ObjectState) => obj.clientID === targetClientID);
        if (!state) return;

        const previousPoints = state.points && state.points.length >= 9 ? [...state.points] : [
            cuboid.perspective.position.x,
            cuboid.perspective.position.y,
            cuboid.perspective.position.z,
            cuboid.perspective.rotation.x,
            cuboid.perspective.rotation.y,
            cuboid.perspective.rotation.z,
            cuboid.perspective.scale.x,
            cuboid.perspective.scale.y,
            cuboid.perspective.scale.z,
            0, 0, 0, 0, 0, 0, 0,
        ];
        if (!this.previousCuboids.has(targetClientID)) {
            this.previousCuboids.set(targetClientID, []);
        }
        this.previousCuboids.get(targetClientID)!.push(previousPoints);

        const currentRot = cuboid.perspective.rotation;
        const currentEuler = new THREE.Euler(currentRot.x, currentRot.y, currentRot.z, 'XYZ');
        const norm = new THREE.Vector3(0, 0, 1).applyEuler(currentEuler).normalize();

        // Invert horizontal heading by PI radians
        const newYaw = Math.atan2(Math.sin(currentRot.z + Math.PI), Math.cos(currentRot.z + Math.PI));

        let finalRx = currentRot.x;
        let finalRy = currentRot.y;
        let finalRz = newYaw;

        // If tilted on a road plane, conform inverted heading to terrain slope
        if (norm.z >= 0.70) {
            const fHoriz = new THREE.Vector3(Math.cos(newYaw), Math.sin(newYaw), 0);
            const dot = fHoriz.dot(norm);
            const xLocal = fHoriz.clone().sub(norm.clone().multiplyScalar(dot));
            if (xLocal.lengthSq() > 1e-6) {
                xLocal.normalize();
                const yLocal = new THREE.Vector3().crossVectors(norm, xLocal).normalize();
                const zLocal = norm;
                const basisMatrix = new THREE.Matrix4().makeBasis(xLocal, yLocal, zLocal);
                const euler = new THREE.Euler().setFromRotationMatrix(basisMatrix, 'XYZ');
                finalRx = parseFloat(euler.x.toFixed(4));
                finalRy = parseFloat(euler.y.toFixed(4));
                finalRz = Math.cos(euler.z - newYaw) > 0.7 ? parseFloat(euler.z.toFixed(4)) : newYaw;
            }
        }

        const [x, y, z] = previousPoints;
        cuboid.setPosition(x, y, z);
        cuboid.setRotation(finalRx, finalRy, finalRz);
        const newPoints = [...previousPoints];
        newPoints[0] = x;
        newPoints[1] = y;
        newPoints[2] = z;
        newPoints[3] = finalRx;
        newPoints[4] = finalRy;
        newPoints[5] = finalRz;
        data.points = newPoints;

        this.translateReferencePlane(new THREE.Vector3(x, y, z));
        try {
            this.rotatePlane(null as any, null as any);
        } catch (_) {
            // ignore
        }

        this.updateResizeHelperPos();
        this.updateRotationHelperPos();
        this.detachCamera();
        BOTTOM_VIEWS.forEach((view) => this.updateHelperPointsSize(view));
        this.updateCameraFrustumPlane();

        this.dispatchEvent(
            new CustomEvent('canvas.edited', {
                bubbles: false,
                cancelable: true,
                detail: {
                    state,
                    points: [...newPoints],
                    previousPoints,
                },
            }),
        );

        this.dispatchEvent(
            new CustomEvent('canvas.inverted', {
                bubbles: false,
                cancelable: true,
                detail: {
                    state,
                    yawDeg: parseFloat(((finalRz * 180) / Math.PI).toFixed(1)),
                },
            }),
        );
        this.render();
    }

    public setCuboidDimensions(clientID: number | null, dimensions: [number, number, number]): void {
        let targetClientID: number | null = typeof clientID === 'number' ? clientID : null;
        if (targetClientID === null) {
            targetClientID = this.getActiveElement();
        }

        if (targetClientID === null || !(targetClientID in this.drawnObjects)) {
            return;
        }

        this.model.data.activeElement.clientID = `${targetClientID}`;
        this.activatedElementID = targetClientID;

        const { cuboid, data } = this.drawnObjects[targetClientID];
        if (data.lock) return;

        const state = this.model.objects.find((obj: ObjectState) => obj.clientID === targetClientID);
        if (!state) return;

        const previousPoints = state.points && state.points.length >= 9 ? [...state.points] : [
            cuboid.perspective.position.x,
            cuboid.perspective.position.y,
            cuboid.perspective.position.z,
            cuboid.perspective.rotation.x,
            cuboid.perspective.rotation.y,
            cuboid.perspective.rotation.z,
            cuboid.perspective.scale.x,
            cuboid.perspective.scale.y,
            cuboid.perspective.scale.z,
            0, 0, 0, 0, 0, 0, 0,
        ];
        if (!this.previousCuboids.has(targetClientID)) {
            this.previousCuboids.set(targetClientID, []);
        }
        this.previousCuboids.get(targetClientID)!.push(previousPoints);

        const [dx, dy, dz] = dimensions;
        cuboid.setScale(dx, dy, dz);

        const newPoints = [...previousPoints];
        newPoints[6] = dx;
        newPoints[7] = dy;
        newPoints[8] = dz;
        data.points = newPoints;
        if (state.points && state.points.length >= 9) {
            state.points[6] = dx;
            state.points[7] = dy;
            state.points[8] = dz;
        }

        this.updateResizeHelperPos();
        this.updateRotationHelperPos();
        this.detachCamera();
        BOTTOM_VIEWS.forEach((view) => this.updateHelperPointsSize(view));
        this.updateCameraFrustumPlane();

        this.dispatchEvent(
            new CustomEvent('canvas.edited', {
                bubbles: false,
                cancelable: true,
                detail: {
                    state,
                    points: [...newPoints],
                    previousPoints,
                },
            }),
        );
        this.render();
    }

    public nudgeCuboid(
        dxLocal: number,
        dyLocal: number,
        dzWorld: number,
        dYawRad: number,
        clientID?: number | null,
    ): boolean {
        let targetClientID: number | null = typeof clientID === 'number' ? clientID : null;
        if (targetClientID === null) {
            targetClientID = this.getActiveElement();
        }

        if (targetClientID === null || !(targetClientID in this.drawnObjects)) {
            return false;
        }

        const { cuboid, data } = this.drawnObjects[targetClientID];
        if (data.lock) return false;

        const state = this.model.objects.find((obj: ObjectState) => obj.clientID === targetClientID);
        if (!state) return false;

        const previousPoints = state.points && state.points.length >= 9 ? [...state.points] : [
            cuboid.perspective.position.x,
            cuboid.perspective.position.y,
            cuboid.perspective.position.z,
            cuboid.perspective.rotation.x,
            cuboid.perspective.rotation.y,
            cuboid.perspective.rotation.z,
            cuboid.perspective.scale.x,
            cuboid.perspective.scale.y,
            cuboid.perspective.scale.z,
            0, 0, 0, 0, 0, 0, 0,
        ];
        if (!this.previousCuboids.has(targetClientID)) {
            this.previousCuboids.set(targetClientID, []);
        }
        this.previousCuboids.get(targetClientID)!.push(previousPoints);

        const [x, y, z, rx, ry, rz] = previousPoints;
        const euler = new THREE.Euler(rx, ry, rz, 'XYZ');

        // Displace position along object's local orientation frame
        const localDelta = new THREE.Vector3(dxLocal, dyLocal, 0).applyEuler(euler);
        const newX = parseFloat((x + localDelta.x).toFixed(4));
        const newY = parseFloat((y + localDelta.y).toFixed(4));
        const newZ = parseFloat((z + dzWorld).toFixed(4));

        let finalRx = rx;
        let finalRy = ry;
        let finalRz = rz;

        if (dYawRad !== 0) {
            let newYaw = rz + dYawRad;
            newYaw = Math.atan2(Math.sin(newYaw), Math.cos(newYaw));
            finalRz = parseFloat(newYaw.toFixed(4));

            // If object is tilted on road plane, conform yaw to terrain slope normal
            const norm = new THREE.Vector3(0, 0, 1).applyEuler(euler).normalize();
            if (norm.z >= 0.70) {
                const fHoriz = new THREE.Vector3(Math.cos(newYaw), Math.sin(newYaw), 0);
                const dot = fHoriz.dot(norm);
                const xLocal = fHoriz.clone().sub(norm.clone().multiplyScalar(dot));
                if (xLocal.lengthSq() > 1e-6) {
                    xLocal.normalize();
                    const yLocal = new THREE.Vector3().crossVectors(norm, xLocal).normalize();
                    const zLocal = norm;
                    const basisMatrix = new THREE.Matrix4().makeBasis(xLocal, yLocal, zLocal);
                    const newEuler = new THREE.Euler().setFromRotationMatrix(basisMatrix, 'XYZ');
                    finalRx = parseFloat(newEuler.x.toFixed(4));
                    finalRy = parseFloat(newEuler.y.toFixed(4));
                    finalRz = Math.cos(newEuler.z - newYaw) > 0.7 ? parseFloat(newEuler.z.toFixed(4)) : finalRz;
                }
            }
        }

        cuboid.setPosition(newX, newY, newZ);
        cuboid.setRotation(finalRx, finalRy, finalRz);
        const newPoints = [...previousPoints];
        newPoints[0] = newX;
        newPoints[1] = newY;
        newPoints[2] = newZ;
        newPoints[3] = finalRx;
        newPoints[4] = finalRy;
        newPoints[5] = finalRz;
        data.points = newPoints;
        if (state.points && state.points.length >= 9) {
            state.points[0] = newX;
            state.points[1] = newY;
            state.points[2] = newZ;
            state.points[3] = finalRx;
            state.points[4] = finalRy;
            state.points[5] = finalRz;
        }

        this.translateReferencePlane(new THREE.Vector3(newX, newY, newZ));
        try {
            this.rotatePlane(null as any, null as any);
        } catch (_) {
            // ignore
        }

        this.updateResizeHelperPos();
        this.updateRotationHelperPos();
        this.detachCamera();
        BOTTOM_VIEWS.forEach((view) => this.updateHelperPointsSize(view));
        this.updateCameraFrustumPlane();

        this.dispatchEvent(
            new CustomEvent('canvas.edited', {
                bubbles: false,
                cancelable: true,
                detail: {
                    state,
                    points: [...newPoints],
                    previousPoints,
                },
            }),
        );

        if (this.isObjectIsolated) {
            this.updateIsolationPlanes();
        }

        this.requestRender();
        return true;
    }

    private buildGroundStrippedGeometry(baseGeo: THREE.BufferGeometry): THREE.BufferGeometry {
        const frameNumber = this.model.data.imageID;
        if (frameNumber !== null && this.geometryCache.has(frameNumber)) {
            const cached = this.geometryCache.get(frameNumber)!;
            if (cached.geometry === baseGeo && cached.strippedGeometry) {
                return cached.strippedGeometry;
            }
        }

        const posAttr = baseGeo.getAttribute('position');
        if (!posAttr || posAttr.count < 3) return baseGeo;
        const positions = posAttr.array as Float32Array;
        const colAttr = baseGeo.getAttribute('color');
        const colors = colAttr ? (colAttr.array as Float32Array) : null;
        const normAttr = baseGeo.getAttribute('normal');
        const normals = normAttr ? (normAttr.array as Float32Array) : null;
        const numPoints = posAttr.count;

        const planeEq = this.cachedGlobalGroundPlane || computeSceneGroundPlane(positions);
        let [gA, gB, gC, gD] = planeEq;
        const pLen = Math.sqrt(gA * gA + gB * gB + gC * gC);
        if (pLen > 1e-6) {
            gA /= pLen; gB /= pLen; gC /= pLen; gD /= pLen;
        } else {
            gA = 0; gB = 0; gC = 1; gD = 1.65;
        }
        if (gC < 0) {
            gA = -gA; gB = -gB; gC = -gC; gD = -gD;
        }
        const safeGC = Math.abs(gC) > 0.25 ? gC : 1.0;

        if (!baseGeo.boundingBox) {
            baseGeo.computeBoundingBox();
        }
        const bbox = baseGeo.boundingBox!;
        const minX = bbox.min.x;
        const minY = bbox.min.y;
        const maxX = Math.max(bbox.max.x, minX + 1);
        const maxY = Math.max(bbox.max.y, minY + 1);

        // 1) Coarse 2.5m terrain grid to estimate local road elevation across slopes, crowns, and hills
        const coarseCell = Math.max(2.5, (maxX - minX) / 240, (maxY - minY) / 240);
        const cCols = Math.min(256, Math.max(1, Math.ceil((maxX - minX) / coarseCell) + 1));
        const cRows = Math.min(256, Math.max(1, Math.ceil((maxY - minY) / coarseCell) + 1));
        const coarseMinZ = new Float32Array(cCols * cRows);
        coarseMinZ.fill(Infinity);

        for (let i = 0; i < numPoints; i++) {
            const i3 = i * 3;
            const x = positions[i3];
            const y = positions[i3 + 1];
            const z = positions[i3 + 2];
            const zPlane = -(gA * x + gB * y + gD) / safeGC;
            const distXY = Math.sqrt(x * x + y * y);
            if (z >= zPlane - 1.5 && z <= zPlane + 2.5 + 0.04 * distXY) {
                const cx = Math.min(cCols - 1, Math.max(0, Math.floor((x - minX) / coarseCell)));
                const cy = Math.min(cRows - 1, Math.max(0, Math.floor((y - minY) / coarseCell)));
                const idx = cy * cCols + cx;
                if (z < coarseMinZ[idx]) {
                    coarseMinZ[idx] = z;
                }
            }
        }

        // Smooth coarse grid with 3x3 ditch-resistant minimum (plus 5x5 fallback for empty cells)
        const smoothedRoadZ = new Float32Array(cCols * cRows);
        for (let cy = 0; cy < cRows; cy++) {
            for (let cx = 0; cx < cCols; cx++) {
                const ownZ = coarseMinZ[cy * cCols + cx];
                let min1 = Infinity;
                let min2 = Infinity;
                for (let dy = -1; dy <= 1; dy++) {
                    const ny = cy + dy;
                    if (ny < 0 || ny >= cRows) continue;
                    for (let dx = -1; dx <= 1; dx++) {
                        const nx = cx + dx;
                        if (nx < 0 || nx >= cCols) continue;
                        const nz = coarseMinZ[ny * cCols + nx];
                        if (nz < min1) {
                            min2 = min1;
                            min1 = nz;
                        } else if (nz < min2) {
                            min2 = nz;
                        }
                    }
                }
                if (ownZ < Infinity) {
                    const refMin = min2 < Infinity ? min2 : min1;
                    smoothedRoadZ[cy * cCols + cx] = Math.min(ownZ, refMin + 0.15);
                } else if (min1 < Infinity) {
                    smoothedRoadZ[cy * cCols + cx] = min1;
                } else {
                    let min5 = Infinity;
                    for (let dy = -2; dy <= 2; dy++) {
                        const ny = cy + dy;
                        if (ny < 0 || ny >= cRows) continue;
                        for (let dx = -2; dx <= 2; dx++) {
                            const nx = cx + dx;
                            if (nx < 0 || nx >= cCols) continue;
                            const nz = coarseMinZ[ny * cCols + nx];
                            if (nz < min5) min5 = nz;
                        }
                    }
                    smoothedRoadZ[cy * cCols + cx] = min5;
                }
            }
        }

        // 2) Fine 1.0m vertical-span grid to distinguish flat pavement/curb cells from cells with vertical objects
        const fineCell = Math.max(1.0, (maxX - minX) / 480, (maxY - minY) / 480);
        const fCols = Math.min(512, Math.max(1, Math.ceil((maxX - minX) / fineCell) + 1));
        const fRows = Math.min(512, Math.max(1, Math.ceil((maxY - minY) / fineCell) + 1));
        const fineMinZ = new Float32Array(fCols * fRows);
        const fineMaxZ = new Float32Array(fCols * fRows);
        fineMinZ.fill(Infinity);
        fineMaxZ.fill(-Infinity);

        for (let i = 0; i < numPoints; i++) {
            const i3 = i * 3;
            const x = positions[i3];
            const y = positions[i3 + 1];
            const z = positions[i3 + 2];
            const cx = Math.min(cCols - 1, Math.max(0, Math.floor((x - minX) / coarseCell)));
            const cy = Math.min(cRows - 1, Math.max(0, Math.floor((y - minY) / coarseCell)));
            let zRoad = smoothedRoadZ[cy * cCols + cx];
            if (zRoad === Infinity) {
                zRoad = -(gA * x + gB * y + gD) / safeGC;
            }
            if (z >= zRoad - 0.35 && z <= zRoad + 2.5) {
                const fx = Math.min(fCols - 1, Math.max(0, Math.floor((x - minX) / fineCell)));
                const fy = Math.min(fRows - 1, Math.max(0, Math.floor((y - minY) / fineCell)));
                const fIdx = fy * fCols + fx;
                if (z < fineMinZ[fIdx]) fineMinZ[fIdx] = z;
                if (z > fineMaxZ[fIdx]) fineMaxZ[fIdx] = z;
            }
        }

        // 3) Collect visible 3D cuboids so points inside annotated object footprints are protected above tire bottoms
        const activeCuboids: {
            px: number; py: number; bottomZ: number;
            uxX: number; uxY: number; uyX: number; uyY: number;
            halfL: number; halfW: number; radiusSq: number;
        }[] = [];
        for (const key of Object.keys(this.drawnObjects)) {
            const obj = this.drawnObjects[+key];
            if (!obj || obj.data.outside || obj.data.hidden || !obj.cuboid?.perspective) continue;
            const { position: pos, rotation: rot, scale } = obj.cuboid.perspective;
            const cosZ = Math.cos(rot.z);
            const sinZ = Math.sin(rot.z);
            const halfL = scale.x / 2 + 0.12;
            const halfW = scale.y / 2 + 0.12;
            activeCuboids.push({
                px: pos.x,
                py: pos.y,
                bottomZ: pos.z - scale.z / 2,
                uxX: cosZ,
                uxY: sinZ,
                uyX: -sinZ,
                uyY: cosZ,
                halfL,
                halfW,
                radiusSq: halfL * halfL + halfW * halfW,
            });
        }

        const grid = this.getGroundGrid();
        const baseOffset = Math.max(this.groundStrippingOffset, 0.18);
        const keepMask = new Uint8Array(numPoints);
        let keptCount = 0;

        for (let i = 0; i < numPoints; i++) {
            const i3 = i * 3;
            const x = positions[i3];
            const y = positions[i3 + 1];
            const z = positions[i3 + 2];

            const cx = Math.min(cCols - 1, Math.max(0, Math.floor((x - minX) / coarseCell)));
            const cy = Math.min(cRows - 1, Math.max(0, Math.floor((y - minY) / coarseCell)));
            let zRoad = smoothedRoadZ[cy * cCols + cx];
            if (zRoad === Infinity) {
                zRoad = -(gA * x + gB * y + gD) / safeGC;
            }
            if (grid) {
                const gElev = grid.getElevation(x, y);
                if (gElev !== null && Math.abs(gElev - zRoad) <= 0.35) {
                    zRoad = Math.max(zRoad, gElev);
                }
            }

            const fx = Math.min(fCols - 1, Math.max(0, Math.floor((x - minX) / fineCell)));
            const fy = Math.min(fRows - 1, Math.max(0, Math.floor((y - minY) / fineCell)));
            const fIdx = fy * fCols + fx;
            const vSpan = fineMaxZ[fIdx] - fineMinZ[fIdx];
            const distXY = Math.sqrt(x * x + y * y);

            // Flat cells without vertical objects (vSpan <= 0.28m) are pure road/curb/sidewalk: strip generously
            let stripCeiling = vSpan <= 0.28 ?
                (zRoad + baseOffset + 0.26 + 0.003 * distXY) :
                (zRoad + baseOffset + 0.04 + 0.002 * distXY);

            for (let c = 0; c < activeCuboids.length; c++) {
                const box = activeCuboids[c];
                const dx = x - box.px;
                const dy = y - box.py;
                if (dx * dx + dy * dy <= box.radiusSq) {
                    const lx = Math.abs(dx * box.uxX + dy * box.uxY);
                    const ly = Math.abs(dx * box.uyX + dy * box.uyY);
                    if (lx <= box.halfL && ly <= box.halfW) {
                        stripCeiling = Math.min(stripCeiling, box.bottomZ + 0.08);
                        break;
                    }
                }
            }

            if (z > stripCeiling) {
                keepMask[i] = 1;
                keptCount++;
            }
        }

        const outPos = new Float32Array(keptCount * 3);
        const outCol = colors ? new Float32Array(keptCount * 3) : null;
        const outNorm = normals ? new Float32Array(keptCount * 3) : null;
        let outIdx = 0;
        for (let i = 0; i < numPoints; i++) {
            if (keepMask[i]) {
                const src = i * 3;
                const dst = outIdx * 3;
                outPos[dst] = positions[src];
                outPos[dst + 1] = positions[src + 1];
                outPos[dst + 2] = positions[src + 2];
                if (outCol && colors) {
                    outCol[dst] = colors[src];
                    outCol[dst + 1] = colors[src + 1];
                    outCol[dst + 2] = colors[src + 2];
                }
                if (outNorm && normals) {
                    outNorm[dst] = normals[src];
                    outNorm[dst + 1] = normals[src + 1];
                    outNorm[dst + 2] = normals[src + 2];
                }
                outIdx++;
            }
        }

        const strippedGeo = new THREE.BufferGeometry();
        strippedGeo.setAttribute('position', new THREE.Float32BufferAttribute(outPos, 3));
        if (outCol) {
            strippedGeo.setAttribute('color', new THREE.Float32BufferAttribute(outCol, 3));
        }
        if (outNorm) {
            strippedGeo.setAttribute('normal', new THREE.Float32BufferAttribute(outNorm, 3));
        }
        if (baseGeo.boundingBox) {
            strippedGeo.boundingBox = baseGeo.boundingBox.clone();
        }
        if (baseGeo.boundingSphere) {
            strippedGeo.boundingSphere = baseGeo.boundingSphere.clone();
        }

        if (frameNumber !== null && this.geometryCache.has(frameNumber)) {
            const cached = this.geometryCache.get(frameNumber)!;
            if (cached.geometry === baseGeo) {
                cached.strippedGeometry = strippedGeo;
            }
        }
        return strippedGeo;
    }

    private buildIsolatedGeometry(
        baseGeo: THREE.BufferGeometry,
        clientID: number,
        stripGround: boolean,
    ): THREE.BufferGeometry {
        const drawn = this.drawnObjects[clientID];
        if (!drawn || !drawn.cuboid?.perspective) {
            return stripGround ? this.buildGroundStrippedGeometry(baseGeo) : baseGeo;
        }

        const posAttr = baseGeo.getAttribute('position');
        if (!posAttr || posAttr.count < 1) return baseGeo;
        const positions = posAttr.array as Float32Array;
        const colAttr = baseGeo.getAttribute('color');
        const colors = colAttr ? (colAttr.array as Float32Array) : null;
        const normAttr = baseGeo.getAttribute('normal');
        const normals = normAttr ? (normAttr.array as Float32Array) : null;
        const numPoints = posAttr.count;

        const { position: pos, rotation: rot, scale } = drawn.cuboid.perspective;
        const euler = new THREE.Euler(rot.x, rot.y, rot.z, 'XYZ');
        const ux = new THREE.Vector3(1, 0, 0).applyEuler(euler).normalize();
        const uy = new THREE.Vector3(0, 1, 0).applyEuler(euler).normalize();
        const uz = new THREE.Vector3(0, 0, 1).applyEuler(euler).normalize();

        const halfL = scale.x / 2 + 1.2;
        const halfW = scale.y / 2 + 1.2;
        const maxLocalZ = scale.z / 2 + 0.8;
        const minLocalZ = stripGround ? (-scale.z / 2 + 0.10) : (-scale.z / 2 - 0.6);
        const maxRadiusSq = halfL * halfL + halfW * halfW + maxLocalZ * maxLocalZ;

        const keepMask = new Uint8Array(numPoints);
        let keptCount = 0;

        for (let i = 0; i < numPoints; i++) {
            const i3 = i * 3;
            const dx = positions[i3] - pos.x;
            const dy = positions[i3 + 1] - pos.y;
            const dz = positions[i3 + 2] - pos.z;
            if (dx * dx + dy * dy + dz * dz <= maxRadiusSq) {
                const lx = dx * ux.x + dy * ux.y + dz * ux.z;
                if (Math.abs(lx) <= halfL) {
                    const ly = dx * uy.x + dy * uy.y + dz * uy.z;
                    if (Math.abs(ly) <= halfW) {
                        const lz = dx * uz.x + dy * uz.y + dz * uz.z;
                        if (lz >= minLocalZ && lz <= maxLocalZ) {
                            keepMask[i] = 1;
                            keptCount++;
                        }
                    }
                }
            }
        }

        const outPos = new Float32Array(keptCount * 3);
        const outCol = colors ? new Float32Array(keptCount * 3) : null;
        const outNorm = normals ? new Float32Array(keptCount * 3) : null;
        let outIdx = 0;
        for (let i = 0; i < numPoints; i++) {
            if (keepMask[i]) {
                const src = i * 3;
                const dst = outIdx * 3;
                outPos[dst] = positions[src];
                outPos[dst + 1] = positions[src + 1];
                outPos[dst + 2] = positions[src + 2];
                if (outCol && colors) {
                    outCol[dst] = colors[src];
                    outCol[dst + 1] = colors[src + 1];
                    outCol[dst + 2] = colors[src + 2];
                }
                if (outNorm && normals) {
                    outNorm[dst] = normals[src];
                    outNorm[dst + 1] = normals[src + 1];
                    outNorm[dst + 2] = normals[src + 2];
                }
                outIdx++;
            }
        }

        const isoGeo = new THREE.BufferGeometry();
        isoGeo.setAttribute('position', new THREE.Float32BufferAttribute(outPos, 3));
        if (outCol) {
            isoGeo.setAttribute('color', new THREE.Float32BufferAttribute(outCol, 3));
        }
        if (outNorm) {
            isoGeo.setAttribute('normal', new THREE.Float32BufferAttribute(outNorm, 3));
        }
        if (baseGeo.boundingBox) {
            isoGeo.boundingBox = baseGeo.boundingBox.clone();
        }
        if (baseGeo.boundingSphere) {
            isoGeo.boundingSphere = baseGeo.boundingSphere.clone();
        }
        return isoGeo;
    }

    private applyPointCloudFilter(): void {
        const baseGeo = this.basePointCloudGeometry;
        if (!baseGeo || this.pointCloudMeshes.length === 0) return;

        if (this.filteredPointCloudGeometry) {
            this.filteredPointCloudGeometry.dispose();
            this.filteredPointCloudGeometry = null;
        }

        let targetGeo: THREE.BufferGeometry = baseGeo;
        if (
            this.isObjectIsolated &&
            this.isolatedClientID !== null &&
            this.drawnObjects[this.isolatedClientID]?.cuboid?.perspective
        ) {
            targetGeo = this.buildIsolatedGeometry(baseGeo, this.isolatedClientID, this.isGroundStripped);
            this.filteredPointCloudGeometry = targetGeo;
        } else if (this.isGroundStripped) {
            targetGeo = this.buildGroundStrippedGeometry(baseGeo);
        }

        this.orthoPointsGeometry = targetGeo;
        for (const mesh of this.pointCloudMeshes) {
            if (mesh) {
                mesh.geometry = targetGeo;
            }
        }
        this.updateClippingPlanes(false);
    }

    private updateClippingPlanes(rebuildGeometry: boolean = true): void {
        if (rebuildGeometry) {
            this.applyPointCloudFilter();
            return;
        }

        const planes: THREE.Plane[] = [];

        if (this.isObjectIsolated && this.isolatedClientID !== null) {
            const drawn = this.drawnObjects[this.isolatedClientID];
            if (drawn && drawn.cuboid) {
                const { cuboid } = drawn;
                const { position: pos, rotation: rot, scale } = cuboid.perspective;

                const euler = new THREE.Euler(rot.x, rot.y, rot.z, 'XYZ');
                const ux = new THREE.Vector3(1, 0, 0).applyEuler(euler).normalize();
                const uy = new THREE.Vector3(0, 1, 0).applyEuler(euler).normalize();
                const uz = new THREE.Vector3(0, 0, 1).applyEuler(euler).normalize();

                const marginHoriz = 1.2;
                const marginVert = 0.8;

                const halfL = scale.x / 2 + marginHoriz;
                const halfW = scale.y / 2 + marginHoriz;
                const halfH = scale.z / 2 + marginVert;

                const pPosX = new THREE.Plane(ux.clone().negate(), ux.dot(pos) + halfL);
                const pNegX = new THREE.Plane(ux.clone(), -ux.dot(pos) + halfL);

                const pPosY = new THREE.Plane(uy.clone().negate(), uy.dot(pos) + halfW);
                const pNegY = new THREE.Plane(uy.clone(), -uy.dot(pos) + halfW);

                const pPosZ = new THREE.Plane(uz.clone().negate(), uz.dot(pos) + halfH);
                const pNegZ = new THREE.Plane(uz.clone(), -uz.dot(pos) + halfH);

                planes.push(pPosX, pNegX, pPosY, pNegY, pPosZ, pNegZ);
            }
        }

        for (const cloud of this.pointCloudMeshes) {
            if (cloud && cloud.material) {
                const mat = cloud.material as THREE.PointsMaterial;
                const prevLen = mat.clippingPlanes ? mat.clippingPlanes.length : 0;
                mat.clippingPlanes = planes;
                if (prevLen !== planes.length) {
                    mat.needsUpdate = true;
                }
            }
        }
    }

    private updateIsolationPlanes(rebuildGeometry: boolean = true): void {
        this.updateClippingPlanes(rebuildGeometry);
    }

    public toggleObjectIsolation(clientID?: number | null): boolean {
        const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
        if (now - this.lastIsolationToggleTime < 80) {
            return this.isObjectIsolated;
        }
        this.lastIsolationToggleTime = now;

        let targetClientID: number | null = typeof clientID === 'number' ? clientID : null;
        if (targetClientID === null) {
            targetClientID = this.getActiveElement();
        }

        if (this.isObjectIsolated) {
            this.isObjectIsolated = false;
            this.isolatedClientID = null;

            for (const id of this.hiddenByIsolation) {
                if (this.drawnObjects[id]?.cuboid?.perspective) {
                    this.drawnObjects[id].cuboid.perspective.visible = true;
                }
            }
            this.hiddenByIsolation.clear();

            this.applyPointCloudFilter();
            this.updateCameraFrustumPlane();
            this.dispatchEvent(new CustomEvent('canvas.isolated', {
                bubbles: false,
                cancelable: true,
                detail: { isolated: false, clientID: null },
            }));
            this.requestRender();
            return false;
        }

        // If no cuboid is currently selected, find the visible cuboid closest to camera lookAt target
        if (targetClientID === null || !(targetClientID in this.drawnObjects)) {
            const lookTarget = this.views.perspective.controls ?
                this.views.perspective.controls.getTarget(new THREE.Vector3()) :
                new THREE.Vector3(0, 0, 0);
            let bestID: number | null = null;
            let bestDistSq = Infinity;
            for (const strId of Object.keys(this.drawnObjects)) {
                const id = +strId;
                const entry = this.drawnObjects[id];
                if (entry && !entry.data.outside && !entry.data.hidden && entry.cuboid?.perspective) {
                    const distSq = entry.cuboid.perspective.position.distanceToSquared(lookTarget);
                    if (distSq < bestDistSq) {
                        bestDistSq = distSq;
                        bestID = id;
                    }
                }
            }
            if (bestID !== null) {
                targetClientID = bestID;
                this.model.data.activeElement.clientID = `${bestID}`;
                this.dispatchEvent(
                    new CustomEvent('canvas.selected', {
                        bubbles: false,
                        cancelable: true,
                        detail: { clientID: bestID },
                    }),
                );
            }
        }

        if (targetClientID === null || !(targetClientID in this.drawnObjects)) {
            this.dispatchEvent(new CustomEvent('canvas.isolated', {
                bubbles: false,
                cancelable: true,
                detail: {
                    isolated: false,
                    clientID: null,
                    warning: 'No 3D objects in the current frame to isolate.',
                },
            }));
            return false;
        }

        this.isObjectIsolated = true;
        this.isolatedClientID = targetClientID;

        this.hiddenByIsolation.clear();
        Object.keys(this.drawnObjects).forEach((strId: string) => {
            const id = +strId;
            if (id !== targetClientID && this.drawnObjects[id]?.cuboid?.perspective?.visible) {
                this.drawnObjects[id].cuboid.perspective.visible = false;
                this.hiddenByIsolation.add(id);
            }
        });

        if (this.drawnObjects[targetClientID]?.cuboid) {
            this.drawnObjects[targetClientID].cuboid.perspective.visible = true;
            this.fitObject(this.drawnObjects[targetClientID].cuboid, false);
        }

        this.applyPointCloudFilter();
        this.updateCameraFrustumPlane();
        this.updateSideViewsZoom();
        this.dispatchEvent(new CustomEvent('canvas.isolated', {
            bubbles: false,
            cancelable: true,
            detail: { isolated: true, clientID: targetClientID },
        }));
        this.requestRender();
        return true;
    }

    public isObjectIsolatedState(): boolean {
        return this.isObjectIsolated;
    }

    public getGroundPlane(): [number, number, number, number] {
        const positions = this.getBasePointCloudPositions();
        const targetClientID = (this.isObjectIsolated && this.isolatedClientID !== null) ?
            this.isolatedClientID : this.getActiveElement();

        if (targetClientID !== null && this.drawnObjects[targetClientID]?.cuboid?.perspective) {
            const { position: pos } = this.drawnObjects[targetClientID].cuboid.perspective;
            const grid = this.getGroundGrid();
            if (grid) {
                const localPlane = grid.getRoadPlane(pos.x, pos.y);
                if (localPlane) {
                    return localPlane;
                }
            }
            if (positions && positions.length >= 9) {
                const localPlane = estimateLocalRoadPlane(pos.x, pos.y, pos.z, positions, 3.5);
                if (localPlane) {
                    return localPlane;
                }
            }
        }

        if (this.cachedGlobalGroundPlane) {
            return this.cachedGlobalGroundPlane;
        }

        if (!positions || positions.length < 9) {
            return [0, 0, 1, 1.65];
        }

        const scenePlane = computeSceneGroundPlane(positions);
        this.cachedGlobalGroundPlane = scenePlane;
        return scenePlane;
    }

    private refreshGroundClippingPlane(): [number, number, number, number] {
        const planeEq = this.getGroundPlane();
        let [A, B, C, D] = planeEq;
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

        if (C < 0) {
            A = -A;
            B = -B;
            C = -C;
            D = -D;
        }

        this.groundClippingPlane = new THREE.Plane(new THREE.Vector3(A, B, C), D - this.groundStrippingOffset);
        return [A, B, C, D];
    }

    public toggleGroundStripping(elevationOffset: number = 0.18): boolean {
        if (this.isGroundStripped) {
            this.isGroundStripped = false;
            this.groundClippingPlane = null;
            this.applyPointCloudFilter();

            this.dispatchEvent(new CustomEvent('canvas.groundStripped', {
                bubbles: false,
                cancelable: true,
                detail: {
                    stripped: false,
                    elevationOffset: this.groundStrippingOffset,
                },
            }));
            this.requestRender();
            return false;
        }

        this.isGroundStripped = true;
        this.groundStrippingOffset = typeof elevationOffset === 'number' && Number.isFinite(elevationOffset) ?
            elevationOffset : 0.18;

        const [, , C, D] = this.refreshGroundClippingPlane();
        this.applyPointCloudFilter();

        const groundZ = C !== 0 ? -D / C : -1.65;
        this.dispatchEvent(new CustomEvent('canvas.groundStripped', {
            bubbles: false,
            cancelable: true,
            detail: {
                stripped: true,
                elevationOffset: this.groundStrippingOffset,
                groundZ,
            },
        }));
        this.requestRender();
        return true;
    }

    public isGroundStrippedState(): boolean {
        return this.isGroundStripped;
    }

    public async snapTrackToRoad(
        jobInstance: any,
        clientID?: number | null,
        onProgress?: (message: string, percent: number) => void,
    ): Promise<{ processedFrames: number }> {
        let targetClientID: number | null = typeof clientID === 'number' ? clientID : null;
        if (targetClientID === null) {
            if (this.activatedElementID !== null) {
                targetClientID = this.activatedElementID;
            } else if (this.model.data.activeElement.clientID !== null) {
                targetClientID = +this.model.data.activeElement.clientID;
            }
        }

        if (targetClientID === null) {
            throw new Error('No 3D object selected.');
        }

        const state = this.model.objects.find((obj: ObjectState) => obj.clientID === targetClientID);
        if (!state) {
            throw new Error(`Object #${targetClientID} not found.`);
        }

        if (state.objectType !== ObjectType.TRACK) {
            this.snapToRoad(targetClientID);
            return { processedFrames: 1 };
        }

        const { first, last } = state.keyframes || {};
        if (first === null || last === null || first === undefined || last === undefined) {
            this.snapToRoad(targetClientID);
            return { processedFrames: 1 };
        }

        const loader = new PCDLoader();
        const snappedFrames: { frame: number; state: ObjectState; points: number[] }[] = [];

        onProgress?.(`Starting track snap across frames ${first} to ${last}...`, 5);

        // Pre-fetch track states to determine motion vectors
        const trackStatesByFrame = new Map<number, ObjectState>();
        for (let f = first; f <= last; f++) {
            try {
                const frameStates = await jobInstance.annotations.get(f, true);
                const trackState = frameStates.find((obj: ObjectState) => obj.clientID === targetClientID);
                if (trackState && !trackState.outside && trackState.points) {
                    trackStatesByFrame.set(f, trackState);
                }
            } catch (err) {
                console.warn(`Could not get annotations for frame ${f}:`, err);
            }
        }

        const sortedFrames = Array.from(trackStatesByFrame.keys()).sort((a, b) => a - b);
        if (sortedFrames.length === 0) {
            throw new Error('No valid track frames found to snap.');
        }

        if (this.drawnObjects[targetClientID]) {
            const { cuboid } = this.drawnObjects[targetClientID];
            const currentPoints = [
                cuboid.perspective.position.x,
                cuboid.perspective.position.y,
                cuboid.perspective.position.z,
                cuboid.perspective.rotation.x,
                cuboid.perspective.rotation.y,
                cuboid.perspective.rotation.z,
                cuboid.perspective.scale.x,
                cuboid.perspective.scale.y,
                cuboid.perspective.scale.z,
                0, 0, 0, 0, 0, 0, 0,
            ];
            if (!this.previousCuboids.has(targetClientID)) {
                this.previousCuboids.set(targetClientID, []);
            }
            this.previousCuboids.get(targetClientID)!.push(currentPoints);
        }

        for (let idx = 0; idx < sortedFrames.length; idx++) {
            const f = sortedFrames[idx];
            const trackState = trackStatesByFrame.get(f)!;
            const currentPct = 10 + Math.round(((idx + 1) / sortedFrames.length) * 75);
            onProgress?.(`Processing frame ${f} of ${last}...`, currentPct);

            let positions: ArrayLike<number> | null = null;
            if (f === this.model.data.imageID && this.pointCloudPoints?.geometry?.attributes?.position?.array) {
                positions = this.pointCloudPoints.geometry.attributes.position.array;
            } else if (this.geometryCache.has(f)) {
                positions = this.geometryCache.get(f)!.geometry.attributes.position?.array ?? null;
            } else {
                try {
                    const frameData = await jobInstance.frames.get(f);
                    if (frameData.deleted) continue;
                    const frameImage = await frameData.data();
                    const arrayBuffer = await frameImage.imageData.arrayBuffer();
                    const pcd = loader.parse(arrayBuffer) as THREE.Points;
                    if (pcd?.geometry?.attributes?.position?.array) {
                        positions = pcd.geometry.attributes.position.array;
                    }
                } catch (err) {
                    console.warn(`Could not load PCD for frame ${f}:`, err);
                    continue;
                }
            }

            if (!positions) continue;

            // Strictly preserve the annotator's existing heading (yaw) on each track frame.
            // In datasets where ego-vehicle is moving, sensor-frame displacement does not match
            // vehicle heading and causes 180-degree heading flips.
            const existingYaw = trackState.points[5];

            try {
                const snapResult = snapCuboidToRoadAndOrient(trackState.points, positions, {
                    autoOrient: false,
                    labelName: trackState.label?.name,
                });
                const snappedPoints = [...snapResult.points];
                snappedPoints[5] = existingYaw;

                snappedFrames.push({
                    frame: f,
                    state: trackState,
                    points: snappedPoints,
                });
            } catch (err) {
                console.warn(`Could not snap frame ${f}:`, err);
            }
        }

        if (snappedFrames.length === 0) {
            throw new Error('No valid track frames found to snap.');
        }

        // Apply 1D Temporal Smoothing to prevent elevation & tilt jitter
        if (snappedFrames.length >= 3) {
            onProgress?.('Smoothing road trajectory across track frames...', 88);
            const zSmoothed = snappedFrames.map((s) => s.points[2]);
            const rxSmoothed = snappedFrames.map((s) => s.points[3]);
            const rySmoothed = snappedFrames.map((s) => s.points[4]);

            for (let i = 1; i < snappedFrames.length - 1; i++) {
                zSmoothed[i] = 0.25 * snappedFrames[i - 1].points[2] +
                    0.5 * snappedFrames[i].points[2] +
                    0.25 * snappedFrames[i + 1].points[2];
                rxSmoothed[i] = 0.25 * snappedFrames[i - 1].points[3] +
                    0.5 * snappedFrames[i].points[3] +
                    0.25 * snappedFrames[i + 1].points[3];
                rySmoothed[i] = 0.25 * snappedFrames[i - 1].points[4] +
                    0.5 * snappedFrames[i].points[4] +
                    0.25 * snappedFrames[i + 1].points[4];
            }

            for (let i = 1; i < snappedFrames.length - 1; i++) {
                snappedFrames[i].points[2] = parseFloat(zSmoothed[i].toFixed(3));
                snappedFrames[i].points[3] = parseFloat(rxSmoothed[i].toFixed(4));
                snappedFrames[i].points[4] = parseFloat(rySmoothed[i].toFixed(4));
            }
        }

        onProgress?.('Saving track updates...', 95);
        for (const item of snappedFrames) {
            item.state.points = item.points;
            item.state.keyframe = true;
        }

        await jobInstance.annotations.saveStates(snappedFrames.map((s) => s.state));

        // If the current active frame was part of the track, update canvas visual
        if (this.drawnObjects[targetClientID]) {
            const currentItem = snappedFrames.find((s) => s.frame === this.model.data.imageID);
            if (currentItem) {
                const { cuboid, data } = this.drawnObjects[targetClientID];
                const [x, y, z, rx, ry, rz, dx, dy, dz] = currentItem.points;
                cuboid.setPosition(x, y, z);
                cuboid.setRotation(rx, ry, rz);
                cuboid.setScale(dx, dy, dz);
                data.points = [...currentItem.points];
                this.requestRender();
            }
        }

        onProgress?.('Track snapping completed.', 100);
        return { processedFrames: snappedFrames.length };
    }

    private action: {
        translation: any;
        resize: {
            status: boolean;
            previousPosition: null | THREE.Vector3;
            helperElement: THREE.Object3D;
        };
        scan: any;
        rotation: any;
        detected: any;
        initialMouseVector: any;
    };

    private cameraSettings: {
        [key in ViewType]: {
            position: [number, number, number],
            lookAt: [number, number, number],
            up: [number, number, number],
        }
    };

    private get selectedCuboid(): CuboidModel | null {
        const { clientID } = this.model.data.activeElement;
        if (clientID !== null) {
            return this.drawnObjects[+clientID].cuboid || null;
        }

        return null;
    }

    private set mode(value: Mode) {
        this.controller.mode = value;
    }

    private get mode(): Mode {
        return this.controller.mode;
    }

    public constructor(model: Canvas3dModel & Master, controller: Canvas3dController) {
        this.controller = controller;
        this.clock = new THREE.Clock();
        this.speed = consts.MOVEMENT_FACTOR;
        this.cube = new CuboidModel('line', '#ffffff');
        this.stateToBeSplitted = null;
        this.statesToBeGrouped = [];
        this.statesToBeMerged = [];
        this.isPerspectiveBeingDragged = false;
        this.activatedElementID = null;
        this.sideViewsZoomMemory = {};
        this.drawnObjects = {};
        this.model = model;
        this.sceneBBox = new THREE.Box3();
        this.cameraSettings = {
            perspective: {
                position: [-15, 0, 8],
                lookAt: [10, 0, 0],
                up: [0, 0, 1],
            },
            top: {
                position: [0, 0, 8],
                lookAt: [0, 0, 0],
                up: [0, 0, 1],
            },
            side: {
                position: [0, -8, 0],
                lookAt: [0, 0, 0],
                up: [0, 0, 1],
            },
            front: {
                position: [8, 0, 0],
                lookAt: [0, 0, 0],
                up: [0, 0, 1],
            },
        };

        this.isCtrlDown = false;
        this.hoverNeedsUpdate = false;
        this.action = {
            scan: null,
            detected: false,
            initialMouseVector: new THREE.Vector2(),
            translation: {
                status: false,
                helper: null,
                coordinates: null,
                offset: new THREE.Vector3(),
                inverseMatrix: new THREE.Matrix4(),
            },
            rotation: {
                status: false,
                helper: null,
                recentMouseVector: new THREE.Vector2(0, 0),
                screenInit: {
                    x: 0,
                    y: 0,
                },
                screenMove: {
                    x: 0,
                    y: 0,
                },
            },
            resize: {
                status: false,
                helperElement: null,
                previousPosition: null,
            },
        };

        this.gpuCapabilities = detectGpuCapabilities();
        this.mainSceneWorker = new MainSceneWorker();

        const perspectiveCanvas = window.document.createElement('canvas');
        perspectiveCanvas.style.width = '100%';
        perspectiveCanvas.style.height = '100%';

        this.mainRenderer = new THREE.WebGLRenderer({
            canvas: perspectiveCanvas,
            antialias: this.gpuCapabilities.antialias,
            powerPreference: 'high-performance',
        });
        this.mainRenderer.localClippingEnabled = true;
        this.mainRenderer.setPixelRatio(this.gpuCapabilities.pixelRatio);
        const origMainSetSize = this.mainRenderer.setSize.bind(this.mainRenderer);
        this.mainRenderer.setSize = (w: number, h: number, updateStyle: boolean = false) => {
            origMainSetSize(w, h, updateStyle);
        };
        this.mainRenderer.dispose = () => {
            this.mainRenderer.renderLists.dispose();
            this.mainRenderer.clear();
        };

        const createOrthoViewRenderer = (): THREE.WebGLRenderer => {
            const canvas = window.document.createElement('canvas');
            canvas.style.width = '100%';
            canvas.style.height = '100%';
            const renderer = new THREE.WebGLRenderer({
                canvas,
                antialias: this.gpuCapabilities.antialias,
                powerPreference: 'high-performance',
            });
            renderer.localClippingEnabled = true;
            renderer.setPixelRatio(this.gpuCapabilities.pixelRatio);
            const origSetSize = renderer.setSize.bind(renderer);
            renderer.setSize = (w: number, h: number, updateStyle: boolean = false) => {
                origSetSize(w, h, updateStyle);
            };
            renderer.dispose = () => {
                renderer.renderLists.dispose();
                renderer.clear();
            };
            return renderer;
        };

        this.views = {
            perspective: {
                renderer: this.mainRenderer,
                scene: new THREE.Scene(),
                rayCaster: {
                    renderer: new THREE.Raycaster(),
                    mouseVector: new THREE.Vector2(),
                },
            },
            top: {
                renderer: createOrthoViewRenderer(),
                scene: new THREE.Scene(),
                rayCaster: {
                    renderer: new THREE.Raycaster(),
                    mouseVector: new THREE.Vector2(),
                },
            },
            side: {
                renderer: createOrthoViewRenderer(),
                scene: new THREE.Scene(),
                rayCaster: {
                    renderer: new THREE.Raycaster(),
                    mouseVector: new THREE.Vector2(),
                },
            },
            front: {
                renderer: createOrthoViewRenderer(),
                scene: new THREE.Scene(),
                rayCaster: {
                    renderer: new THREE.Raycaster(),
                    mouseVector: new THREE.Vector2(),
                },
            },
        };
        CameraControls.install({ THREE });

        const canvasPerspectiveView = this.views.perspective.renderer.domElement;
        const canvasTopView = this.views.top.renderer.domElement;
        const canvasSideView = this.views.side.renderer.domElement;
        const canvasFrontView = this.views.front.renderer.domElement;

        [
            [canvasPerspectiveView, this.views.perspective.scene],
            [canvasTopView, this.views.top.scene],
            [canvasSideView, this.views.side.scene],
            [canvasFrontView, this.views.front.scene],
        ].forEach(([view, scene]) => {
            Object.defineProperty(view, 'scene', {
                value: scene,
                enumerable: false,
                configurable: false,
                writable: false,
            });

            Object.defineProperty(view, 'getDrawnObjects', {
                value: () => Object.values(this.drawnObjects).map((object) => {
                    const { clientID } = object.data;
                    return {
                        ...object,
                        state: this.model.objects.find((_state: ObjectState) => _state.clientID === clientID),
                    };
                }),
                enumerable: false,
                configurable: false,
                writable: false,
            });

            Object.defineProperty(view, 'updatePosition', {
                value: (state: ObjectState, points: number[]) => {
                    this.dispatchEvent(
                        new CustomEvent('canvas.edited', {
                            bubbles: false,
                            cancelable: true,
                            detail: {
                                state,
                                points: [...points],
                            },
                        }),
                    );
                },
                enumerable: false,
                configurable: false,
                writable: false,
            });

            Object.defineProperty(view, 'getPerformanceStats', {
                value: () => this.getPerformanceStats(),
                enumerable: false,
                configurable: false,
                writable: false,
            });

            Object.defineProperty(view, 'togglePerformanceHud', {
                value: (enabled?: boolean) => this.togglePerformanceHud(enabled),
                enumerable: false,
                configurable: false,
                writable: false,
            });
        });

        canvasPerspectiveView.addEventListener('contextmenu', (e: MouseEvent): void => {
            if (this.model.data.activeElement.clientID !== null) {
                this.dispatchEvent(
                    new CustomEvent('canvas.contextmenu', {
                        bubbles: false,
                        cancelable: true,
                        detail: {
                            clientID: Number(this.model.data.activeElement.clientID),
                            clientX: e.clientX,
                            clientY: e.clientY,
                        },
                    }),
                );
            }
            if (this.model.mode === Mode.DRAW && e.ctrlKey && this.model.data.drawData.initialState) {
                const { x, y, z } = this.cube.perspective.position;
                const { x: width, y: height, z: depth } = this.cube.perspective.scale;
                const { x: rotationX, y: rotationY, z: rotationZ } = this.cube.perspective.rotation;
                let points = [x, y, z, rotationX, rotationY, rotationZ, width, height, depth, 0, 0, 0, 0, 0, 0, 0];
                const initState = this.model.data.drawData.initialState;
                if (this.pointCloudPoints?.geometry?.attributes?.position?.array) {
                    const label = initState?.label;
                    const labelName = label?.name;
                    points = fitClusterAtPoint(
                        [x, y, z],
                        this.pointCloudPoints.geometry.attributes.position.array,
                        {
                            labelName,
                            label,
                            attributes: initState?.attributes,
                            defaultDims: this.controller.drawData.initialDimensions,
                            snapGround: this.controller.drawData.snapGround ?? true,
                            autoOrient: this.controller.drawData.autoOrient ?? true,
                        },
                    ).points;
                }
                this.dispatchEvent(
                    new CustomEvent('canvas.drawn', {
                        bubbles: false,
                        cancelable: true,
                        detail: {
                            state: {
                                shapeType: 'cuboid',
                                objectType: initState.objectType,
                                frame: this.model.data.imageID,
                                points,
                                attributes: { ...initState.attributes },
                                group: initState.group?.id || null,
                                label: initState.label,
                            },
                            continue: true,
                            duration: 0,
                        },
                    }),
                );
            }
        });

        canvasPerspectiveView.addEventListener('mousedown', this.onPerspectiveDrag);
        canvasPerspectiveView.addEventListener('mouseup', (e: MouseEvent): void => {
            if (this.mode === Mode.DRAW && e.button === 0) {
                this.finalizeDraw();
            }
        });
        this.onDocumentMouseUp = () => {
            this.disablePerspectiveDragging();
            if (this.isPerspectiveBeingDragged && this.mode !== Mode.DRAG_CANVAS) {
                // call this body only of drag was activated inside the canvas, but not globally
                this.isPerspectiveBeingDragged = false;
            }
        };
        window.document.addEventListener('mouseup', this.onDocumentMouseUp);

        canvasTopView.addEventListener('mousedown', this.startAction.bind(this, 'top'));
        canvasSideView.addEventListener('mousedown', this.startAction.bind(this, 'side'));
        canvasFrontView.addEventListener('mousedown', this.startAction.bind(this, 'front'));

        canvasTopView.addEventListener('mousemove', this.moveAction.bind(this, 'top'));
        canvasSideView.addEventListener('mousemove', this.moveAction.bind(this, 'side'));
        canvasFrontView.addEventListener('mousemove', this.moveAction.bind(this, 'front'));

        canvasTopView.addEventListener('mouseup', this.completeActions.bind(this));
        canvasTopView.addEventListener('mouseleave', this.completeActions.bind(this));
        canvasSideView.addEventListener('mouseup', this.completeActions.bind(this));
        canvasSideView.addEventListener('mouseleave', this.completeActions.bind(this));
        canvasFrontView.addEventListener('mouseup', this.completeActions.bind(this));
        canvasFrontView.addEventListener('mouseleave', this.completeActions.bind(this));

        canvasPerspectiveView.addEventListener('mousemove', (event: MouseEvent): void => {
            event.preventDefault();
            this.isCtrlDown = event.ctrlKey;
            if (this.mode === Mode.DRAG_CANVAS || this.isPerspectiveBeingDragged) return;
            const canvas = this.views.perspective.renderer.domElement;
            const rect = canvas.getBoundingClientRect();
            const { mouseVector } = this.views.perspective.rayCaster as { mouseVector: THREE.Vector2 };
            mouseVector.x = ((event.clientX - (canvas.offsetLeft + rect.left)) / canvas.clientWidth) * 2 - 1;
            mouseVector.y = -((event.clientY - (canvas.offsetTop + rect.top)) / canvas.clientHeight) * 2 + 1;
            this.hoverNeedsUpdate = true;
        });

        canvasPerspectiveView.addEventListener('click', (e: MouseEvent): void => {
            e.preventDefault();
            const selectionIsBlocked = ![Mode.GROUP, Mode.MERGE, Mode.SPLIT, Mode.IDLE].includes(this.mode) ||
                !this.views.perspective.rayCaster ||
                this.isPerspectiveBeingDragged;

            if (e.detail !== 1 || selectionIsBlocked) return;
            const intersects = this.views.perspective.rayCaster.renderer
                .intersectObjects(this.getAllVisibleCuboids(), false);
            const intersectionClientID = +(intersects[0]?.object?.name) || null;
            const objectState = Number.isInteger(intersectionClientID) ? this.model.objects
                .find((state: ObjectState) => state.clientID === intersectionClientID) : null;

            if (objectState) {
                this.dispatchEvent(
                    new CustomEvent('canvas.clicked', {
                        bubbles: false,
                        cancelable: true,
                        detail: {
                            clientID: intersectionClientID,
                        },
                    }),
                );
            }

            const handleClick = (targetList: ObjectState[]): void => {
                const objectStateIdx = targetList
                    .findIndex((state: ObjectState) => state.clientID === intersectionClientID);
                if (objectStateIdx !== -1) {
                    targetList.splice(objectStateIdx, 1);
                } else {
                    targetList.push(objectState);
                }

                this.drawnObjects[intersectionClientID].cuboid.setColor(this.receiveShapeColor(objectState));
            };

            if (objectState && this.mode === Mode.GROUP) {
                handleClick(this.statesToBeGrouped);
            } else if (objectState && this.mode === Mode.MERGE) {
                const [latest] = this.statesToBeMerged;
                const drawnStates = Object.keys(this.drawnObjects).map((key: string): number => +key);
                if (!latest ||
                    (latest &&
                        objectState.label.id === latest.label.id &&
                        objectState.shapeType === latest.shapeType &&
                        !this.statesToBeMerged.some((state) => drawnStates.includes(state.clientID)))
                ) {
                    handleClick(this.statesToBeMerged);
                }
            } else if (objectState?.objectType === ObjectType.TRACK && this.mode === Mode.SPLIT) {
                this.onSplitDone(objectState);
            } else if (this.mode === Mode.IDLE) {
                const intersectedClientID = intersects[0]?.object?.name || null;
                const nextID = typeof intersectedClientID === 'string' && intersectedClientID !== '' ? +intersectedClientID : null;
                this.dispatchEvent(
                    new CustomEvent('canvas.selected', {
                        bubbles: false,
                        cancelable: true,
                        detail: {
                            clientID: nextID,
                        },
                    }),
                );
            }
        });

        canvasPerspectiveView.addEventListener('dblclick', (e: MouseEvent): void => {
            e.preventDefault();
            if (this.mode !== Mode.DRAW) {
                const { perspective: viewType } = this.views;
                viewType.rayCaster.renderer.setFromCamera(viewType.rayCaster.mouseVector, viewType.camera);
                const intersects = viewType.rayCaster.renderer.intersectObjects(this.getAllVisibleCuboids(), false);
                if (!intersects.length) {
                    this.fitCanvas(true);
                } else {
                    const clientID = intersects[0].object.name;
                    this.fitObject(this.drawnObjects[clientID].cuboid, true);

                    this.dispatchEvent(
                        new CustomEvent('canvas.doubleclicked', {
                            bubbles: false,
                            cancelable: true,
                            detail: {
                                clientID: Number(clientID),
                            },
                        }),
                    );
                }
                return;
            }

            this.finalizeDraw();
        });

        this.mode = Mode.IDLE;

        Object.keys(this.views).forEach((view: string): void => {
            this.views[view as keyof Views].scene.background = new THREE.Color(0x000000);
        });

        const viewSize = consts.ORTHO_FRUSTUM_HEIGHT;
        const height = window.innerHeight;
        const width = window.innerWidth;
        const aspectRatio = window.innerWidth / window.innerHeight;

        // setting up the camera and adding it in the scene
        this.views.perspective.camera = new THREE.PerspectiveCamera(50, aspectRatio, 0.1, 500);
        this.views.top.camera = new THREE.OrthographicCamera(
            (-aspectRatio * viewSize) / 2 - 2,
            (aspectRatio * viewSize) / 2 + 2,
            viewSize / 2 + 2,
            -viewSize / 2 - 2,
            -50,
            50,
        );
        this.views.side.camera = new THREE.OrthographicCamera(
            (-aspectRatio * viewSize) / 2,
            (aspectRatio * viewSize) / 2,
            viewSize / 2,
            -viewSize / 2,
            -50,
            50,
        );
        this.views.front.camera = new THREE.OrthographicCamera(
            (-aspectRatio * viewSize) / 2,
            (aspectRatio * viewSize) / 2,
            viewSize / 2,
            -viewSize / 2,
            -50,
            50,
        );

        for (const cameraType of ALL_VIEWS) {
            this.views[cameraType].camera.position.set(...this.cameraSettings[cameraType].position);
            this.views[cameraType].camera.up.set(...this.cameraSettings[cameraType].up);
            this.views[cameraType].camera.lookAt(...this.cameraSettings[cameraType].lookAt);
            this.views[cameraType].camera.name = `camera${cameraType[0].toUpperCase()}${cameraType.slice(1)}`;
        }

        Object.keys(this.views).forEach((view: string): void => {
            const viewType = this.views[view as keyof Views];
            if (viewType.camera) {
                viewType.renderer.setSize(width, height, false);
                if (view !== ViewType.PERSPECTIVE) {
                    viewType.controls = new CameraControls(viewType.camera, viewType.renderer.domElement);
                    viewType.controls.mouseButtons.left = CameraControls.ACTION.NONE;
                    viewType.controls.mouseButtons.right = CameraControls.ACTION.NONE;
                } else {
                    viewType.controls = new CameraControls(viewType.camera, viewType.renderer.domElement);
                    viewType.controls.mouseButtons.left = CameraControls.ACTION.NONE;
                    viewType.controls.mouseButtons.right = CameraControls.ACTION.NONE;
                    viewType.controls.touches.one = CameraControls.ACTION.NONE;
                    viewType.controls.touches.two = CameraControls.ACTION.NONE;
                    viewType.controls.touches.three = CameraControls.ACTION.NONE;
                }
                viewType.controls.minDistance = consts.MIN_DISTANCE;
                viewType.controls.maxDistance = consts.MAX_DISTANCE;
            }
        });
        this.views.top.controls.enabled = false;
        this.views.side.controls.enabled = false;
        this.views.front.controls.enabled = false;

        BOTTOM_VIEWS.forEach((view: ViewType): void => {
            this.views[view].renderer.domElement.addEventListener(
                'wheel',
                (event: WheelEvent): void => {
                    event.preventDefault();
                    const { camera } = this.views[view];

                    // Adaptive zoom algorithm from 2D canvas
                    const basicZoomCoef = 6 / 5;
                    const adjustCoef = 1 / 100;
                    const scaleFactor = basicZoomCoef ** (-event.deltaY * adjustCoef);
                    camera.zoom = Math.min(
                        Math.max(
                            camera.zoom * scaleFactor,
                            consts.SIDE_VIEWS_MIN_ZOOM,
                        ), consts.SIDE_VIEWS_MAX_ZOOM,
                    );
                    camera.updateProjectionMatrix();

                    if (this.activatedElementID) {
                        // try to remember applied zoom level per each object
                        // also save serverID to be able to track case when objects are reloaded
                        // (e.g. annotations uploaded), in this rare case zoom memory should be reset
                        if (!this.sideViewsZoomMemory[this.activatedElementID]) {
                            this.sideViewsZoomMemory[this.activatedElementID] = {
                                serverID: null,
                                top: null,
                                side: null,
                                front: null,
                            };
                        }

                        this.sideViewsZoomMemory[this.activatedElementID].serverID =
                            this.drawnObjects[this.activatedElementID].data.serverID;
                        this.sideViewsZoomMemory[this.activatedElementID][view] = camera.zoom;
                    }

                    // after changing zoom, need to recalculate control points size
                    this.updateHelperPointsSize(view);
                    this.requestOrthoRender(2);
                },
                { passive: false },
            );
        });

        model.subscribe(this);
    }

    private finalizeDraw(): void {
        if (this.mode !== Mode.DRAW) return;

        this.controller.drawData.enabled = false;
        this.mode = Mode.IDLE;
        const { x, y, z } = this.cube.perspective.position;
        const { x: width, y: height, z: depth } = this.cube.perspective.scale;
        const { x: rotationX, y: rotationY, z: rotationZ } = this.cube.perspective.rotation;
        let points = [x, y, z, rotationX, rotationY, rotationZ, width, height, depth, 0, 0, 0, 0, 0, 0, 0];
        const initState = this.model.data.drawData.initialState;
        const { redraw } = this.model.data.drawData;
        if (typeof redraw === 'number') {
            const state = this.model.objects
                .find((object: ObjectState): boolean => object.clientID === redraw);
            const { cuboid } = this.drawnObjects[redraw];
            cuboid.perspective.visible = true;

            this.dispatchEvent(
                new CustomEvent('canvas.edited', {
                    bubbles: false,
                    cancelable: true,
                    detail: {
                        state,
                        points,
                    },
                }),
            );
        } else {
            if (this.pointCloudPoints?.geometry?.attributes?.position?.array) {
                const label = initState?.label;
                const labelName = label?.name;
                const fitted = fitClusterAtPoint(
                    [x, y, z],
                    this.pointCloudPoints.geometry.attributes.position.array,
                    {
                        labelName,
                        label,
                        attributes: initState?.attributes,
                        defaultDims: this.controller.drawData.initialDimensions,
                        snapGround: this.controller.drawData.snapGround ?? true,
                        autoOrient: this.controller.drawData.autoOrient ?? true,
                    },
                );
                points = fitted.points;
            }

            this.dispatchEvent(
                new CustomEvent('canvas.drawn', {
                    bubbles: false,
                    cancelable: true,
                    detail: {
                        state: {
                            shapeType: 'cuboid',
                            frame: this.model.data.imageID,
                            points,
                            ...(initState ? {
                                attributes: { ...initState.attributes },
                                group: initState.group?.id || null,
                                label: initState.label,
                                shapeType: initState.shapeType,
                                objectType: initState.objectType,
                            } : {}),
                        },
                        duration: 0,
                    },
                }),
            );
        }

        this.views[ViewType.PERSPECTIVE].scene.children[0].remove(this.cube.perspective);
        this.dispatchEvent(new CustomEvent('canvas.canceled'));
    }

    private fitObject(object: CuboidModel, animate: boolean = false): void {
        const camera = this.views.perspective.camera as THREE.PerspectiveCamera;
        const mesh = object.perspective as THREE.Mesh;
        const geom = mesh.geometry as THREE.BufferGeometry;
        if (!geom.boundingBox) {
            geom.computeBoundingBox();
        }

        // 1) Collect 8 corners of the local bounding box (base geometry 0.5 cube)
        const bb = geom.boundingBox!;
        const cornersLocal: THREE.Vector3[] = [
            new THREE.Vector3(bb.min.x, bb.min.y, bb.min.z),
            new THREE.Vector3(bb.min.x, bb.min.y, bb.max.z),
            new THREE.Vector3(bb.min.x, bb.max.y, bb.min.z),
            new THREE.Vector3(bb.min.x, bb.max.y, bb.max.z),
            new THREE.Vector3(bb.max.x, bb.min.y, bb.min.z),
            new THREE.Vector3(bb.max.x, bb.min.y, bb.max.z),
            new THREE.Vector3(bb.max.x, bb.max.y, bb.min.z),
            new THREE.Vector3(bb.max.x, bb.max.y, bb.max.z),
        ];

        // 2) Transform corners to world space (includes scale/rotation/translation)
        const worldCorners: THREE.Vector3[] = cornersLocal.map((v) => mesh.localToWorld(v.clone()));

        // 3) Build camera basis in world
        const forward = new THREE.Vector3();
        camera.getWorldDirection(forward).normalize();
        const up = camera.up.clone().normalize();
        const right = new THREE.Vector3().crossVectors(forward, up).normalize();
        up.copy(new THREE.Vector3().crossVectors(right, forward).normalize());

        // 4) Project corners onto right/up to get world extents independent of current distance
        let minU = +Infinity;
        let maxU = -Infinity;
        let minV = +Infinity;
        let maxV = -Infinity;
        for (const c of worldCorners) {
            const u = c.dot(right);
            const v = c.dot(up);
            if (u < minU) minU = u;
            if (u > maxU) maxU = u;
            if (v < minV) minV = v;
            if (v > maxV) maxV = v;
        }
        const widthWorld = maxU - minU;
        const heightWorld = maxV - minV;

        // 5) Compute required distance to fit both width and height within FOV and aspect
        const vFov = THREE.MathUtils.degToRad(camera.fov);
        const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect);
        const distByHeight = (heightWorld / 2) / Math.tan(vFov / 2);
        const distByWidth = (widthWorld / 2) / Math.tan(hFov / 2);
        const distance = Math.max(distByHeight, distByWidth);

        // 6) Center of geometry box in world
        const centerLocal = bb.getCenter(new THREE.Vector3());
        const center = mesh.localToWorld(centerLocal);

        // 7) Move along forward and look at center
        const margin = 1.5;
        const targetPosition = center.clone().sub(forward.multiplyScalar(distance * margin));
        this.views.perspective.controls.setLookAt(
            targetPosition.x,
            targetPosition.y,
            targetPosition.z,
            center.x,
            center.y,
            center.z,
            animate,
        );
    }

    private fitCanvas(animation: boolean): void {
        const [x, y, z] = this.cameraSettings.perspective.lookAt;
        this.positionAllViews(x, y, z, animation);
        this.updateCameraFrustumPlane();
    }

    private getAllVisibleCuboids(view: ViewType = ViewType.PERSPECTIVE): THREE.Mesh[] {
        return Object.values(this.drawnObjects)
            .map(({ cuboid }) => cuboid[view]).filter((mesh: THREE.Mesh) => mesh.visible);
    }

    private updateCameraFrustumPlane(viewType?: ViewType): void {
        const setCameraFrustumPlane = (
            camera: THREE.OrthographicCamera,
            center: THREE.Vector3, dimensions: THREE.Vector3,
            view: ViewType,
        ): void => {
            const [width, length] = dimensions.toArray();
            const [cx, cy, cz] = center.toArray();
            const distanceUpToCamera = Math.sqrt(
                (camera.position.x - cx) ** 2 +
                (camera.position.y - cy) ** 2 +
                (camera.position.z - cz) ** 2,
            );

            const MARGIN = 0.1;
            if (view === ViewType.FRONT) {
                const objectOffset = Math.min(1 - ((width / 2) / distanceUpToCamera));
                camera.near = distanceUpToCamera * objectOffset - MARGIN;
                camera.far = camera.near + width + MARGIN * 2;
            } else if (view === ViewType.SIDE) {
                const objectOffset = Math.min(1, 1 - ((length / 2) / distanceUpToCamera));
                camera.near = distanceUpToCamera * objectOffset - MARGIN;
                camera.far = camera.near + length + MARGIN * 2;
            } else if (view === ViewType.TOP) {
                if (this.isObjectIsolated && this.selectedCuboid) {
                    const height = dimensions.z;
                    const objectOffset = Math.min(1, 1 - ((height / 2) / distanceUpToCamera));
                    camera.near = Math.max(0.01, distanceUpToCamera * objectOffset - 0.5);
                    camera.far = camera.near + height + 1.0;
                } else if (this.selectedCuboid) {
                    const height = dimensions.z;
                    const objectOffset = Math.min(1, 1 - ((height / 2) / distanceUpToCamera));
                    camera.near = Math.max(0.01, distanceUpToCamera * objectOffset - 1.5);
                    camera.far = camera.near + height + 3.0;
                } else {
                    camera.near = 0;
                    camera.far = 1000;
                }
            }
            camera.updateProjectionMatrix();
        };

        const { selectedCuboid } = this;
        const sceneCenter = this.sceneBBox.getCenter(new THREE.Vector3());
        const sceneDimensions = this.sceneBBox.max.clone().sub(this.sceneBBox.min);

        let center = sceneCenter;
        let dimensions = sceneDimensions;

        if (selectedCuboid) {
            center = selectedCuboid.perspective.position.clone();
            dimensions = selectedCuboid.perspective.scale.clone();
        }

        if (viewType !== ViewType.FRONT) {
            setCameraFrustumPlane(
                this.views.front.camera as THREE.OrthographicCamera, center, dimensions, ViewType.FRONT,
            );
        }

        if (viewType !== ViewType.TOP) {
            setCameraFrustumPlane(
                this.views.top.camera as THREE.OrthographicCamera, center, dimensions, ViewType.TOP,
            );
        }

        if (viewType !== ViewType.SIDE) {
            setCameraFrustumPlane(
                this.views.side.camera as THREE.OrthographicCamera, center, dimensions, ViewType.SIDE,
            );
        }

        if (this.isObjectIsolated) {
            this.updateIsolationPlanes();
        }
    }

    private updateSideViewsZoom(): void {
        if (this.activatedElementID !== null) {
            const { selectedCuboid } = this;
            const { top, front, side } = this.views;

            let memoizedZoom = this.sideViewsZoomMemory[this.activatedElementID];
            const drawnState = this.drawnObjects[this.activatedElementID];
            if (memoizedZoom &&
                drawnState &&
                memoizedZoom.serverID !== null &&
                memoizedZoom.serverID !== drawnState.data.serverID
            ) {
                // oops, it seems that objects were reloaded, reset zoom memory
                this.sideViewsZoomMemory = {};
                memoizedZoom = undefined;
            }

            // Padding in pixels around the object comes from cvat-ui
            const padding = this.model.data.configuration.focusedObjectPadding;
            const baseline = consts.ORTHO_FRUSTUM_HEIGHT;

            // Helper function to compute zoom that fits width/height with corresponding padding
            const computeZoomWithMargin = (
                objWidthWorld: number,
                objHeightWorld: number,
                canvasWidthPx: number,
                canvasHeightPx: number,
                aspect: number,
            ): number => {
                // fallback if padding is too high for the current canvas size
                const marginPx = Math.max(0, Math.min(padding, Math.min(canvasWidthPx, canvasHeightPx) / 4));

                const heightFactor = (canvasHeightPx - 2 * marginPx) / canvasHeightPx;
                const widthFactor = (canvasWidthPx - 2 * marginPx) / canvasWidthPx;
                const zoomFromHeight = (baseline * heightFactor) / Math.max(objHeightWorld, Number.EPSILON);
                const zoomFromWidth = (aspect * baseline * widthFactor) / Math.max(objWidthWorld, Number.EPSILON);

                return Math.min(zoomFromHeight, zoomFromWidth);
            };

            const defaultOrMemoized = (
                computed: number,
                memoized: number | null,
            ): number => (memoized !== null ? memoized : Math.max(
                consts.SIDE_VIEWS_MIN_ZOOM,
                Math.min(computed, consts.SIDE_VIEWS_MAX_ZOOM),
            ));

            const { renderer: { domElement: canvasTop }, camera: cameraTop } = top;
            const { renderer: { domElement: canvasFront }, camera: cameraFront } = front;
            const { renderer: { domElement: canvasSide }, camera: cameraSide } = side;

            // TOP view: X (width), Y (height)
            // considering bbox geometry always equal 1, we only need its scale
            const zoomTopDefault = computeZoomWithMargin(
                selectedCuboid.top.scale.x,
                selectedCuboid.top.scale.y,
                canvasTop.offsetWidth,
                canvasTop.offsetHeight,
                top.renderer.domElement.width / Math.max(top.renderer.domElement.height, 1),
            );

            cameraTop.zoom = defaultOrMemoized(zoomTopDefault, memoizedZoom?.top ?? null);
            cameraTop.updateProjectionMatrix();
            cameraTop.updateMatrix();
            this.updateHelperPointsSize(ViewType.TOP);

            // FRONT view: Y (width), Z (height)
            const zoomFrontDefault = computeZoomWithMargin(
                selectedCuboid.front.scale.y,
                selectedCuboid.front.scale.z,
                canvasFront.offsetWidth,
                canvasFront.offsetHeight,
                front.renderer.domElement.width / Math.max(front.renderer.domElement.height, 1),
            );
            cameraFront.zoom = defaultOrMemoized(zoomFrontDefault, memoizedZoom?.front ?? null);
            cameraFront.updateProjectionMatrix();
            cameraFront.updateMatrix();
            this.updateHelperPointsSize(ViewType.FRONT);

            // SIDE view: X (width), Z (height)
            const zoomSideDefault = computeZoomWithMargin(
                selectedCuboid.side.scale.x,
                selectedCuboid.side.scale.z,
                canvasSide.offsetWidth,
                canvasSide.offsetHeight,
                side.renderer.domElement.width / Math.max(side.renderer.domElement.height, 1),
            );
            cameraSide.zoom = defaultOrMemoized(zoomSideDefault, memoizedZoom?.side ?? null);
            cameraSide.updateProjectionMatrix();
            cameraSide.updateMatrix();
            this.updateHelperPointsSize(ViewType.SIDE);
        }
    }

    private enablePerspectiveDragging(): void {
        const { controls } = this.views.perspective;
        controls.mouseButtons.left = CameraControls.ACTION.ROTATE;
        controls.mouseButtons.right = CameraControls.ACTION.TRUCK;
        controls.touches.one = CameraControls.ACTION.TOUCH_ROTATE;
        controls.touches.two = CameraControls.ACTION.TOUCH_DOLLY_TRUCK;
        controls.touches.three = CameraControls.ACTION.TOUCH_TRUCK;
    }

    private disablePerspectiveDragging(): void {
        const { controls } = this.views.perspective;
        controls.mouseButtons.left = CameraControls.ACTION.NONE;
        controls.mouseButtons.right = CameraControls.ACTION.NONE;
        controls.touches.one = CameraControls.ACTION.NONE;
        controls.touches.two = CameraControls.ACTION.NONE;
        controls.touches.three = CameraControls.ACTION.NONE;
    }

    private onPerspectiveDrag = (): void => {
        if (![Mode.DRAG_CANVAS, Mode.IDLE].includes(this.mode)) return;
        this.isPerspectiveBeingDragged = true;
        this.enablePerspectiveDragging();
    };

    private onError = (exception: unknown, domain?: string): void => {
        this.dispatchEvent(
            new CustomEvent('canvas.error', {
                bubbles: false,
                cancelable: true,
                detail: {
                    domain,
                    exception: exception instanceof Error ?
                        exception : new Error(`Unknown exception: "${exception}"`),
                },
            }),
        );
    };

    private startAction(view: any, event: MouseEvent): void {
        const { clientID } = this.model.data.activeElement;
        if (event.detail !== 1 || this.mode !== Mode.IDLE || clientID === null || !(clientID in this.drawnObjects)) {
            return;
        }

        const canvas = this.views[view as keyof Views].renderer.domElement;
        const rect = canvas.getBoundingClientRect();
        const { mouseVector } = this.views[view as keyof Views].rayCaster as { mouseVector: THREE.Vector2 };
        const diffX = event.clientX - rect.left;
        const diffY = event.clientY - rect.top;
        mouseVector.x = (diffX / canvas.clientWidth) * 2 - 1;
        mouseVector.y = -(diffY / canvas.clientHeight) * 2 + 1;
        this.action.rotation.screenInit = { x: diffX, y: diffY };
        this.action.rotation.screenMove = { x: diffX, y: diffY };
        const { data } = this.drawnObjects[+clientID];

        if (!data.lock) {
            this.action.scan = view;
            this.model.mode = Mode.EDIT;
            this.requestRender();
        }
    }

    private moveAction(view: any, event: MouseEvent): void {
        event.preventDefault();
        const { clientID } = this.model.data.activeElement;
        if (this.model.mode === Mode.DRAG_CANVAS || clientID === null) {
            return;
        }

        const canvas = this.views[view as keyof Views].renderer.domElement;
        const rect = canvas.getBoundingClientRect();
        const { mouseVector } = this.views[view as keyof Views].rayCaster as { mouseVector: THREE.Vector2 };
        const diffX = event.clientX - rect.left;
        const diffY = event.clientY - rect.top;
        const clientW = canvas.clientWidth || 1;
        const clientH = canvas.clientHeight || 1;
        mouseVector.x = (diffX / clientW) * 2 - 1;
        mouseVector.y = -(diffY / clientH) * 2 + 1;
        this.action.rotation.screenMove = { x: diffX, y: diffY };
        if (this.action.scan !== null) {
            this.requestRender();
        }
    }

    private translateReferencePlane(coordinates: any): void {
        const topPlane = this.scenePlanes[ViewType.TOP] || this.views.top.scene.getObjectByName(Planes.TOP);
        if (topPlane) {
            topPlane.position.x = coordinates.x;
            topPlane.position.y = coordinates.y;
            topPlane.position.z = coordinates.z;
        }
        const sidePlane = this.scenePlanes[ViewType.SIDE] || this.views.side.scene.getObjectByName(Planes.SIDE);
        if (sidePlane) {
            sidePlane.position.x = coordinates.x;
            sidePlane.position.y = coordinates.y;
            sidePlane.position.z = coordinates.z;
        }
        const frontPlane = this.scenePlanes[ViewType.FRONT] || this.views.front.scene.getObjectByName(Planes.FRONT);
        if (frontPlane) {
            frontPlane.position.x = coordinates.x;
            frontPlane.position.y = coordinates.y;
            frontPlane.position.z = coordinates.z;
        }
    }

    private resetActions(): void {
        this.action = {
            ...this.action,
            scan: null,
            detected: false,
            translation: {
                status: false,
                helper: null,
            },
            rotation: {
                status: false,
                helper: null,
                recentMouseVector: new THREE.Vector2(0, 0),
            },
            resize: {
                ...this.action.resize,
                status: false,
                helperElement: null,
                previousPosition: null,
            },
        };
        this.model.mode = Mode.IDLE;
    }

    private completeActions(): void {
        const { scan, detected } = this.action;
        if (this.model.data.activeElement.clientID === null) return;
        if (!detected) {
            this.resetActions();
            return;
        }

        const isTranslation = Boolean(this.action.translation.status);
        const targetClientID = Number(this.selectedCuboid[scan].name);
        const autoRoadSnap = this.model.data.configuration.autoRoadSnap ?? true;

        let { x, y, z } = this.selectedCuboid[scan].position;
        const { x: width, y: height, z: depth } = this.selectedCuboid[scan].scale;
        const { x: rotationX, y: rotationY, z: rotationZ } = this.selectedCuboid[scan].rotation;
        let points = [x, y, z, rotationX, rotationY, rotationZ, width, height, depth, 0, 0, 0, 0, 0, 0, 0];
        const [state] = this.model.objects.filter(
            (_state: any): boolean => _state.clientID === targetClientID,
        );

        if (isTranslation && autoRoadSnap && this.pointCloudPoints?.geometry?.attributes?.position?.array) {
            const positions = this.pointCloudPoints.geometry.attributes.position.array;
            try {
                const snapResult = snapCuboidToRoadAndOrient(points, positions, {
                    labelName: state?.label?.name,
                    autoOrient: false,
                    snapGround: true,
                });
                points = [...snapResult.points];
                const [sx, sy, sz, srx, sry, srz, sdx, sdy, sdz] = points;
                this.selectedCuboid.setPosition(sx, sy, sz);
                this.selectedCuboid.setRotation(srx, sry, srz);
                this.selectedCuboid.setScale(sdx, sdy, sdz);
                if (targetClientID in this.drawnObjects) {
                    this.drawnObjects[targetClientID].data.points = [...points];
                }
                x = sx;
                y = sy;
                z = sz;
            } catch (err) {
                console.warn('Auto road snap on translation failed:', err);
            }
        }

        this.dispatchEvent(
            new CustomEvent('canvas.edited', {
                bubbles: false,
                cancelable: true,
                detail: {
                    state,
                    points,
                },
            }),
        );

        // this.adjustPerspectiveCameras();
        this.translateReferencePlane(new THREE.Vector3(x, y, z));
        this.resetActions();
    }

    private onGroupDone(objects?: any[], duration?: number): void {
        if (objects && objects.length !== 0) {
            this.dispatchEvent(
                new CustomEvent('canvas.grouped', {
                    bubbles: false,
                    cancelable: true,
                    detail: {
                        duration,
                        states: objects,
                    },
                }),
            );
        } else {
            this.dispatchEvent(
                new CustomEvent('canvas.canceled', {
                    bubbles: false,
                    cancelable: true,
                }),
            );
        }

        this.mode = Mode.IDLE;
    }

    private onMergeDone(objects: any[] | null, duration?: number): void {
        if (objects) {
            const event: CustomEvent = new CustomEvent('canvas.merged', {
                bubbles: false,
                cancelable: true,
                detail: {
                    duration,
                    states: objects,
                },
            });

            this.dispatchEvent(event);
        } else {
            const event: CustomEvent = new CustomEvent('canvas.canceled', {
                bubbles: false,
                cancelable: true,
            });

            this.dispatchEvent(event);
        }

        this.mode = Mode.IDLE;
    }

    private onSplitDone(object: ObjectState, duration?: number): void {
        if (object) {
            const event: CustomEvent = new CustomEvent('canvas.splitted', {
                bubbles: false,
                cancelable: true,
                detail: {
                    duration,
                    state: object,
                    frame: object.frame,
                },
            });

            this.dispatchEvent(event);
        } else {
            const event: CustomEvent = new CustomEvent('canvas.canceled', {
                bubbles: false,
                cancelable: true,
            });

            this.dispatchEvent(event);
        }

        this.controller.split({ enabled: false });
        this.mode = Mode.IDLE;
    }

    private receiveShapeColor(state: ObjectState | DrawnObjectData): string {
        const includedInto = (states: ObjectState[]): boolean => states
            .some((_state: ObjectState): boolean => _state.clientID === state.clientID);
        const { colorBy } = this.model.data.configuration;

        if (this.mode === Mode.GROUP && includedInto(this.statesToBeGrouped)) {
            return consts.GROUPING_COLOR;
        }

        if (this.mode === Mode.MERGE && includedInto(this.statesToBeMerged)) {
            return consts.MERGING_COLOR;
        }

        if (this.mode === Mode.SPLIT && this.stateToBeSplitted?.clientID === state.clientID) {
            return consts.SPLITTING_COLOR;
        }

        if (state instanceof ObjectState) {
            if (colorBy === 'Label') {
                return state.label.color;
            }

            if (colorBy === 'Group') {
                return state.group?.color || consts.DEFAULT_GROUP_COLOR;
            }

            return state.color;
        }

        if (colorBy === 'Label') {
            return state.labelColor;
        }

        if (colorBy === 'Group') {
            return state.groupColor;
        }

        return state.color;
    }

    private addCuboid(state: ObjectState): CuboidModel {
        const {
            shapeOpacity, outlinedBorders, orientationVisibility,
        } = this.model.data.configuration;
        const clientID = String(state.clientID);
        const cuboid = new CuboidModel(state.occluded ? 'dashed' : 'line', outlinedBorders || '#ffffff');
        const color = this.receiveShapeColor(state);

        cuboid.setName(clientID);
        cuboid.setColor(color);
        cuboid.setOpacity(shapeOpacity);
        cuboid.setPosition(state.points[0], state.points[1], state.points[2]);
        cuboid.setScale(state.points[6], state.points[7], state.points[8]);
        cuboid.setRotation(state.points[3], state.points[4], state.points[5]);
        cuboid.attachCameraReference();
        cuboid.setOrientationVisibility(orientationVisibility);

        cuboid[ViewType.PERSPECTIVE].visible = !(state.hidden || state.outside);
        for (const view of BOTTOM_VIEWS) {
            cuboid[view].visible = false;
        }

        return cuboid;
    }

    private resetBottomViewsCamera(targetX?: number, targetY?: number, targetZ?: number): void {
        const [lx, ly, lz] = this.cameraSettings.perspective.lookAt;
        const x = typeof targetX === 'number' ? targetX : lx;
        const y = typeof targetY === 'number' ? targetY : ly;
        const z = typeof targetZ === 'number' ? targetZ : lz;
        for (const cameraType of BOTTOM_VIEWS) {
            const { camera } = this.views[cameraType];
            if (!camera) continue;
            camera.position.set(
                x + this.cameraSettings[cameraType].position[0],
                y + this.cameraSettings[cameraType].position[1],
                z + this.cameraSettings[cameraType].position[2],
            );
            camera.up.set(...this.cameraSettings[cameraType].up);
            camera.lookAt(x, y, z);
            camera.zoom = consts.FOV_DEFAULT;
            camera.updateProjectionMatrix();
        }
        this.updateCameraFrustumPlane();
    }

    private deactivateObject(): void {
        const { shapeOpacity } = this.model.data.configuration;
        if (this.activatedElementID !== null) {
            const { cuboid } = this.drawnObjects[this.activatedElementID];
            cuboid.setOpacity(shapeOpacity);
            for (const view of BOTTOM_VIEWS) {
                cuboid[view].visible = false;
                removeCuboidEdges(cuboid[view]);
                removeResizeHelper(cuboid[view]);
                removeRotationHelper(cuboid[view]);
            }
            this.activatedElementID = null;
            this.resetBottomViewsCamera();
            if (this.isGroundStripped) {
                this.refreshGroundClippingPlane();
                this.updateClippingPlanes();
            }
            this.updateOrthoPointCloudROI(true);
            this.requestOrthoRender(2);
        }
    }

    private activateObject(): void {
        const { selectedShapeOpacity } = this.model.data.configuration;
        const { clientID } = this.model.data.activeElement;
        if (clientID !== null && this.isObjectIsolated && +clientID !== this.isolatedClientID) {
            const newId = +clientID;
            this.isolatedClientID = newId;
            if (this.drawnObjects[newId]?.cuboid?.perspective) {
                this.drawnObjects[newId].cuboid.perspective.visible = true;
            }
            this.hiddenByIsolation.delete(newId);
            Object.keys(this.drawnObjects).forEach((strId: string) => {
                const id = +strId;
                if (id !== newId && this.drawnObjects[id]?.cuboid?.perspective?.visible) {
                    this.drawnObjects[id].cuboid.perspective.visible = false;
                    this.hiddenByIsolation.add(id);
                }
            });
            this.updateIsolationPlanes();
        }
        if (clientID !== null && this.drawnObjects[+clientID]?.cuboid?.perspective?.visible) {
            const { cuboid, data } = this.drawnObjects[+clientID];
            cuboid.setOpacity(selectedShapeOpacity);
            for (const view of BOTTOM_VIEWS) {
                cuboid[view].visible = true;
                createCuboidEdges(cuboid[view]);

                if (!data.lock) {
                    createResizeHelper(cuboid, view);
                    createRotationHelper(cuboid, view);
                    BOTTOM_VIEWS
                        .forEach((type) => this.updateHelperPointsSize(type));
                }
            }

            this.activatedElementID = +clientID;
            this.rotatePlane(null, null);
            this.detachCamera();
            this.updateCameraFrustumPlane();
            if (this.isGroundStripped) {
                this.refreshGroundClippingPlane();
                this.updateClippingPlanes();
            }
            this.updateOrthoPointCloudROI(true);
        }
    }

    private createObjects(states: ObjectState[]): void {
        states.forEach((state: ObjectState) => {
            const cuboid = this.addCuboid(state);
            this.addSceneChildren(cuboid);
            ALL_VIEWS.forEach((v: ViewType) => cuboid[v].updateMatrixWorld(true));
            this.drawnObjects[state.clientID] = {
                cuboid,
                data: drawnDataFromState(state),
            };
        });
    }

    private updateObjects(states: ObjectState[]): void {
        const { outlinedBorders } = this.model.data.configuration;
        states.forEach((state: ObjectState) => {
            const {
                clientID, points, color, label, group, occluded, outside, hidden,
            } = state;
            const { cuboid, data } = this.drawnObjects[clientID];

            if (points.length !== data.points.length ||
                points.some((point: number, idx: number) => point !== data.points[idx])) {
                cuboid.setPosition(state.points[0], state.points[1], state.points[2]);
                cuboid.setScale(state.points[6], state.points[7], state.points[8]);
                cuboid.setRotation(state.points[3], state.points[4], state.points[5]);
                ALL_VIEWS.forEach((v: ViewType) => cuboid[v].updateMatrixWorld(true));
            }

            if (
                color !== data.color ||
                label.id !== data.labelID ||
                group.id !== data.groupID ||
                group.color !== data.groupColor
            ) {
                const newColor = this.receiveShapeColor(state);
                cuboid.setColor(newColor);
                if (outlinedBorders) {
                    cuboid.setOutlineColor(outlinedBorders);
                }
            }

            if (outside !== data.outside || hidden !== data.hidden) {
                cuboid.perspective.visible = !(outside || hidden);
                if (outside || hidden) {
                    cuboid.top.visible = false;
                    cuboid.side.visible = false;
                    cuboid.front.visible = false;
                }
            }

            if (occluded !== data.occluded) {
                this.deleteObjects([clientID]);
                this.createObjects([state]);
                return;
            }

            this.drawnObjects[clientID].data = drawnDataFromState(state);
        });
    }

    private deleteObjects(clientIDs: number[]): void {
        clientIDs.forEach((clientID: number): void => {
            const { cuboid } = this.drawnObjects[clientID];
            this.removeSceneChildren(cuboid, clientID);
        });
    }

    private setupObjectsIncremental(states: ObjectState[]): void {
        const created = states.filter((state: ObjectState): boolean => !(state.clientID in this.drawnObjects));
        const updated = states.filter((state: ObjectState): boolean => {
            if (!(state.clientID in this.drawnObjects)) return false;
            const prev = this.drawnObjects[state.clientID].data;
            return (
                prev.updated !== state.updated ||
                prev.outside !== state.outside ||
                prev.hidden !== state.hidden ||
                prev.occluded !== state.occluded ||
                prev.color !== state.color ||
                prev.labelID !== state.label.id ||
                prev.groupID !== (state.group?.id || null) ||
                prev.groupColor !== (state.group?.color || '#ffffff') ||
                prev.points.length !== state.points.length ||
                state.points.some((p: number, idx: number) => p !== prev.points[idx])
            );
        });
        const deleted = Object.keys(this.drawnObjects).map((key: string): number => +key)
            .filter((clientID: number): boolean => (
                states.findIndex((state: ObjectState) => state.clientID === clientID) === -1
            ));

        const prevIsolatedPos = (
            this.isObjectIsolated &&
            this.isolatedClientID !== null &&
            this.drawnObjects[this.isolatedClientID]?.cuboid?.perspective
        ) ? this.drawnObjects[this.isolatedClientID].cuboid.perspective.position.clone() : null;

        const nextActiveRaw = this.model.data.activeElement.clientID;
        const nextActiveID = nextActiveRaw !== null ? +nextActiveRaw : null;
        const nextActiveState = nextActiveID !== null ?
            states.find((s: ObjectState) => s.clientID === nextActiveID) : undefined;
        const prevActiveData = (this.activatedElementID !== null && this.drawnObjects[this.activatedElementID]) ?
            this.drawnObjects[this.activatedElementID].data : null;
        const canKeepActiveHelpers = (
            this.activatedElementID !== null &&
            this.activatedElementID === nextActiveID &&
            Boolean(nextActiveState) &&
            !nextActiveState!.outside &&
            !nextActiveState!.hidden &&
            Boolean(prevActiveData) &&
            prevActiveData!.occluded === nextActiveState!.occluded &&
            prevActiveData!.lock === nextActiveState!.lock
        );

        if (!canKeepActiveHelpers) {
            this.deactivateObject();
        }
        this.createObjects(created);
        this.updateObjects(updated);
        this.deleteObjects(deleted);

        if (this.isObjectIsolated && this.isolatedClientID !== null) {
            const isolatedEntry = this.drawnObjects[this.isolatedClientID];
            if (isolatedEntry && !isolatedEntry.data.outside && !isolatedEntry.data.hidden && isolatedEntry.cuboid?.perspective) {
                isolatedEntry.cuboid.perspective.visible = true;
                this.hiddenByIsolation.clear();
                Object.keys(this.drawnObjects).forEach((strId: string) => {
                    const id = +strId;
                    if (id !== this.isolatedClientID && this.drawnObjects[id]?.cuboid?.perspective?.visible) {
                        this.drawnObjects[id].cuboid.perspective.visible = false;
                        this.hiddenByIsolation.add(id);
                    }
                });
                if (prevIsolatedPos && this.views.perspective.controls) {
                    const newPos = isolatedEntry.cuboid.perspective.position;
                    const dx = newPos.x - prevIsolatedPos.x;
                    const dy = newPos.y - prevIsolatedPos.y;
                    const dz = newPos.z - prevIsolatedPos.z;
                    if (dx * dx + dy * dy + dz * dz > 1e-8) {
                        const camPos = this.views.perspective.controls.getPosition(new THREE.Vector3());
                        const camTarget = this.views.perspective.controls.getTarget(new THREE.Vector3());
                        this.views.perspective.controls.setLookAt(
                            camPos.x + dx, camPos.y + dy, camPos.z + dz,
                            camTarget.x + dx, camTarget.y + dy, camTarget.z + dz,
                            false,
                        );
                    }
                }
            } else {
                this.isObjectIsolated = false;
                this.isolatedClientID = null;
                this.hiddenByIsolation.forEach((id: number) => {
                    const obj = this.drawnObjects[id];
                    if (obj?.cuboid?.perspective && !obj.data.outside && !obj.data.hidden) {
                        obj.cuboid.perspective.visible = true;
                    }
                });
                this.hiddenByIsolation.clear();
            }
        }

        if (canKeepActiveHelpers) {
            const { cuboid, data } = this.drawnObjects[this.activatedElementID!];
            const { selectedShapeOpacity } = this.model.data.configuration;
            cuboid.setOpacity(selectedShapeOpacity);
            for (const view of BOTTOM_VIEWS) {
                cuboid[view].visible = true;
            }
            if (!data.lock) {
                this.updateResizeHelperPos();
                this.updateRotationHelperPos();
            }
            this.rotatePlane(null, null);
            this.detachCamera();
            this.updateCameraFrustumPlane();
            this.updateOrthoPointCloudROI(true);
        } else {
            this.activateObject();
        }

        if (this.isGroundStripped) {
            this.refreshGroundClippingPlane();
        }
        if (this.isGroundStripped || this.isObjectIsolated) {
            this.updateClippingPlanes();
        }

        this.updateSideViewsZoom();
    }

    private addSceneChildren(shapeObject: CuboidModel): void {
        this.views.perspective.scene.children[0].add(shapeObject.perspective);
        this.views.top.scene.children[0].add(shapeObject.top);
        this.views.side.scene.children[0].add(shapeObject.side);
        this.views.front.scene.children[0].add(shapeObject.front);
    }

    private removeSceneChildren(shapeObject: CuboidModel, clientID: number): void {
        this.views.perspective.scene.children[0].remove(shapeObject.perspective);
        this.views.top.scene.children[0].remove(shapeObject.top);
        this.views.side.scene.children[0].remove(shapeObject.side);
        this.views.front.scene.children[0].remove(shapeObject.front);

        shapeObject.dispose();
        delete this.drawnObjects[clientID];
    }

    private dispatchEvent(event: CustomEvent): void {
        this.views.perspective.renderer.domElement.dispatchEvent(event);
    }

    public notify(model: Canvas3dModel & Master, reason: UpdateReasons): void {
        const resetColor = (list: ObjectState[]): void => {
            list.forEach((state: ObjectState) => {
                const { clientID } = state;
                const { cuboid } = this.drawnObjects[clientID] || {};
                if (cuboid) {
                    cuboid.setColor(this.receiveShapeColor(state));
                }
            });
        };

        if (reason === UpdateReasons.IMAGE_CHANGED) {
            this.statesToBeGrouped = [];

            const onPCDLoadSuccess = (points: any, preInitialized: boolean = false): void => {
                try {
                    this.onSceneImageLoaded(points, preInitialized);
                    model.updateCanvasObjects();
                } finally {
                    model.unlockFrameUpdating();
                    this.dispatchEvent(new CustomEvent('canvas.setup'));
                }
            };

            try {
                if (model.data.image) {
                    if (this.controller.imageIsDeleted) {
                        this.clearScene();
                        this.views.perspective.renderer.dispose();
                        try {
                            this.render();
                            const [container] = window.document.getElementsByClassName('cvat-canvas-container');
                            const overlay = window.document.createElement('canvas');
                            overlay.classList.add('cvat_3d_canvas_deleted_overlay');
                            overlay.style.width = '100%';
                            overlay.style.height = '100%';
                            overlay.style.position = 'absolute';
                            overlay.style.top = '0px';
                            overlay.style.left = '0px';
                            container.appendChild(overlay);
                            const { clientWidth: width, clientHeight: height } = overlay;
                            overlay.width = width;
                            overlay.height = height;
                            const canvasContext = overlay.getContext('2d');
                            const fontSize = width / 10;
                            canvasContext.font = `bold ${fontSize}px serif`;
                            canvasContext.textAlign = 'center';
                            canvasContext.lineWidth = fontSize / 20;
                            canvasContext.strokeStyle = 'white';
                            canvasContext.strokeText('IMAGE REMOVED', width / 2, height / 2);
                            canvasContext.fillStyle = 'black';
                            canvasContext.fillText('IMAGE REMOVED', width / 2, height / 2);
                            this.dispatchEvent(new CustomEvent('canvas.setup'));
                        } finally {
                            model.unlockFrameUpdating();
                        }
                    } else {
                        this.mainSceneWorker.clear();
                        this.previousCuboids.clear();

                        const frameNumber = model.data.imageID;
                        if (frameNumber !== null && this.geometryCache.has(frameNumber)) {
                            const cached = this.geometryCache.get(frameNumber)!;
                            this.geometryCache.delete(frameNumber);
                            this.geometryCache.set(frameNumber, cached);
                            this.groundGrid = cached.groundGrid;
                            this.cachedGlobalGroundPlane = cached.globalGroundPlane;

                            const hasColors = Boolean(cached.geometry.getAttribute('color'));
                            const material = new THREE.PointsMaterial({ size: 0.05, vertexColors: hasColors });
                            const cloud = new THREE.Points(cached.geometry, material);
                            const [overlay] = window.document.getElementsByClassName('cvat_3d_canvas_deleted_overlay');
                            if (overlay) {
                                overlay.remove();
                            }
                            onPCDLoadSuccess(cloud, true);
                        } else {
                            model.data.image.imageData.arrayBuffer().then((data: ArrayBuffer) => {
                                this.mainSceneWorker.parseAndInitScene(
                                    data,
                                    (result) => {
                                        const geometry = new THREE.BufferGeometry();
                                        geometry.setAttribute('position', new THREE.Float32BufferAttribute(result.positions, 3));
                                        if (result.colors) {
                                            geometry.setAttribute('color', new THREE.Float32BufferAttribute(result.colors, 3));
                                        }
                                        if (result.normals) {
                                            geometry.setAttribute('normal', new THREE.Float32BufferAttribute(result.normals, 3));
                                        }

                                        const [minX, minY, minZ, maxX, maxY, maxZ] = result.bbox;
                                        geometry.boundingBox = new THREE.Box3(
                                            new THREE.Vector3(minX, minY, minZ),
                                            new THREE.Vector3(maxX, maxY, maxZ),
                                        );
                                        geometry.boundingSphere = geometry.boundingBox.getBoundingSphere(new THREE.Sphere());

                                        const groundGrid = GroundElevationGrid.fromPrecomputed(result.groundGrid);
                                        this.groundGrid = groundGrid;
                                        this.cachedGlobalGroundPlane = result.globalGroundPlane;

                                        if (frameNumber !== null) {
                                            if (this.geometryCache.size >= 64) {
                                                const oldestKey = this.geometryCache.keys().next().value;
                                                if (oldestKey !== undefined) {
                                                    const evicted = this.geometryCache.get(oldestKey);
                                                    this.geometryCache.delete(oldestKey);
                                                    evicted?.geometry.dispose();
                                                }
                                            }
                                            this.geometryCache.set(frameNumber, {
                                                geometry,
                                                groundGrid,
                                                globalGroundPlane: result.globalGroundPlane,
                                            });
                                        }

                                        const material = new THREE.PointsMaterial({
                                            size: 0.05,
                                            vertexColors: Boolean(result.colors),
                                        });
                                        const cloud = new THREE.Points(geometry, material);
                                        onPCDLoadSuccess(cloud, true);
                                    },
                                    0.5,
                                    (buf: ArrayBuffer) => {
                                        const loader = new PCDLoader();
                                        const defaultImpl = console.error;
                                        let parsedCloud: THREE.Points;
                                        try {
                                            console.error = () => { };
                                            parsedCloud = loader.parse(buf) as THREE.Points;
                                        } finally {
                                            console.error = defaultImpl;
                                        }
                                        const posAttr = parsedCloud.geometry.attributes.position;
                                        const colAttr = parsedCloud.geometry.attributes.color;
                                        const normAttr = parsedCloud.geometry.attributes.normal;
                                        const rawPos = posAttr ? (posAttr.array as Float32Array) : new Float32Array(0);
                                        const rawCol = colAttr ? (colAttr.array as Float32Array) : null;
                                        const rawNorm = normAttr ? (normAttr.array as Float32Array) : null;
                                        const len = rawPos.length;
                                        const validPositions = new Float32Array(len);
                                        const validColors = rawCol ? new Float32Array(len) : null;
                                        const validNormals = rawNorm ? new Float32Array(len) : null;
                                        let validCount = 0;
                                        for (let i = 0; i < len; i += 3) {
                                            const x = rawPos[i];
                                            const y = rawPos[i + 1];
                                            const z = rawPos[i + 2];
                                            if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
                                                const outIdx = validCount * 3;
                                                validPositions[outIdx] = x;
                                                validPositions[outIdx + 1] = y;
                                                validPositions[outIdx + 2] = z;
                                                if (validColors && rawCol) {
                                                    if (Number.isFinite(rawCol[i])) {
                                                        validColors[outIdx] = rawCol[i];
                                                        validColors[outIdx + 1] = rawCol[i + 1];
                                                        validColors[outIdx + 2] = rawCol[i + 2];
                                                    } else {
                                                        validColors[outIdx] = 1;
                                                        validColors[outIdx + 1] = 1;
                                                        validColors[outIdx + 2] = 1;
                                                    }
                                                }
                                                if (validNormals && rawNorm) {
                                                    validNormals[outIdx] = rawNorm[i];
                                                    validNormals[outIdx + 1] = rawNorm[i + 1];
                                                    validNormals[outIdx + 2] = rawNorm[i + 2];
                                                }
                                                validCount++;
                                            }
                                        }
                                        return {
                                            positions: validPositions.subarray(0, validCount * 3),
                                            colors: validColors ? validColors.subarray(0, validCount * 3) : null,
                                            normals: validNormals ? validNormals.subarray(0, validCount * 3) : null,
                                        };
                                    },
                                );
                            });
                            const [overlay] = window.document.getElementsByClassName('cvat_3d_canvas_deleted_overlay');
                            if (overlay) {
                                overlay.remove();
                            }
                        }
                    }
                } else {
                    this.clearScene();
                }
            } catch (error: any) {
                model.unlockFrameUpdating();
                throw error;
            }
        } else if (reason === UpdateReasons.CONFIG_UPDATED) {
            const config = { ...model.data.configuration };
            for (const key of Object.keys(this.drawnObjects)) {
                const clientID = +key;
                const { cuboid, data } = this.drawnObjects[clientID];
                const newColor = this.receiveShapeColor(data);
                cuboid.setColor(newColor);
                cuboid.setOpacity(
                    ((clientID === this.activatedElementID) ? config.selectedShapeOpacity : config.shapeOpacity),
                );

                if (config.outlinedBorders) {
                    cuboid.setOutlineColor(config.outlinedBorders || consts.DEFAULT_OUTLINE_COLOR);
                }

                cuboid.setOrientationVisibility(config.orientationVisibility);
            }

            BOTTOM_VIEWS.forEach((view) => {
                // in case of changing control points size, need to update all helper points size
                this.updateHelperPointsSize(view);
            });
        } else if (reason === UpdateReasons.SHAPE_ACTIVATED) {
            this.deactivateObject();
            this.activateObject();
            this.updateSideViewsZoom();
        } else if (reason === UpdateReasons.DRAW) {
            const data: DrawData = this.controller.drawData;
            if (Number.isInteger(data.redraw)) {
                if (this.drawnObjects[data.redraw]?.cuboid?.perspective?.visible) {
                    const { cuboid } = this.drawnObjects[data.redraw];
                    this.cube.perspective = cuboid.perspective.clone() as THREE.Mesh;
                    cuboid.perspective.visible = false;
                } else {
                    // an object must be drawn and visible to be redrawn
                    model.cancel();
                    return;
                }
            } else if (data.initialState && Array.isArray(data.initialState.points)) {
                if (!data.initialState.outside && !data.initialState.hidden) {
                    this.cube = this.addCuboid(data.initialState);
                } else {
                    // an object must visible to paste it
                    model.cancel();
                    return;
                }
            } else {
                this.cube = new CuboidModel('line', '#ffffff');
                this.cube.setOrientationVisibility(this.model.data.configuration.orientationVisibility);
            }

            const label = data.initialState?.label;
            const labelName = label?.name;
            const [dx, dy, dz] = data.initialDimensions || getClassDimensionPrior(label || labelName, {
                attributes: data.initialState?.attributes,
                label,
            });
            if (!Number.isInteger(data.redraw) && (!data.initialState || !Array.isArray(data.initialState.points))) {
                this.cube.perspective.scale.set(dx, dy, dz);
            }

            this.cube.setName('drawTemplate');
            this.deactivateObject();
            this.views[ViewType.PERSPECTIVE].scene.children[0].add(this.cube.perspective);
        } else if (reason === UpdateReasons.OBJECTS_UPDATED) {
            this.setupObjectsIncremental(model.objects);
        } else if (reason === UpdateReasons.DRAG_CANVAS) {
            this.isPerspectiveBeingDragged = true;
            this.dispatchEvent(
                new CustomEvent('canvas.dragstart', {
                    bubbles: false,
                    cancelable: true,
                }),
            );
            model.data.activeElement.clientID = null;
            this.deactivateObject();
        } else if (reason === UpdateReasons.CANCEL) {
            if (this.mode === Mode.DRAG_CANVAS) {
                this.isPerspectiveBeingDragged = false;
                this.dispatchEvent(
                    new CustomEvent('canvas.dragstop', {
                        bubbles: false,
                        cancelable: true,
                    }),
                );
            }

            if (this.mode === Mode.DRAW) {
                this.controller.drawData.enabled = false;
                const { redraw } = this.controller.drawData;
                if (Number.isInteger(redraw)) {
                    this.drawnObjects[redraw].cuboid.perspective.visible = true;
                    this.controller.drawData.redraw = undefined;
                }
                const scene = this.views[ViewType.PERSPECTIVE].scene.children[0];
                const template = scene.getObjectByName('drawTemplate');
                if (template) {
                    scene.remove(template);
                }
            }

            if (this.mode === Mode.MERGE) {
                const { statesToBeMerged } = this;
                this.statesToBeMerged = [];
                resetColor(statesToBeMerged);
                this.model.merge({ enabled: false });
            }

            if (this.mode === Mode.GROUP) {
                const { statesToBeGrouped } = this;
                this.statesToBeGrouped = [];
                resetColor(statesToBeGrouped);
                this.model.group({ enabled: false });
            }

            if (this.mode === Mode.SPLIT) {
                if (this.stateToBeSplitted) {
                    const state = this.stateToBeSplitted;
                    this.stateToBeSplitted = null;
                    this.drawnObjects[state.clientID].cuboid.setColor(this.receiveShapeColor(state));
                }
                this.model.split({ enabled: false });
            }

            this.mode = Mode.IDLE;
            this.dispatchEvent(new CustomEvent('canvas.canceled'));
        } else if (reason === UpdateReasons.FITTED_CANVAS) {
            this.fitCanvas(false);
            this.dispatchEvent(new CustomEvent('canvas.fit'));
        } else if (reason === UpdateReasons.GROUP) {
            if (!model.groupData.enabled && this.statesToBeGrouped.length) {
                this.onGroupDone(this.statesToBeGrouped);
                resetColor(this.statesToBeGrouped);
            } else if (model.groupData.enabled) {
                this.deactivateObject();
                this.statesToBeGrouped = [];
                model.data.activeElement.clientID = null;
            }
        } else if (reason === UpdateReasons.SPLIT) {
            this.deactivateObject();
            this.stateToBeSplitted = null;
            model.data.activeElement.clientID = null;
        } else if (reason === UpdateReasons.MERGE) {
            if (!model.mergeData.enabled && this.statesToBeMerged.length) {
                this.onMergeDone(this.statesToBeMerged);
                resetColor(this.statesToBeMerged);
            } else if (model.mergeData.enabled) {
                this.deactivateObject();
                this.statesToBeMerged = [];
                model.data.activeElement.clientID = null;
            }
        } else if (reason === UpdateReasons.DATA_FAILED) {
            this.onError(model.exception, 'data fetching');
        }
        this.requestRender();
    }

    private clearScene(): void {
        const previousPoints = this.pointCloudPoints;
        const previousMeshes = this.pointCloudMeshes;
        const previousOrthoGeo = this.orthoPointsGeometry;
        this.pointCloudPoints = null;
        this.perfPointCount = 0;
        this.perfOrthoRoiPoints = 0;
        this.groundGrid = null;
        this.cachedGlobalGroundPlane = null;
        this.groundClippingPlane = null;
        this.mainSceneWorker.clear();
        this.previousCuboids.clear();
        this.isObjectIsolated = false;
        this.isolatedClientID = null;
        this.hiddenByIsolation.clear();
        this.pointCloudMeshes = [];
        this.orthoPointsGeometry = null;
        this.orthoRoiSphereUniform.value.set(0, 0, 0, 0);
        this.lastRoiCenter = null;
        this.scenePlanes = {
            [ViewType.TOP]: null,
            [ViewType.SIDE]: null,
            [ViewType.FRONT]: null,
        };
        Object.entries(this.drawnObjects).forEach(([clientID, { cuboid }]) => {
            this.removeSceneChildren(cuboid, +clientID);
        });

        this.drawnObjects = {};
        this.activatedElementID = null;

        // Detach pointCloudMeshes before disposeScene so cached BufferGeometry
        // in this.geometryCache stays resident in GPU VRAM across frame navigation.
        if (previousPoints) {
            while (previousPoints.children.length > 0) {
                const child = previousPoints.children[0];
                previousPoints.remove(child);
            }
            this.views.perspective.scene.remove(previousPoints);
            if (previousPoints.material instanceof THREE.Material) {
                previousPoints.material.dispose();
            }
        }
        for (const mesh of previousMeshes) {
            if (mesh && mesh !== previousPoints) {
                mesh.parent?.remove(mesh);
                if (mesh.material instanceof THREE.Material) {
                    mesh.material.dispose();
                }
            }
        }
        if (previousOrthoGeo && previousOrthoGeo !== previousPoints?.geometry) {
            previousOrthoGeo.dispose();
        }

        Object.keys(this.views).forEach((view: string): void => {
            const viewData = this.views[view as keyof Views];
            disposeScene(viewData.scene);
        });
    }

    private updateRotationHelperPos(): void {
        const cuboid = this.selectedCuboid;
        if (!cuboid) {
            return;
        }

        BOTTOM_VIEWS.forEach((view: ViewType): void => {
            const rotationHelper = cuboid[view].parent.getObjectByName(consts.ROTATION_HELPER_NAME);
            if (rotationHelper) {
                const position = cuboid.getRotationHelperPosition(view);
                rotationHelper.position.copy(position);
            }
        });
    }

    private updateResizeHelperPos(): void {
        const cuboid = this.selectedCuboid;
        if (cuboid === null) {
            return;
        }

        BOTTOM_VIEWS.forEach((view: ViewType): void => {
            const positions = cuboid.getResizeHelperPositions();
            const pointsToBeUpdated = cuboid[view].parent.children
                .filter((child: THREE.Object3D) => child.name.startsWith(consts.RESIZE_HELPER_NAME))
                .sort((child1: THREE.Object3D, child2: THREE.Object3D) => {
                    const order1 = +child1.name.split('_')[1];
                    const order2 = +child2.name.split('_')[1];
                    return order1 - order2;
                });

            for (let i = 0; i < positions.length; i++) {
                const position = positions[i];
                pointsToBeUpdated[i].position.copy(position);
            }
        });
    }

    private updateHelperPointsSize(viewType: ViewType): void {
        const helperWorldPointsSize = this.computeHelperSphereWorldSize(viewType);
        if (BOTTOM_VIEWS.includes(viewType)) {
            const { scene } = this.views[viewType];
            const isSceneRendered = scene.children.length > 0;
            if (!isSceneRendered) {
                return;
            }

            // as physical size is 1, we just scale to target size
            const rotationObject = scene.children[0]?.getObjectByName(consts.ROTATION_HELPER_NAME);
            if (rotationObject) {
                rotationObject.scale.set(helperWorldPointsSize, helperWorldPointsSize, helperWorldPointsSize);
            }

            scene.children[0].children
                .filter((child: THREE.Object3D) => child.name.startsWith(consts.RESIZE_HELPER_NAME))
                .forEach((child: THREE.Object3D) => {
                    child.scale.set(helperWorldPointsSize, helperWorldPointsSize, helperWorldPointsSize);
                });
        }
    }

    /**
     * Compute helper sphere size in world units for orthographic cameras.
     *
     * Goal: given desired helper size in pixels (controlsPointSize) and the current
     * canvas size in pixels, convert that into world-space radius/diameter so that
     * helpers (rotation/resize spheres) appear with consistent on-screen size.
     *
     * This method only targets orthographic views (BOTTOM_VIEWS). For perspective
     *   views, apparent size depends on distance to the camera and FOV.
     */
    private computeHelperSphereWorldSize(viewType: ViewType): number {
        const { controlPointsSize: controlPointSizePx } = this.model.data.configuration;
        const view = this.views[viewType];
        const camera = view.camera as THREE.OrthographicCamera;
        const canvas = view.renderer.domElement;
        if (!camera || !canvas) {
            return controlPointSizePx;
        }

        // 1) Baseline vertical frustum height in world units divided by zoom
        const baselineWorldHeight = consts.ORTHO_FRUSTUM_HEIGHT; // ORTHO_FRUSTUM_HEIGHT
        const visibleWorldHeight = baselineWorldHeight / Math.max(camera.zoom, Number.EPSILON);

        // 2) World units per one pixel along vertical axis
        const canvasHeightPx = canvas.clientHeight || canvas.parentElement?.clientHeight || 1;
        const worldPerPixelY = visibleWorldHeight / canvasHeightPx;

        // 3) Convert desired pixel size to world units
        // 1.5 is a scaling factor for better visibility
        const helperWorldSize = controlPointSizePx * worldPerPixelY * 1.5;

        return helperWorldSize;
    }

    private getGroundGrid(): GroundElevationGrid | null {
        if (!this.pointCloudPoints?.geometry?.attributes?.position?.array) {
            this.groundGrid = null;
            return null;
        }
        if (!this.groundGrid) {
            const positions = this.pointCloudPoints.geometry.attributes.position.array;
            this.groundGrid = new GroundElevationGrid(positions, 0.5);
        }
        return this.groundGrid;
    }

    private onSceneImageLoaded(points: any, preInitialized: boolean = false): void {
        const newGeo: THREE.BufferGeometry = points.geometry;
        const positions = newGeo?.attributes?.position?.array;
        this.perfPointCount = positions ? Math.floor(positions.length / 3) : 0;
        if (!preInitialized) {
            this.groundGrid = null;
            this.cachedGlobalGroundPlane = null;
        }
        if (positions && positions.length >= 9 && !preInitialized) {
            const expectedGeo = newGeo;
            this.mainSceneWorker.initScene(positions, (res) => {
                if (this.pointCloudPoints?.geometry !== expectedGeo) return;
                this.groundGrid = GroundElevationGrid.fromPrecomputed(res.groundGrid);
                this.cachedGlobalGroundPlane = res.globalGroundPlane;
                if (this.isGroundStripped) {
                    this.refreshGroundClippingPlane();
                    this.updateClippingPlanes();
                    this.requestRender();
                }
            });
        }
        const getCameraSettingsToFitScene = (
            camera: THREE.PerspectiveCamera,
            boundingBox: THREE.Box3,
        ): {
            position: [number, number, number],
            lookAt: [number, number, number],
        } => {
            const width = boundingBox.max.x - boundingBox.min.x;
            const height = boundingBox.max.y - boundingBox.min.y;
            const depth = boundingBox.max.z - boundingBox.min.z;

            // Center of the scene
            const centerX = boundingBox.min.x + width / 2;
            const centerY = boundingBox.min.y + height / 2;
            const centerZ = boundingBox.min.z + depth / 2;

            // Calculate distance to fit the scene: distance = (size / 2) / tan(fov / 2)
            const maxDim = Math.max(width, height, depth);
            const fovRadians = camera.fov * (Math.PI / 180);
            const cameraDistance = (maxDim / 2) / Math.tan(fovRadians / 2);

            // Position camera above and slightly offset from center
            const offset = 5;

            return {
                position: [
                    centerX + offset,
                    centerY + offset,
                    centerZ + cameraDistance + offset,
                ],
                lookAt: [
                    centerX,
                    centerY,
                    centerZ,
                ],
            };
        };

        const { controls } = this.views.perspective;
        controls.mouseButtons.wheel = CameraControls.ACTION.DOLLY;
        if (!this.views.perspective.camera) return;

        if (!newGeo.boundingBox) {
            newGeo.computeBoundingBox();
        }
        const { position, lookAt } = getCameraSettingsToFitScene(
            this.views.perspective.camera as THREE.PerspectiveCamera, newGeo.boundingBox,
        );

        this.cameraSettings.perspective.position = position;
        this.sceneBBox = newGeo.boundingBox ?
            newGeo.boundingBox.clone() :
            new THREE.Box3().setFromObject(points);

        const origin = new THREE.Vector3(0, 0, 0);
        const isOriginInScene = this.sceneBBox.containsPoint(origin);
        if (isOriginInScene) {
            this.cameraSettings.perspective.lookAt = [0, 0, 0];
        } else {
            this.cameraSettings.perspective.lookAt = lookAt;
        }

        const hasColors = Boolean(newGeo.getAttribute('color'));
        if (this.pointCloudMeshes.length === 4 && this.views.perspective.scene.children.length > 0) {
            const mainPoints = this.pointCloudMeshes[0];
            if (points !== mainPoints && points.material instanceof THREE.Material) {
                points.material.dispose();
            }
            this.pointCloudPoints = mainPoints;
            this.orthoPointsGeometry = newGeo;
            this.lastRoiCenter = null;
            this.orthoRoiSphereUniform.value.set(0, 0, 0, 0);

            for (const mesh of this.pointCloudMeshes) {
                mesh.geometry = newGeo;
                const mat = mesh.material as THREE.PointsMaterial;
                if (mat && mat.vertexColors !== hasColors) {
                    mat.vertexColors = hasColors;
                    mat.needsUpdate = true;
                }
            }

            const existingAxes = this.views.perspective.scene.children.find(
                (child: THREE.Object3D) => child instanceof THREE.AxesHelper,
            );
            if (existingAxes) {
                if (isOriginInScene) {
                    existingAxes.position.set(0, 0, 0);
                } else {
                    existingAxes.position.set(lookAt[0], lookAt[1], lookAt[2]);
                }
            }
        } else {
            this.pointCloudPoints = points;
            // eslint-disable-next-line no-param-reassign
            points.material.size = 0.05;
            // eslint-disable-next-line no-param-reassign
            points.material.vertexColors = hasColors;
            points.material.color.set(new THREE.Color(0xffffff));

            const material = points.material.clone();
            this.views.perspective.scene.add(points);

            const axesHelper = new THREE.AxesHelper(5);
            if (isOriginInScene) {
                axesHelper.position.set(0, 0, 0);
            } else {
                axesHelper.position.set(lookAt[0], lookAt[1], lookAt[2]);
            }
            this.views.perspective.scene.add(axesHelper);

            // Setup TopView
            const canvasTopView = this.views.top.renderer.domElement;
            const topScenePlane = new THREE.Mesh(
                new THREE.PlaneGeometry(
                    Math.max(canvasTopView.offsetHeight, 1000),
                    Math.max(canvasTopView.offsetWidth, 1000),
                    1,
                    1,
                ),
                new THREE.MeshBasicMaterial({
                    color: 0xffffff,
                    visible: false,
                }),
            );
            topScenePlane.position.set(0, 0, 0);
            topScenePlane.name = Planes.TOP;
            (topScenePlane.material as THREE.MeshBasicMaterial).side = THREE.DoubleSide;
            (topScenePlane as any).verticesNeedUpdate = true;
            this.scenePlanes[ViewType.TOP] = topScenePlane;

            const orthoGeo: THREE.BufferGeometry = newGeo;
            this.orthoRoiSphereUniform.value.set(0, 0, 0, 0);
            const createOrthoMaterial = (): THREE.PointsMaterial => {
                const mat = material.clone();
                mat.size = 0.5;
                mat.onBeforeCompile = (shader: any) => {
                    // eslint-disable-next-line no-param-reassign
                    shader.uniforms.uOrthoRoiSphere = this.orthoRoiSphereUniform;
                    // eslint-disable-next-line no-param-reassign
                    shader.vertexShader = `uniform vec4 uOrthoRoiSphere;\n${shader.vertexShader}`.replace(
                        '#include <project_vertex>',
                        `#include <project_vertex>
                        if (uOrthoRoiSphere.w > 0.0 && distance(transformed, uOrthoRoiSphere.xyz) > uOrthoRoiSphere.w) {
                            gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                            gl_PointSize = 0.0;
                        }`,
                    );
                };
                return mat;
            };
            this.orthoPointsGeometry = orthoGeo;
            this.lastRoiCenter = null;

            const topCloud = new THREE.Points(orthoGeo, createOrthoMaterial());
            this.views.top.scene.add(topCloud);
            this.views.top.scene.add(topScenePlane);

            // Setup Side View
            const canvasSideView = this.views.side.renderer.domElement;
            const sideScenePlane = new THREE.Mesh(
                new THREE.PlaneGeometry(
                    Math.max(canvasSideView.offsetHeight, 1000),
                    Math.max(canvasSideView.offsetWidth, 1000),
                    1,
                    1,
                ),
                new THREE.MeshBasicMaterial({
                    color: 0x00ff00,
                    visible: false,
                    opacity: 0.5,
                }),
            );
            sideScenePlane.position.set(0, 0, 0);
            sideScenePlane.rotation.set(0, 0, 0);
            sideScenePlane.name = Planes.SIDE;
            (sideScenePlane.material as THREE.MeshBasicMaterial).side = THREE.DoubleSide;
            (sideScenePlane as any).verticesNeedUpdate = true;
            this.scenePlanes[ViewType.SIDE] = sideScenePlane;

            const sideCloud = new THREE.Points(orthoGeo, createOrthoMaterial());
            this.views.side.scene.add(sideCloud);
            this.views.side.scene.add(sideScenePlane);

            // Setup front View
            const canvasFrontView = this.views.front.renderer.domElement;
            const frontScenePlane = new THREE.Mesh(
                new THREE.PlaneGeometry(
                    Math.max(canvasFrontView.offsetHeight, 1000),
                    Math.max(canvasFrontView.offsetWidth, 1000),
                    1,
                    1,
                ),
                new THREE.MeshBasicMaterial({
                    color: 0xffffff,
                    visible: false,
                }),
            );
            frontScenePlane.position.set(0, 0, 0);
            frontScenePlane.rotation.set(0, 0, 0);
            frontScenePlane.name = Planes.FRONT;
            (frontScenePlane.material as THREE.MeshBasicMaterial).side = THREE.DoubleSide;
            (frontScenePlane as any).verticesNeedUpdate = true;
            this.scenePlanes[ViewType.FRONT] = frontScenePlane;

            const frontCloud = new THREE.Points(orthoGeo, createOrthoMaterial());
            this.views.front.scene.add(frontCloud);
            this.views.front.scene.add(frontScenePlane);

            this.pointCloudMeshes = [points, topCloud, sideCloud, frontCloud];
        }

        if (this.isGroundStripped) {
            this.refreshGroundClippingPlane();
        }
        if (this.getActiveElement() === null) {
            this.resetBottomViewsCamera();
        }
        this.updateClippingPlanes();
        this.updateOrthoPointCloudROI(true);

        if (this.mode === Mode.DRAW && !this.views[ViewType.PERSPECTIVE].scene.children[0].children.includes(this.cube.perspective)) {
            this.views[ViewType.PERSPECTIVE].scene.children[0].add(this.cube.perspective);
        }
        this.requestRender();
    }

    private positionAllViews(x: number, y: number, z: number, animation: boolean): void {
        if (
            this.views.perspective.controls &&
            this.views.top.controls &&
            this.views.side.controls &&
            this.views.front.controls
        ) {
            this.views.perspective.controls.setLookAt(
                this.cameraSettings.perspective.position[0],
                this.cameraSettings.perspective.position[1],
                this.cameraSettings.perspective.position[2],
                x, y, z, animation,
            );

            this.resetBottomViewsCamera(x, y, z);
        }
    }

    private resizeRendererToDisplaySize(viewType: ViewType, view: RenderView): boolean {
        const { camera, renderer } = view;
        const canvas = renderer.domElement;
        if (!canvas.parentElement) {
            return false;
        }
        const width = canvas.parentElement.clientWidth;
        const height = canvas.parentElement.clientHeight;
        if (width <= 0 || height <= 0) {
            return false;
        }
        const cached = this.viewSizes[viewType];
        const shouldResize = cached.width !== width || cached.height !== height;
        if (shouldResize && camera && view.camera) {
            cached.width = width;
            cached.height = height;
            if (camera instanceof THREE.PerspectiveCamera) {
                camera.aspect = width / height;
            } else {
                const viewSize = consts.ORTHO_FRUSTUM_HEIGHT;
                const aspectRatio = width / height;
                camera.left = (-aspectRatio * viewSize) / 2;
                camera.right = (aspectRatio * viewSize) / 2;
                camera.top = viewSize / 2;
                camera.bottom = -viewSize / 2;
                this.updateHelperPointsSize(viewType);
            }

            view.renderer.setSize(width, height, false);
            view.camera.updateProjectionMatrix();
            return true;
        }
        return false;
    }

    private renderRayCaster = (viewType: RenderView): void => {
        const hadHoverUpdate = this.hoverNeedsUpdate;
        this.hoverNeedsUpdate = false;
        viewType.rayCaster.renderer.setFromCamera(viewType.rayCaster.mouseVector, viewType.camera);
        if (this.mode === Mode.DRAW) {
            const object = this.views.perspective.scene.getObjectByName('drawTemplate');
            if (!object) return;

            const groundGrid = this.getGroundGrid();
            const globalPlane = this.cachedGlobalGroundPlane || this.getGroundPlane();
            const ray = viewType.rayCaster.renderer.ray;

            let hitPoint: THREE.Vector3 | null = null;
            if (groundGrid) {
                hitPoint = groundGrid.intersectRay(ray, globalPlane, 150);
            }
            if (!hitPoint) {
                let [A, B, C, D] = globalPlane;
                const len = Math.sqrt(A * A + B * B + C * C);
                if (len > 1e-6) {
                    A /= len; B /= len; C /= len; D /= len;
                } else {
                    A = 0; B = 0; C = 1; D = 1.65;
                }
                const fallbackPlane = new THREE.Plane(new THREE.Vector3(A, B, C), D);
                hitPoint = ray.intersectPlane(fallbackPlane, new THREE.Vector3());
            }

            if (hitPoint) {
                const { x, y, z } = hitPoint;
                const h = this.cube.perspective.scale.z;
                let zGround = z;
                let rx = 0;
                let ry = 0;
                const rz = object.rotation.z;

                if (groundGrid) {
                    const roadZ = groundGrid.getElevation(x, y);
                    if (roadZ !== null && Math.abs(roadZ - z) <= 2.5) {
                        zGround = roadZ;
                    }
                    const label = this.controller.drawData.initialState?.label;
                    const isPedestrian = isPedestrianClass(label || label?.name);
                    if (!isPedestrian) {
                        const plane = groundGrid.getRoadPlane(x, y);
                        if (plane) {
                            const aligned = alignEulerToRoadPlane(rz, plane);
                            rx = aligned.rx;
                            ry = aligned.ry;
                        }
                    }
                }

                object.position.set(x, y, zGround + h / 2);
                object.rotation.set(rx, ry, rz);
            }
        } else {
            const { renderer } = this.views.perspective.rayCaster;
            const intersects = renderer.intersectObjects(this.getAllVisibleCuboids(), false);
            if (intersects.length !== 0 && !this.isPerspectiveBeingDragged) {
                const clientID = intersects[0].object.name;
                const castedClientID = +clientID;

                if (this.mode === Mode.SPLIT) {
                    const objectState = Number.isInteger(castedClientID) ? this.model.objects
                        .find((state: ObjectState) => state.clientID === castedClientID) : null;
                    if (this.stateToBeSplitted !== objectState) {
                        this.stateToBeSplitted = objectState;
                        this.drawnObjects[castedClientID].cuboid.setColor(this.receiveShapeColor(objectState));
                        this.requestPerspectiveRender(1);
                    }
                } else if (this.mode === Mode.IDLE && !this.isCtrlDown) {
                    const intersectedClientID = intersects[0]?.object?.name || null;
                    const activeClientID = this.model.data.activeElement.clientID;
                    if (activeClientID !== null && !hadHoverUpdate) {
                        return;
                    }

                    if (activeClientID !== intersectedClientID) {
                        const object = intersectedClientID ?
                            this.views.perspective.scene.getObjectByName(intersectedClientID) :
                            null;
                        if (intersectedClientID && object === undefined) return;

                        const numericClientID =
                            typeof intersectedClientID === 'string' ? +intersectedClientID : null;

                        this.dispatchEvent(
                            new CustomEvent('canvas.selected', {
                                bubbles: false,
                                cancelable: true,
                                detail: {
                                    clientID: numericClientID,
                                },
                            }),
                        );
                    }
                }
            } else if (this.mode === Mode.SPLIT && this.stateToBeSplitted) {
                const state = this.stateToBeSplitted;
                this.stateToBeSplitted = null;
                this.drawnObjects[state.clientID].cuboid.setColor(this.receiveShapeColor(state));
                this.requestPerspectiveRender(1);
            }
        }
    };

    public render(): void {
        const renderStart = typeof performance !== 'undefined' ? performance.now() : Date.now();
        const delta = this.clock.getDelta();
        const { clientID } = this.model.data.activeElement;

        // Process active ortho view action BEFORE rendering so all 4 views render with zero frame lag
        const scanView = this.action.scan as ViewType | null;
        if (clientID !== null && scanView && scanView !== ViewType.PERSPECTIVE) {
            const scanViewType = this.views[scanView];
            if (scanViewType && scanViewType.camera && scanViewType.rayCaster) {
                scanViewType.rayCaster.renderer.setFromCamera(
                    scanViewType.rayCaster.mouseVector,
                    scanViewType.camera,
                );
                if (!(this.action.translation.status || this.action.resize.status || this.action.rotation.status)) {
                    this.initiateAction(scanView, scanViewType);
                }
                if (this.action.detected) {
                    if (this.action.translation.status) {
                        this.renderTranslateAction(scanView, scanViewType);
                    } else if (this.action.resize.status) {
                        this.renderResizeAction(scanView, scanViewType);
                    } else {
                        this.renderRotateAction(scanView, scanViewType);
                    }
                } else {
                    this.resetActions();
                }
            }
        }

        const isActionActive = Boolean(
            this.action.detected ||
            this.action.translation.status ||
            this.action.resize.status ||
            this.action.rotation.status,
        );

        if (isActionActive) {
            this.needsPerspectiveRenderCount = 2;
            this.needsOrthoRenderCount = 2;
        }

        const forceRenderPerspective = this.needsPerspectiveRenderCount > 0;
        if (this.needsPerspectiveRenderCount > 0) {
            this.needsPerspectiveRenderCount--;
        }

        const initialForceOrtho = this.needsOrthoRenderCount > 0;
        if (this.needsOrthoRenderCount > 0) {
            this.needsOrthoRenderCount--;
        }

        let didRenderAnyView = false;
        Object.keys(this.views).forEach((view: string): void => {
            const viewType = this.views[view as keyof Views];
            if (!(viewType.controls && viewType.camera && viewType.rayCaster)) return;

            const resized = this.resizeRendererToDisplaySize(view as ViewType, viewType);
            let controlsUpdated = false;
            if (viewType.controls.enabled) {
                controlsUpdated = viewType.controls.update(delta);
            } else {
                viewType.camera.updateProjectionMatrix();
            }

            const isPerspective = view === ViewType.PERSPECTIVE;
            const isScanningView = this.action.scan === view;
            const needsPerspectiveInteraction = isPerspective &&
                this.mode === Mode.DRAW &&
                this.hoverNeedsUpdate;

            let shouldRender = isPerspective ?
                (forceRenderPerspective || resized || controlsUpdated || needsPerspectiveInteraction) :
                (initialForceOrtho || this.needsOrthoRenderCount > 0 || resized || controlsUpdated || isScanningView);

            if (isPerspective && viewType.scene.children.length !== 0) {
                if (!this.isPerspectiveBeingDragged && (this.hoverNeedsUpdate || (this.mode === Mode.DRAW && shouldRender))) {
                    this.renderRayCaster(viewType);
                    if (this.needsPerspectiveRenderCount > 0) {
                        shouldRender = true;
                    }
                }
            }

            if (shouldRender) {
                viewType.renderer.render(viewType.scene, viewType.camera);
                didRenderAnyView = true;
            }
        });

        const renderEnd = typeof performance !== 'undefined' ? performance.now() : Date.now();
        const currentRenderMs = renderEnd - renderStart;
        if (didRenderAnyView) {
            this.perfRenderTimeMs = this.perfRenderTimeMs > 0 ?
                this.perfRenderTimeMs * 0.8 + currentRenderMs * 0.2 :
                currentRenderMs;
        }
        if (delta > 0.001 && delta < 0.5) {
            const currentFrameMs = delta * 1000;
            this.perfFrameTimeMs = this.perfFrameTimeMs > 0 ?
                this.perfFrameTimeMs * 0.85 + currentFrameMs * 0.15 :
                currentFrameMs;
            this.perfFps = Math.min(240, Math.round(1000 / Math.max(1, this.perfFrameTimeMs)));
        }
        if (this.perfHudEnabled && renderEnd - this.perfHudLastUpdate >= 250) {
            this.perfHudLastUpdate = renderEnd;
            this.updatePerformanceHud();
        }
    }

    private adjustPerspectiveCameras(): void {
        const { camera: cameraTop } = this.views.top;
        const { camera: cameraSide } = this.views.side;
        const { camera: cameraFront } = this.views.front;
        const { selectedCuboid } = this;

        const coordinatesTop = this.selectedCuboid.getReferenceCoordinates(ViewType.TOP);
        const sphericalTop = new THREE.Spherical();
        sphericalTop.setFromVector3(coordinatesTop);
        cameraTop.position.setFromSpherical(sphericalTop);
        cameraTop.updateProjectionMatrix();

        const coordinatesSide = selectedCuboid.getReferenceCoordinates(ViewType.SIDE);
        const sphericalSide = new THREE.Spherical();
        sphericalSide.setFromVector3(coordinatesSide);
        cameraSide.position.setFromSpherical(sphericalSide);
        cameraSide.updateProjectionMatrix();

        const coordinatesFront = selectedCuboid.getReferenceCoordinates(ViewType.FRONT);
        const sphericalFront = new THREE.Spherical();
        sphericalFront.setFromVector3(coordinatesFront);
        cameraFront.position.setFromSpherical(sphericalFront);
        cameraFront.updateProjectionMatrix();
    }

    private renderTranslateAction(view: ViewType, viewType: any): void {
        if (
            this.action.translation.helper.x === this.views[view].rayCaster.mouseVector.x &&
            this.action.translation.helper.y === this.views[view].rayCaster.mouseVector.y
        ) {
            return;
        }
        const plane = this.scenePlanes[view] || viewType.scene.getObjectByName(`${view}Plane`);
        const intersects = viewType.rayCaster.renderer.intersectObjects(
            plane ? [plane] : [],
            true,
        );

        if (intersects.length !== 0 && intersects[0].point) {
            const coordinates = intersects[0].point;
            this.action.translation.coordinates = coordinates;
            this.moveObject(coordinates);
            this.detachCamera(view);
            this.updateCameraFrustumPlane(view);
            this.updateOrthoPointCloudROI();
        }
    }

    private moveObject(coordinates: THREE.Vector3): void {
        const {
            perspective, top, side, front,
        } = this.selectedCuboid;
        let localCoordinates = coordinates;
        if (this.action.translation.status) {
            localCoordinates = coordinates
                .clone()
                .sub(this.action.translation.offset)
                .applyMatrix4(this.action.translation.inverseMatrix);
        }
        perspective.position.copy(localCoordinates.clone());
        top.position.copy(localCoordinates.clone());
        side.position.copy(localCoordinates.clone());
        front.position.copy(localCoordinates.clone());

        this.updateResizeHelperPos();
        this.updateRotationHelperPos();
    }

    private renderResizeAction(view: ViewType, viewType: any): void {
        const cuboid = this.selectedCuboid;
        const plane = this.scenePlanes[view] || viewType.scene.getObjectByName(`${view}Plane`);
        const intersects = viewType.rayCaster.renderer
            .intersectObjects(plane ? [plane] : [], true);

        if (
            cuboid === null || intersects.length === 0) {
            return;
        }

        if (!this.action.resize.previousPosition) {
            this.action.resize.previousPosition = intersects[0].object.worldToLocal(intersects[0].point.clone());
            return;
        }

        if (Math.abs(this.action.resize.previousPosition.x - intersects[0].point.x) < Number.EPSILON ||
            Math.abs(this.action.resize.previousPosition.y - intersects[0].point.y) < Number.EPSILON) {
            return;
        }

        // first let's find the point that is used to resize
        // and the opposite point in another corner
        const currentPointNumber = +this.action.resize.helperElement.name.split('_')[1];
        const cuboidNodes = makeCornerPointsMatrix(0.5, 0.5, 0.5);
        const crosslyingPointInternalCoordinates = (new THREE.Vector3())
            .fromArray(cuboidNodes[+currentPointNumber]).multiply(new THREE.Vector3(-1, -1, -1));
        const crosslyingHelperIndex = cuboidNodes
            .findIndex(([x, y, z]): boolean => (
                Math.sign(crosslyingPointInternalCoordinates.x) === Math.sign(x) &&
                Math.sign(crosslyingPointInternalCoordinates.y) === Math.sign(y) &&
                Math.sign(crosslyingPointInternalCoordinates.z) === Math.sign(z)
            ));
        const crosslyingHelper = cuboid.perspective.getObjectByName(`cuboidNodeHelper_${crosslyingHelperIndex}`);
        const crosslyingPointCoordinates = crosslyingHelper.getWorldPosition(new THREE.Vector3());

        // after we've found two points
        // we can get all the information from them (scale and center)
        // but first we need to move the current point
        // we will move point in "internal" cuboid coordinates
        // and then using localToWorld we will receive world coordinates
        const currentPointCoordOnPlane = intersects[0].object.worldToLocal(intersects[0].point.clone());
        const scale = cuboid.perspective.scale.clone();
        const currentPointInternalCoordinates = new THREE.Vector3();
        if (view === ViewType.FRONT) {
            const diffX = currentPointCoordOnPlane.x - this.action.resize.previousPosition.x;
            const diffY = currentPointCoordOnPlane.y - this.action.resize.previousPosition.y;
            currentPointInternalCoordinates
                .fromArray(cuboidNodes[currentPointNumber]).add(new THREE.Vector3(0, diffY, -diffX).divide(scale));
        } else if (view === ViewType.SIDE) {
            const diffX = currentPointCoordOnPlane.x - this.action.resize.previousPosition.x;
            const diffY = currentPointCoordOnPlane.y - this.action.resize.previousPosition.y;
            currentPointInternalCoordinates
                .fromArray(cuboidNodes[currentPointNumber]).add(new THREE.Vector3(-diffX, 0, diffY).divide(scale));
        } else if (view === ViewType.TOP) {
            const diffX = currentPointCoordOnPlane.x - this.action.resize.previousPosition.x;
            const diffY = currentPointCoordOnPlane.y - this.action.resize.previousPosition.y;
            currentPointInternalCoordinates
                .fromArray(cuboidNodes[currentPointNumber]).add(new THREE.Vector3(diffX, diffY, 0).divide(scale));
        }
        const perspectivePosition = cuboid.perspective.localToWorld(currentPointInternalCoordinates.clone());

        // small check to avoid case when points change their relative orientation
        if (
            Math.sign(crosslyingPointInternalCoordinates.x - cuboidNodes[currentPointNumber][0]) !==
            Math.sign(crosslyingPointInternalCoordinates.x - currentPointInternalCoordinates.x) ||
            Math.sign(crosslyingPointInternalCoordinates.y - cuboidNodes[currentPointNumber][1]) !==
            Math.sign(crosslyingPointInternalCoordinates.y - currentPointInternalCoordinates.y) ||
            Math.sign(crosslyingPointInternalCoordinates.z - cuboidNodes[currentPointNumber][2]) !==
            Math.sign(crosslyingPointInternalCoordinates.z - currentPointInternalCoordinates.z)
        ) {
            return;
        }

        // finally let's compute new center and scale
        scale.x *= Math.abs(crosslyingPointInternalCoordinates.x - currentPointInternalCoordinates.x);
        scale.y *= Math.abs(crosslyingPointInternalCoordinates.y - currentPointInternalCoordinates.y);
        scale.z *= Math.abs(crosslyingPointInternalCoordinates.z - currentPointInternalCoordinates.z);
        const newPosition = crosslyingPointCoordinates.clone().add(perspectivePosition).divideScalar(2);

        // and apply them
        this.moveObject(newPosition);
        cuboid.setScale(scale.x, scale.y, scale.z);
        this.adjustPerspectiveCameras();
        this.updateCameraFrustumPlane(view);

        this.action.resize.previousPosition = currentPointCoordOnPlane;
    }

    private static isLeft(a: any, b: any, c: any): boolean {
        // For reference
        // A
        // |\                // A = Rotation Center
        // | \               // B = Previous Frame Position
        // |  C              // C = Current Frame Position
        // B
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) > 0;
    }

    private rotateCube(instance: CuboidModel, direction: number, view: ViewType): void {
        switch (view) {
            case ViewType.TOP:
                instance.perspective.rotateZ(direction);
                instance.top.rotateZ(direction);
                instance.side.rotateZ(direction);
                instance.front.rotateZ(direction);
                this.rotateCamera(direction, view);
                break;
            case ViewType.FRONT:
                instance.perspective.rotateX(direction);
                instance.top.rotateX(direction);
                instance.side.rotateX(direction);
                instance.front.rotateX(direction);
                this.rotateCamera(direction, view);
                break;
            case ViewType.SIDE:
                instance.perspective.rotateY(direction);
                instance.top.rotateY(direction);
                instance.side.rotateY(direction);
                instance.front.rotateY(direction);
                this.rotateCamera(direction, view);
                break;
            default:
        }
    }

    private rotateCamera(direction: any, view: ViewType): void {
        switch (view) {
            case ViewType.TOP:
                this.views.top.camera.rotateZ(direction);
                break;
            case ViewType.FRONT:
                this.views.front.camera.rotateZ(direction);
                break;
            case ViewType.SIDE:
                this.views.side.camera.rotateZ(direction);
                break;
            default:
        }
    }

    private detachCamera(view?: ViewType): void {
        const cuboid = this.selectedCuboid;
        if (!cuboid) {
            return;
        }

        const coordTop = cuboid.getReferenceCoordinates(ViewType.TOP);
        const sphericaltop = new THREE.Spherical();
        sphericaltop.setFromVector3(coordTop);

        const coordSide = cuboid.getReferenceCoordinates(ViewType.SIDE);
        const sphericalside = new THREE.Spherical();
        sphericalside.setFromVector3(coordSide);

        const coordFront = cuboid.getReferenceCoordinates(ViewType.FRONT);
        const sphericalfront = new THREE.Spherical();
        sphericalfront.setFromVector3(coordFront);

        const { side: objectSideView, front: objectFrontView, top: objectTopView } = cuboid;
        const sideWorld = objectSideView.getWorldPosition(new THREE.Vector3());
        const frontWorld = objectFrontView.getWorldPosition(new THREE.Vector3());
        const topWorld = objectTopView.getWorldPosition(new THREE.Vector3());

        const { camera: sideCamera } = this.views.side;
        const { camera: frontCamera } = this.views.front;
        const { camera: topCamera } = this.views.top;

        const camFrontRotate = objectFrontView
            .getObjectByName('camRefRot')
            .getWorldQuaternion(new THREE.Quaternion());
        switch (view) {
            case ViewType.TOP: {
                sideCamera.position.setFromSpherical(sphericalside);
                sideCamera.lookAt(objectSideView.position.x, objectSideView.position.y, objectSideView.position.z);
                sideCamera.rotation.z = this.views.side.scene.getObjectByName(Planes.SIDE).rotation.z;
                sideCamera.scale.set(1, 1, 1);

                frontCamera.position.setFromSpherical(sphericalfront);
                frontCamera.lookAt(objectFrontView.position.x, objectFrontView.position.y, objectFrontView.position.z);
                frontCamera.setRotationFromQuaternion(camFrontRotate);
                frontCamera.scale.set(1, 1, 1);
                break;
            }
            case ViewType.SIDE: {
                frontCamera.position.setFromSpherical(sphericalfront);
                frontCamera.lookAt(objectFrontView.position.x, objectFrontView.position.y, objectFrontView.position.z);
                frontCamera.setRotationFromQuaternion(camFrontRotate);
                frontCamera.scale.set(1, 1, 1);

                topCamera.position.setFromSpherical(sphericaltop);
                topCamera.lookAt(objectTopView.position.x, objectTopView.position.y, objectTopView.position.z);
                topCamera.setRotationFromEuler(objectTopView.rotation);
                topCamera.scale.set(1, 1, 1);
                break;
            }
            case ViewType.FRONT: {
                sideCamera.position.setFromSpherical(sphericalside);
                sideCamera.lookAt(objectSideView.position.x, objectSideView.position.y, objectSideView.position.z);
                sideCamera.rotation.z = this.views.side.scene.getObjectByName(Planes.SIDE).rotation.z;
                sideCamera.scale.set(1, 1, 1);

                topCamera.position.setFromSpherical(sphericaltop);
                topCamera.lookAt(objectTopView.position.x, objectTopView.position.y, objectTopView.position.z);
                topCamera.setRotationFromEuler(objectTopView.rotation);
                topCamera.scale.set(1, 1, 1);
                break;
            }
            default: {
                sideCamera.position.setFromSpherical(sphericalside);
                sideCamera.lookAt(sideWorld.x, sideWorld.y, sideWorld.z);
                sideCamera.rotation.z = this.views.side.scene.getObjectByName(Planes.SIDE).rotation.z;
                sideCamera.scale.set(1, 1, 1);

                topCamera.position.setFromSpherical(sphericaltop);
                topCamera.lookAt(topWorld.x, topWorld.y, topWorld.z);
                topCamera.setRotationFromEuler(objectTopView.rotation);
                topCamera.scale.set(1, 1, 1);

                frontCamera.position.setFromSpherical(sphericalfront);
                frontCamera.lookAt(frontWorld.x, frontWorld.y, frontWorld.z);
                frontCamera.setRotationFromQuaternion(camFrontRotate);
                frontCamera.scale.set(1, 1, 1);
            }
        }
    }

    private rotatePlane(direction: number, view: ViewType): void {
        const sceneTopPlane = this.scenePlanes[ViewType.TOP] || this.views.top.scene.getObjectByName(Planes.TOP);
        const sceneSidePlane = this.scenePlanes[ViewType.SIDE] || this.views.side.scene.getObjectByName(Planes.SIDE);
        const sceneFrontPlane = this.scenePlanes[ViewType.FRONT] || this.views.front.scene.getObjectByName(Planes.FRONT);
        if (!sceneTopPlane || !sceneSidePlane || !sceneFrontPlane) return;
        switch (view) {
            case ViewType.TOP:
                sceneTopPlane.rotateZ(direction);
                sceneSidePlane.rotateY(direction);
                sceneFrontPlane.rotateX(-direction);
                break;
            case ViewType.SIDE:
                sceneTopPlane.rotateY(direction);
                sceneSidePlane.rotateZ(direction);
                sceneFrontPlane.rotateY(direction);
                break;
            case ViewType.FRONT:
                sceneTopPlane.rotateX(direction);
                sceneSidePlane.rotateX(-direction);
                sceneFrontPlane.rotateZ(direction);
                break;
            default: {
                const cuboid = this.selectedCuboid;
                if (!cuboid) {
                    return;
                }

                const { top: objectTopView, side: objectSideView, front: objectFrontView } = cuboid;

                const quaternionSide = objectSideView.getObjectByName(consts.PLANE_ROTATION_HELPER)
                    .getWorldQuaternion(new THREE.Quaternion());
                const rotationSide = new THREE.Euler().setFromQuaternion(quaternionSide);

                const quaternionFront = objectFrontView.getObjectByName(consts.PLANE_ROTATION_HELPER)
                    .getWorldQuaternion(new THREE.Quaternion());
                const rotationFront = new THREE.Euler().setFromQuaternion(quaternionFront);

                const quaternionTop = objectTopView.getObjectByName(consts.PLANE_ROTATION_HELPER)
                    .getWorldQuaternion(new THREE.Quaternion());
                const rotationTop = new THREE.Euler().setFromQuaternion(quaternionTop);

                const coordinates = {
                    x: objectTopView.position.x,
                    y: objectTopView.position.y,
                    z: objectTopView.position.z,
                };

                sceneTopPlane.rotation.set(rotationTop.x, rotationTop.y, rotationTop.z);
                sceneSidePlane.rotation.set(rotationSide.x, rotationSide.y, rotationSide.z);
                sceneFrontPlane.rotation.set(rotationFront.x, rotationFront.y, rotationFront.z);

                this.translateReferencePlane(coordinates);
            }
        }
    }

    private renderRotateAction(view: ViewType, viewType: any): void {
        const { renderer } = viewType;
        const canvas = renderer.domElement;
        if (!canvas) return;

        const canvasCentre = {
            x: canvas.offsetLeft + canvas.offsetWidth / 2,
            y: canvas.offsetTop + canvas.offsetHeight / 2,
        };

        if (
            this.action.rotation.screenInit.x === this.action.rotation.screenMove.x &&
            this.action.rotation.screenInit.y === this.action.rotation.screenMove.y
        ) {
            return;
        }

        const startVector = {
            x: this.action.rotation.screenInit.x - canvasCentre.x,
            y: this.action.rotation.screenInit.y - canvasCentre.y,
        };
        const endVector = {
            x: this.action.rotation.screenMove.x - canvasCentre.x,
            y: this.action.rotation.screenMove.y - canvasCentre.y,
        };

        const startMagnitude = Math.sqrt(startVector.x ** 2 + startVector.y ** 2);
        const endMagnitude = Math.sqrt(endVector.x ** 2 + endVector.y ** 2);

        if (startMagnitude === 0 || endMagnitude === 0) {
            return;
        }

        const dotProduct = startVector.x * endVector.x + startVector.y * endVector.y;
        let angle = Math.acos(dotProduct / (startMagnitude * endMagnitude));

        if (Canvas3dViewImpl.isLeft(canvasCentre, this.action.rotation.screenInit, this.action.rotation.screenMove)) {
            angle = -angle;
        }

        this.action.rotation.recentMouseVector = this.views[view].rayCaster.mouseVector.clone();
        this.rotateCube(this.selectedCuboid, angle, view);
        this.rotatePlane(angle, view);

        this.updateResizeHelperPos();
        this.updateRotationHelperPos();
        this.detachCamera();
        this.updateCameraFrustumPlane();

        this.action.rotation.screenInit.x = this.action.rotation.screenMove.x;
        this.action.rotation.screenInit.y = this.action.rotation.screenMove.y;
    }

    private initiateAction(view: ViewType, viewType: any): void {
        const { clientID } = this.model.data.activeElement;
        const { cuboid, data } = this.drawnObjects[+clientID] || {};
        if (!data || !cuboid || data.lock) return;

        const plane = this.scenePlanes[view] || viewType.scene.getObjectByName(`${view}Plane`);

        const intersectsHelperResize = viewType.rayCaster.renderer.intersectObjects(
            cuboid[view].parent.children
                .filter((child: THREE.Object3D) => child.name.startsWith(consts.RESIZE_HELPER_NAME)),
            false,
        );

        const intersectsPlane = viewType.rayCaster.renderer.intersectObjects(
            plane ? [plane] : [],
            false,
        );

        if (intersectsHelperResize.length !== 0 && intersectsPlane.length !== 0) {
            this.action.detected = true;
            this.views.top.controls.enabled = false;
            this.views.side.controls.enabled = false;
            this.views.front.controls.enabled = false;
            this.action.resize.status = true;
            this.action.resize.helperElement = intersectsHelperResize[0].object;
            this.action.resize.previousPosition = null;
            return;
        }

        const intersectsHelperRotation = viewType.rayCaster.renderer.intersectObjects(
            cuboid[view].parent.children
                .filter((child: THREE.Object3D) => child.name.startsWith(consts.ROTATION_HELPER_NAME)),
            false,
        );
        if (intersectsHelperRotation.length !== 0) {
            this.action.rotation.helper = viewType.rayCaster.mouseVector.clone();
            this.action.rotation.status = true;
            this.action.detected = true;
            this.views.top.controls.enabled = false;
            this.views.side.controls.enabled = false;
            this.views.front.controls.enabled = false;
            return;
        }

        const intersectsBox = viewType.rayCaster.renderer.intersectObjects([cuboid[view]], false);
        const intersectsPointCloud = viewType.rayCaster.renderer.intersectObjects(
            plane ? [plane] : [],
            true,
        );
        if (intersectsBox.length !== 0 && !data.pinned) {
            this.action.translation.helper = viewType.rayCaster.mouseVector.clone();
            this.action.translation.inverseMatrix = intersectsBox[0].object.parent.matrixWorld.invert();
            this.action.translation.offset = intersectsPointCloud[0].point.sub(
                new THREE.Vector3().setFromMatrixPosition(intersectsBox[0].object.matrixWorld),
            );
            this.action.translation.status = true;
            this.action.detected = true;
            this.views.top.controls.enabled = false;
            this.views.side.controls.enabled = false;
            this.views.front.controls.enabled = false;
        }
    }

    private updateOrthoPointCloudROI(force: boolean = false): void {
        if (!this.orthoPointsGeometry || !this.pointCloudPoints) return;
        const activeClientID = this.getActiveElement();
        if (activeClientID === null || !(activeClientID in this.drawnObjects)) {
            this.perfOrthoRoiPoints = this.perfPointCount;
            if (this.orthoRoiSphereUniform.value.w !== 0) {
                this.orthoRoiSphereUniform.value.set(0, 0, 0, 0);
                this.lastRoiCenter = null;
                this.requestOrthoRender();
            }
            return;
        }

        const { cuboid } = this.drawnObjects[activeClientID];
        if (!cuboid) return;

        const pos = cuboid.perspective.position;
        const scale = cuboid.perspective.scale;

        if (!force && this.lastRoiCenter) {
            const distSq = pos.distanceToSquared(this.lastRoiCenter);
            if (distSq < 0.04) return;
        }

        const cx = pos.x;
        const cy = pos.y;
        const cz = pos.z;
        const maxDim = Math.max(scale.x, scale.y, scale.z);
        const radius = maxDim / 2 + this.gpuCapabilities.orthoRoiRadiusPadding;

        this.lastRoiCenter = pos.clone();
        this.orthoRoiSphereUniform.value.set(cx, cy, cz, radius);
        this.perfOrthoRoiPoints = this.perfPointCount;
        this.requestOrthoRender();
    }

    public keyControls(key: any): void {
        if (!key.ctrlKey && !key.altKey && !key.metaKey && !key.shiftKey && key.code === 'KeyG') {
            this.toggleGroundStripping();
            key.preventDefault?.();
            key.stopPropagation?.();
            return;
        }

        const activeClientID = this.getActiveElement();

        // Object micro-nudging and hotkeys when a cuboid is active
        if (activeClientID !== null && activeClientID in this.drawnObjects) {
            const step = key.altKey ? 0.20 : 0.05;
            const zStep = (key.altKey || key.shiftKey) ? 0.10 : 0.02;
            const yawStepRad = (key.altKey || key.shiftKey) ? ((5 * Math.PI) / 180) : (Math.PI / 180);

            // Plain or Alt Arrow keys: nudge along vehicle local longitudinal (X) and lateral (Y) axes
            // (Leave Shift+Arrow free for perspective camera roll/tilt below)
            if (!key.shiftKey) {
                if (key.code === 'ArrowUp') {
                    if (this.nudgeCuboid(step, 0, 0, 0, activeClientID)) {
                        key.preventDefault?.();
                        key.stopPropagation?.();
                        return;
                    }
                } else if (key.code === 'ArrowDown') {
                    if (this.nudgeCuboid(-step, 0, 0, 0, activeClientID)) {
                        key.preventDefault?.();
                        key.stopPropagation?.();
                        return;
                    }
                } else if (key.code === 'ArrowLeft') {
                    if (this.nudgeCuboid(0, step, 0, 0, activeClientID)) {
                        key.preventDefault?.();
                        key.stopPropagation?.();
                        return;
                    }
                } else if (key.code === 'ArrowRight') {
                    if (this.nudgeCuboid(0, -step, 0, 0, activeClientID)) {
                        key.preventDefault?.();
                        key.stopPropagation?.();
                        return;
                    }
                }
            }

            // Elevation nudges: PageUp / PageDown (or Alt+KeyE / Alt+KeyQ)
            if (key.code === 'PageUp' || (key.altKey && key.code === 'KeyE')) {
                if (this.nudgeCuboid(0, 0, zStep, 0, activeClientID)) {
                    key.preventDefault?.();
                    key.stopPropagation?.();
                    return;
                }
            } else if (key.code === 'PageDown' || (key.altKey && key.code === 'KeyQ')) {
                if (this.nudgeCuboid(0, 0, -zStep, 0, activeClientID)) {
                    key.preventDefault?.();
                    key.stopPropagation?.();
                    return;
                }
            }

            // Yaw micro-rotation: BracketLeft [ and BracketRight ]
            if (key.code === 'BracketLeft') {
                if (this.nudgeCuboid(0, 0, 0, -yawStepRad, activeClientID)) {
                    key.preventDefault?.();
                    key.stopPropagation?.();
                    return;
                }
            } else if (key.code === 'BracketRight') {
                if (this.nudgeCuboid(0, 0, 0, yawStepRad, activeClientID)) {
                    key.preventDefault?.();
                    key.stopPropagation?.();
                    return;
                }
            }
        }

        const { controls } = this.views.perspective;
        if (!controls) return;
        if (key.shiftKey) {
            switch (key.code) {
                case CameraAction.ROTATE_RIGHT:
                    controls.rotate(0.1 * THREE.MathUtils.DEG2RAD * this.speed, 0, true);
                    break;
                case CameraAction.ROTATE_LEFT:
                    controls.rotate(-0.1 * THREE.MathUtils.DEG2RAD * this.speed, 0, true);
                    break;
                case CameraAction.TILT_UP:
                    controls.rotate(0, -0.05 * THREE.MathUtils.DEG2RAD * this.speed, true);
                    break;
                case CameraAction.TILT_DOWN:
                    controls.rotate(0, 0.05 * THREE.MathUtils.DEG2RAD * this.speed, true);
                    break;
                default:
                    break;
            }
            this.requestPerspectiveRender();
        } else if (key.altKey === true) {
            switch (key.code) {
                case CameraAction.ZOOM_IN:
                    controls.dolly(consts.DOLLY_FACTOR, true);
                    break;
                case CameraAction.ZOOM_OUT:
                    controls.dolly(-consts.DOLLY_FACTOR, true);
                    break;
                case CameraAction.MOVE_LEFT:
                    controls.truck(-0.01 * this.speed, 0, true);
                    break;
                case CameraAction.MOVE_RIGHT:
                    controls.truck(0.01 * this.speed, 0, true);
                    break;
                case CameraAction.MOVE_DOWN:
                    controls.truck(0, -0.01 * this.speed, true);
                    break;
                case CameraAction.MOVE_UP:
                    controls.truck(0, 0.01 * this.speed, true);
                    break;
                default:
                    break;
            }
            this.requestPerspectiveRender();
        }
    }

    public html(): ViewsDOM {
        return {
            perspective: this.views.perspective.renderer.domElement,
            top: this.views.top.renderer.domElement,
            side: this.views.side.renderer.domElement,
            front: this.views.front.renderer.domElement,
        };
    }

    public destroy(): void {
        if (this.onDocumentMouseUp) {
            window.document.removeEventListener('mouseup', this.onDocumentMouseUp);
            this.onDocumentMouseUp = null;
        }
        if (this.perfHudElement) {
            this.perfHudElement.remove();
            this.perfHudElement = null;
        }
        try {
            this.clearScene();
        } catch (_) {
            // ignore
        }
        this.geometryCache.forEach((entry) => {
            entry.geometry.dispose();
        });
        this.geometryCache.clear();
        Object.values(this.views).forEach((view) => {
            view.controls?.dispose();
        });
        this.mainRenderer?.dispose();
    }
}
