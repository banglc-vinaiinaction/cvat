// Copyright (C) 2021-2022 Intel Corporation
// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { Canvas3dController, Canvas3dControllerImpl } from './canvas3dController';
import {
    Canvas3dModel,
    Canvas3dModelImpl,
    Mode,
    DrawData,
    ViewType,
    MouseInteraction,
    Configuration,
    GroupData,
    SplitData,
    MergeData,
} from './canvas3dModel';
import {
    Canvas3dView, Canvas3dViewImpl, ViewsDOM, CameraAction, Canvas3dPerformanceStats,
} from './canvas3dView';
import { Master } from './master';

interface Canvas3d {
    html(): ViewsDOM;
    setup(frameData: any, objectStates: any[]): void;
    isAbleToChangeFrame(): boolean;
    mode(): Mode;
    render(): void;
    requestRender(frameCount?: number): void;
    keyControls(keys: KeyboardEvent): void;
    draw(drawData: DrawData): void;
    cancel(): void;
    dragCanvas(enable: boolean): void;
    activate(clientID: number | null, attributeID?: number): void;
    configure(configuration: Configuration): void;
    fitCanvas(): void;
    fit(): void;
    focus(clientID: number): void;
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
    isObjectIsolated(): boolean;
    toggleGroundStripping(elevationOffset?: number): boolean;
    isGroundStripped(): boolean;
    snapTrackToRoad(
        jobInstance: any,
        clientID?: number | null,
        onProgress?: (message: string, percent: number) => void,
    ): Promise<{ processedFrames: number }>;
    getPerformanceStats(): Canvas3dPerformanceStats;
    togglePerformanceHud(enabled?: boolean): boolean;
    group(groupData: GroupData): void;
    merge(mergeData: MergeData): void;
    split(splitData: SplitData): void;
    destroy(): void;
}

class Canvas3dImpl implements Canvas3d {
    private readonly model: Canvas3dModel & Master;
    private readonly controller: Canvas3dController;
    private view: Canvas3dView;

    public constructor() {
        this.model = new Canvas3dModelImpl();
        this.controller = new Canvas3dControllerImpl(this.model);
        this.view = new Canvas3dViewImpl(this.model, this.controller);
    }

    public html(): ViewsDOM {
        return this.view.html();
    }

    public keyControls(keys: KeyboardEvent): void {
        this.view.keyControls(keys);
    }

    public render(): void {
        this.view.render();
    }

    public requestRender(frameCount?: number): void {
        this.view.requestRender(frameCount);
    }

    public draw(drawData: DrawData): void {
        this.model.draw(drawData);
    }

    public setup(frameData: any, objectStates: any[]): void {
        this.model.setup(frameData, objectStates);
    }

    public mode(): Mode {
        return this.model.mode;
    }

    public group(groupData: GroupData): void {
        this.model.group(groupData);
    }

    public split(splitData: SplitData): void {
        this.model.split(splitData);
    }

    public merge(mergeData: MergeData): void {
        this.model.merge(mergeData);
    }

    public isAbleToChangeFrame(): boolean {
        return this.model.isAbleToChangeFrame();
    }

    public cancel(): void {
        this.model.cancel();
    }

    public dragCanvas(enable: boolean): void {
        this.model.dragCanvas(enable);
    }

    public configure(configuration: Partial<Configuration>): void {
        this.model.configure(configuration);
    }

    public activate(clientID: number | null, attributeID: number | null = null): void {
        this.model.activate(typeof clientID === 'number' ? String(clientID) : null, attributeID);
    }

    public focus(clientID: number): void {
        this.view.focusObjectByClientId(clientID, true);
    }

    public snapToRoad(clientID?: number | null, options?: { snapGround?: boolean; autoOrient?: boolean; snapGeometry?: boolean; }): void {
        this.view.snapToRoad(clientID, options);
    }

    public revertSnap(clientID?: number | null): boolean {
        return this.view.revertSnap(clientID);
    }

    public getPreviousCuboid(clientID?: number | null): number[] | null {
        return this.view.getPreviousCuboid(clientID);
    }

    public invertHeading(clientID?: number | null): void {
        this.view.invertHeading(clientID);
    }

    public getActiveElement(): number | null {
        return this.view.getActiveElement();
    }

    public setCuboidDimensions(clientID: number | null, dimensions: [number, number, number]): void {
        this.view.setCuboidDimensions(clientID, dimensions);
    }

    public nudgeCuboid(
        dxLocal: number,
        dyLocal: number,
        dzWorld: number,
        dYawRad: number,
        clientID?: number | null,
    ): boolean {
        return this.view.nudgeCuboid(dxLocal, dyLocal, dzWorld, dYawRad, clientID);
    }

    public toggleObjectIsolation(clientID?: number | null): boolean {
        return this.view.toggleObjectIsolation(clientID);
    }

    public isObjectIsolated(): boolean {
        return this.view.isObjectIsolatedState();
    }

    public toggleGroundStripping(elevationOffset?: number): boolean {
        return this.view.toggleGroundStripping(elevationOffset);
    }

    public isGroundStripped(): boolean {
        return this.view.isGroundStrippedState();
    }

    public async snapTrackToRoad(
        jobInstance: any,
        clientID?: number | null,
        onProgress?: (message: string, percent: number) => void,
    ): Promise<{ processedFrames: number }> {
        return this.view.snapTrackToRoad(jobInstance, clientID, onProgress);
    }

    public getPerformanceStats(): Canvas3dPerformanceStats {
        return this.view.getPerformanceStats();
    }

    public togglePerformanceHud(enabled?: boolean): boolean {
        return this.view.togglePerformanceHud(enabled);
    }

    public fit(): void {
        this.model.fit();
    }

    public fitCanvas(): void {
        // in spite of 2D canvas, 3D version fits automatically when external container resized
        // so, nothing to do here, but keep the method to keep the same interface as 2D canvas
    }

    public destroy(): void {
        this.view.destroy();
        this.model.destroy();
    }
}

export {
    Canvas3dImpl as Canvas3d, ViewType, MouseInteraction, CameraAction, Mode as CanvasMode,
};

export type { ViewsDOM, Canvas3dPerformanceStats };
