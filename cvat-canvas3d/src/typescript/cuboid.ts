// Copyright (C) 2021-2022 Intel Corporation
// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import * as THREE from 'three';
import { OrientationVisibility, ViewType } from './canvas3dModel';
import constants from './consts';
import getCircleTexture from './controlPointTexture';
import { disposeObject3D, disposeObjectResources } from './utils';

export interface Indexable {
    [key: string]: any;
}

export interface ObjectArrowHelper {
    x: THREE.ArrowHelper;
    y: THREE.ArrowHelper;
    z: THREE.ArrowHelper;
}

export function makeCornerPointsMatrix(x: number, y: number, z: number): number[][] {
    return ([
        [1 * x, 1 * y, 1 * z],
        [1 * x, 1 * y, -1 * z],
        [1 * x, -1 * y, 1 * z],
        [1 * x, -1 * y, -1 * z],
        [-1 * x, 1 * y, 1 * z],
        [-1 * x, 1 * y, -1 * z],
        [-1 * x, -1 * y, 1 * z],
        [-1 * x, -1 * y, -1 * z],
    ]);
}

const SHARED_BOX_GEOMETRY = new THREE.BoxGeometry(1, 1, 1);
SHARED_BOX_GEOMETRY.computeBoundingBox();
SHARED_BOX_GEOMETRY.computeBoundingSphere();
SHARED_BOX_GEOMETRY.userData.shared = true;

const SHARED_EDGES_GEOMETRY = new THREE.EdgesGeometry(SHARED_BOX_GEOMETRY);
SHARED_EDGES_GEOMETRY.computeBoundingBox();
SHARED_EDGES_GEOMETRY.computeBoundingSphere();
SHARED_EDGES_GEOMETRY.userData.shared = true;

const SHARED_DASHED_EDGES_GEOMETRY = new THREE.EdgesGeometry(SHARED_BOX_GEOMETRY);
const dummyDashedLines = new THREE.LineSegments(SHARED_DASHED_EDGES_GEOMETRY);
dummyDashedLines.computeLineDistances();
SHARED_DASHED_EDGES_GEOMETRY.computeBoundingBox();
SHARED_DASHED_EDGES_GEOMETRY.computeBoundingSphere();
SHARED_DASHED_EDGES_GEOMETRY.userData.shared = true;

const SHARED_PLANE_GEOMETRY = new THREE.PlaneGeometry(1, 1, 1, 1);
SHARED_PLANE_GEOMETRY.computeBoundingBox();
SHARED_PLANE_GEOMETRY.computeBoundingSphere();
SHARED_PLANE_GEOMETRY.userData.shared = true;

const SHARED_SPHERE_GEOMETRY = new THREE.SphereGeometry(0.1);
SHARED_SPHERE_GEOMETRY.computeBoundingBox();
SHARED_SPHERE_GEOMETRY.computeBoundingSphere();
SHARED_SPHERE_GEOMETRY.userData.shared = true;

const SHARED_HIDDEN_MATERIAL = new THREE.MeshBasicMaterial({
    color: 0xff0000,
    visible: false,
});
SHARED_HIDDEN_MATERIAL.userData.shared = true;

const SHARED_CORNER_POINTS = makeCornerPointsMatrix(0.5, 0.5, 0.5).map(
    (point) => new THREE.Vector3().fromArray(point),
);

function createTemplateArrow(dir: THREE.Vector3, origin: THREE.Vector3, color: number): THREE.ArrowHelper {
    const arrow = new THREE.ArrowHelper(dir, origin, 1, color);
    arrow.line.geometry.computeBoundingBox();
    arrow.line.geometry.computeBoundingSphere();
    arrow.line.geometry.userData.shared = true;
    (arrow.line.material as THREE.Material).userData.shared = true;

    arrow.cone.geometry.computeBoundingBox();
    arrow.cone.geometry.computeBoundingSphere();
    arrow.cone.geometry.userData.shared = true;
    (arrow.cone.material as THREE.Material).userData.shared = true;
    return arrow;
}

const TEMPLATE_ARROW_X = createTemplateArrow(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0.5, 0, 0), 0xff0000);
const TEMPLATE_ARROW_Y = createTemplateArrow(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0.5, 0), 0x00ff00);
const TEMPLATE_ARROW_Z = createTemplateArrow(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0.5), 0x0000ff);

function cloneSharedArrow(template: THREE.ArrowHelper): THREE.ArrowHelper {
    const arrow = new THREE.Object3D() as THREE.ArrowHelper;
    Object.setPrototypeOf(arrow, THREE.ArrowHelper.prototype);
    (arrow as any).type = 'ArrowHelper';
    arrow.position.copy(template.position);
    arrow.quaternion.copy(template.quaternion);
    arrow.scale.copy(template.scale);
    arrow.visible = false;

    const line = new THREE.Line(template.line.geometry, template.line.material);
    line.matrixAutoUpdate = false;
    line.scale.copy(template.line.scale);
    line.matrix.copy(template.line.matrix);
    arrow.line = line;
    arrow.add(line);

    const cone = new THREE.Mesh(template.cone.geometry, template.cone.material);
    cone.matrixAutoUpdate = false;
    cone.position.copy(template.cone.position);
    cone.scale.copy(template.cone.scale);
    cone.matrix.copy(template.cone.matrix);
    arrow.cone = cone;
    arrow.add(cone);

    return arrow;
}

const TEMPLATE_CAM_ROTATE_HELPER = new THREE.Object3D();
TEMPLATE_CAM_ROTATE_HELPER.translateX(-2);
TEMPLATE_CAM_ROTATE_HELPER.name = 'camRefRot';
TEMPLATE_CAM_ROTATE_HELPER.up = new THREE.Vector3(0, 0, 1);
TEMPLATE_CAM_ROTATE_HELPER.lookAt(new THREE.Vector3(0, 0, 0));

export class CuboidModel {
    public perspective: THREE.Mesh;
    public top: THREE.Mesh;
    public side: THREE.Mesh;
    public front: THREE.Mesh;
    public wireframe: THREE.LineSegments;

    public orientationArrows: Record<ViewType, ObjectArrowHelper> = {
        [ViewType.PERSPECTIVE]: null,
        [ViewType.TOP]: null,
        [ViewType.SIDE]: null,
        [ViewType.FRONT]: null,
    };

    public constructor(outline: string, outlineColor: string) {
        const geometry = SHARED_BOX_GEOMETRY;
        const material = new THREE.MeshBasicMaterial({
            color: 0x00ff00,
            wireframe: false,
            transparent: true,
            opacity: 0.4,
        });

        this.perspective = new THREE.Mesh(geometry, material);
        this.perspective.frustumCulled = true;
        this.perspective.matrixAutoUpdate = true;

        const isLine = outline === 'line';
        const geo = isLine ? SHARED_EDGES_GEOMETRY : SHARED_DASHED_EDGES_GEOMETRY;
        this.wireframe = new THREE.LineSegments(
            geo,
            isLine ? new THREE.LineBasicMaterial({ color: outlineColor, linewidth: 4 }) :
                new THREE.LineDashedMaterial({
                    color: outlineColor,
                    dashSize: 0.05,
                    gapSize: 0.05,
                }),
        );
        this.wireframe.frustumCulled = true;
        this.wireframe.renderOrder = 1;
        this.perspective.add(this.wireframe);

        this.top = new THREE.Mesh(geometry, material);
        this.top.frustumCulled = true;
        this.top.matrixAutoUpdate = true;

        this.side = new THREE.Mesh(geometry, material);
        this.side.frustumCulled = true;
        this.side.matrixAutoUpdate = true;

        this.front = new THREE.Mesh(geometry, material);
        this.front.frustumCulled = true;
        this.front.matrixAutoUpdate = true;

        [ViewType.PERSPECTIVE, ViewType.TOP, ViewType.SIDE, ViewType.FRONT].forEach((view): void => {
            this.orientationArrows[view] = this.createArrows();
            Object.values(this.orientationArrows[view]).forEach((arrow) => {
                this[view].add(arrow);
            });
        });

        const planeTop = new THREE.Mesh(SHARED_PLANE_GEOMETRY, SHARED_HIDDEN_MATERIAL);
        const planeSide = new THREE.Mesh(SHARED_PLANE_GEOMETRY, SHARED_HIDDEN_MATERIAL);
        const planeFront = new THREE.Mesh(SHARED_PLANE_GEOMETRY, SHARED_HIDDEN_MATERIAL);

        this.top.add(planeTop);
        planeTop.rotation.set(0, 0, 0);
        planeTop.position.set(0, 0, 0.5);
        planeTop.name = constants.PLANE_ROTATION_HELPER;

        this.side.add(planeSide);
        planeSide.rotation.set(-Math.PI / 2, 0, Math.PI);
        planeSide.position.set(0, 0.5, 0);
        planeSide.name = constants.PLANE_ROTATION_HELPER;

        this.front.add(planeFront);
        planeFront.rotation.set(0, Math.PI / 2, 0);
        planeFront.position.set(0.5, 0, 0);
        planeFront.name = constants.PLANE_ROTATION_HELPER;

        for (let i = 0; i < SHARED_CORNER_POINTS.length; i++) {
            const helper = new THREE.Mesh(SHARED_SPHERE_GEOMETRY, SHARED_HIDDEN_MATERIAL);
            helper.visible = false;
            helper.name = `cuboidNodeHelper_${i}`;
            helper.position.copy(SHARED_CORNER_POINTS[i]);
            this.perspective.add(helper);
        }

        const camRotateHelper = new THREE.Object3D();
        camRotateHelper.name = 'camRefRot';
        camRotateHelper.position.copy(TEMPLATE_CAM_ROTATE_HELPER.position);
        camRotateHelper.quaternion.copy(TEMPLATE_CAM_ROTATE_HELPER.quaternion);
        camRotateHelper.up.copy(TEMPLATE_CAM_ROTATE_HELPER.up);
        this.front.add(camRotateHelper);
    }

    private createArrows(): ObjectArrowHelper {
        return {
            x: cloneSharedArrow(TEMPLATE_ARROW_X),
            y: cloneSharedArrow(TEMPLATE_ARROW_Y),
            z: cloneSharedArrow(TEMPLATE_ARROW_Z),
        };
    }

    public getRotationHelperPosition(viewType: ViewType): THREE.Vector3 {
        const position = viewType === ViewType.TOP ?
            new THREE.Vector3(0, constants.ROTATION_HELPER_OFFSET, 0) :
            new THREE.Vector3(0, 0, constants.ROTATION_HELPER_OFFSET);
        return this[viewType].localToWorld(position);
    }

    public getResizeHelperPositions(): THREE.Vector3[] {
        return SHARED_CORNER_POINTS.map((point) => this.perspective.localToWorld(point.clone()));
    }

    public setOrientationVisibility(orientationVisibility: OrientationVisibility): void {
        [ViewType.PERSPECTIVE, ViewType.TOP, ViewType.SIDE, ViewType.FRONT].forEach((view): void => {
            Object.entries(this.orientationArrows[view]).forEach(([axis, arrow]) => {
                // eslint-disable-next-line no-param-reassign
                arrow.visible = orientationVisibility[axis];
            });
        });
    }

    public setPosition(x: number, y: number, z: number): void {
        [ViewType.PERSPECTIVE, ViewType.TOP, ViewType.SIDE, ViewType.FRONT].forEach((view): void => {
            (this as Indexable)[view].position.set(x, y, z);
        });
    }

    public setScale(x: number, y: number, z: number): void {
        [ViewType.PERSPECTIVE, ViewType.TOP, ViewType.SIDE, ViewType.FRONT].forEach((view): void => {
            (this as Indexable)[view].scale.set(x, y, z);

            // Arrow direction specifies its local Y axis, where it points to.
            // When we change its direction to align with the X or Z axis,
            // the arrow’s local coordinate system rotates accordingly.
            // To maintain correct proportions, we apply the X or Z scaling of the cuboid
            // to the arrow's Y axis (its original forward direction).
            const xscale = 1.0 / x;
            const yscale = 1.0 / y;
            const zscale = 1.0 / z;
            this.orientationArrows[view].x.scale.set(yscale, xscale, zscale);
            this.orientationArrows[view].y.scale.set(xscale, yscale, zscale);
            this.orientationArrows[view].z.scale.set(xscale, zscale, yscale);
        });
    }

    public setRotation(x: number, y: number, z: number): void {
        [ViewType.PERSPECTIVE, ViewType.TOP, ViewType.SIDE, ViewType.FRONT].forEach((view): void => {
            (this as Indexable)[view].rotation.set(x, y, z);
        });
    }

    public attachCameraReference(): void {
        const topCameraReference = new THREE.Object3D();
        topCameraReference.translateZ(2);
        topCameraReference.name = constants.CAMERA_REFERENCE;
        this.top.add(topCameraReference);

        const sideCameraReference = new THREE.Object3D();
        sideCameraReference.translateY(2);
        sideCameraReference.name = constants.CAMERA_REFERENCE;
        this.side.add(sideCameraReference);

        const frontCameraReference = new THREE.Object3D();
        frontCameraReference.translateX(2);
        frontCameraReference.name = constants.CAMERA_REFERENCE;
        this.front.add(frontCameraReference);
    }

    public getReferenceCoordinates(viewType: string): THREE.Vector3 {
        const camRef = (this as Indexable)[viewType].getObjectByName(constants.CAMERA_REFERENCE);
        return camRef.getWorldPosition(new THREE.Vector3());
    }

    public setName(clientId: any): void {
        [ViewType.PERSPECTIVE, ViewType.TOP, ViewType.SIDE, ViewType.FRONT].forEach((view): void => {
            (this as Indexable)[view].name = clientId;
        });
    }

    public setColor(color: string): void {
        this.setOutlineColor(color);
        [ViewType.PERSPECTIVE, ViewType.TOP, ViewType.SIDE, ViewType.FRONT].forEach((view): void => {
            ((this as Indexable)[view].material as THREE.MeshBasicMaterial).color.set(color);
        });
    }

    public setOutlineColor(color: string): void {
        (this.wireframe.material as THREE.MeshBasicMaterial).color.set(color);
    }

    public setOpacity(opacity: number): void {
        [ViewType.PERSPECTIVE, ViewType.TOP, ViewType.SIDE, ViewType.FRONT].forEach((view): void => {
            ((this as Indexable)[view].material as THREE.MeshBasicMaterial).opacity = opacity / 100;
        });
    }

    public dispose(): void {
        disposeObject3D(this.perspective);
        disposeObject3D(this.top);
        disposeObject3D(this.side);
        disposeObject3D(this.front);

        this.perspective = null;
        this.top = null;
        this.side = null;
        this.front = null;
        this.wireframe = null;
        this.orientationArrows = {
            [ViewType.PERSPECTIVE]: null,
            [ViewType.TOP]: null,
            [ViewType.SIDE]: null,
            [ViewType.FRONT]: null,
        };
    }
}

const SHARED_WHITE_EDGE_MATERIAL = new THREE.LineBasicMaterial({ color: '#ffffff', linewidth: 3 });
SHARED_WHITE_EDGE_MATERIAL.userData.shared = true;

export function createCuboidEdges(instance: THREE.Mesh): THREE.LineSegments {
    const geometry = instance.geometry === SHARED_BOX_GEOMETRY ?
        SHARED_EDGES_GEOMETRY :
        new THREE.EdgesGeometry(instance.geometry);
    const edges = new THREE.LineSegments(geometry, SHARED_WHITE_EDGE_MATERIAL);
    edges.name = constants.CUBOID_EDGE_NAME;
    instance.add(edges);
    return edges;
}

export function removeCuboidEdges(instance: THREE.Mesh): void {
    const edges = instance.getObjectByName(constants.CUBOID_EDGE_NAME) as THREE.LineSegments;
    if (edges) {
        instance.remove(edges);
        disposeObjectResources(edges);
    }
}

const SHARED_RESIZE_HELPER_MATERIAL = new THREE.SpriteMaterial({
    color: '#ff0000',
    opacity: 1,
    map: getCircleTexture,
});
SHARED_RESIZE_HELPER_MATERIAL.userData.shared = true;

const SHARED_ROTATION_HELPER_MATERIAL = new THREE.SpriteMaterial({
    color: '#33b864',
    opacity: 1,
    map: getCircleTexture,
});
SHARED_ROTATION_HELPER_MATERIAL.userData.shared = true;

export function createResizeHelper(cuboid: CuboidModel, viewType: ViewType): void {
    const positions = cuboid.getResizeHelperPositions();
    for (let i = 0; i < positions.length; i++) {
        const position = positions[i];
        const helper = new THREE.Sprite(SHARED_RESIZE_HELPER_MATERIAL);
        helper.renderOrder = Number.MAX_SAFE_INTEGER;
        helper.name = `${constants.RESIZE_HELPER_NAME}_${i}`;
        helper.position.copy(position);
        cuboid[viewType].parent.add(helper);
    }
}

export function removeResizeHelper(instance: THREE.Mesh): void {
    instance.parent.children.filter((child: THREE.Object3D) => child.name.startsWith(constants.RESIZE_HELPER_NAME))
        .forEach((helper) => {
            instance.parent.remove(helper);
            const material = (helper as THREE.Sprite).material as THREE.SpriteMaterial;
            if (material && !material.userData?.shared) {
                material.dispose();
            }
        });
}

export function createRotationHelper(cuboid: CuboidModel, viewType: ViewType): void {
    if ([ViewType.TOP, ViewType.SIDE, ViewType.FRONT].includes(viewType)) {
        const helperPosition = cuboid.getRotationHelperPosition(viewType);
        const rotationHelper = new THREE.Sprite(SHARED_ROTATION_HELPER_MATERIAL);
        rotationHelper.renderOrder = Number.MAX_SAFE_INTEGER;
        rotationHelper.name = constants.ROTATION_HELPER_NAME;
        rotationHelper.position.copy(helperPosition);
        cuboid[viewType].parent.add(rotationHelper);
    }
}

export function removeRotationHelper(instance: THREE.Mesh): void {
    const helper = instance.parent.getObjectByName(constants.ROTATION_HELPER_NAME);
    if (helper) {
        instance.parent.remove(helper);
        const material = (helper as THREE.Sprite).material as THREE.SpriteMaterial;
        if (material && !material.userData?.shared) {
            material.dispose();
        }
    }
}
