/**
 * Worker 2: True OffscreenCanvas Three.js WebGL Rendering Worker (orthoViewsWorker.ts)
 * Dedicated Web Worker that owns the secondary THREE.WebGLRenderer, THREE.Scene,
 * ROI-culled THREE.Points geometry, 3D cuboid wireframe, and 3 THREE.OrthographicCamera
 * instances (Top BEV, Side Profile, Front Profile) on a transferred OffscreenCanvas.
 * All multi-view ROI culling AND WebGL rendering happen 100% off the main UI thread!
 */

import * as THREE from 'three';
import type { CuboidParams } from '../utils/snappingAndOrientation';

export interface OrthoFrustumBounds {
  topHalfExtent: number;
  sideHalfExtent: number;
  frontHalfExtent: number;
  roiRadius: number;
  inBoxPointCount: number;
  totalRoiPointCount: number;
}

export interface OrthoZoomLevels {
  top: number;
  side: number;
  front: number;
}

export interface OrthoInitOffscreenCanvasMessage {
  type: 'INIT_OFFSCREEN_CANVAS';
  canvas: OffscreenCanvas;
  width: number;
  height: number;
  pixelRatio: number;
  antialias?: boolean;
  orthoRoiMultiplier?: number;
  cuboid: CuboidParams;
  activeColor: string;
  isGroundStripped: boolean;
}

export interface OrthoSetPointCloudMessage {
  type: 'SET_POINT_CLOUD';
  positions: Float32Array;
  colors: Float32Array;
}

export interface OrthoUpdateCuboidMessage {
  type: 'UPDATE_CUBOID';
  requestId: number;
  cuboid: CuboidParams;
  activeColor: string;
  isGroundStripped: boolean;
}

export interface OrthoUpdateZoomMessage {
  type: 'UPDATE_ZOOM';
  zoomLevels: OrthoZoomLevels;
}

export interface OrthoResizeMessage {
  type: 'RESIZE';
  width: number;
  height: number;
  pixelRatio: number;
}

export interface OrthoUpdateViewsFallbackMessage {
  type: 'UPDATE_ORTHO_VIEWS';
  requestId: number;
  cuboid: CuboidParams;
  isGroundStripped: boolean;
}

export type OrthoWorkerRequest =
  | OrthoInitOffscreenCanvasMessage
  | OrthoSetPointCloudMessage
  | OrthoUpdateCuboidMessage
  | OrthoUpdateZoomMessage
  | OrthoResizeMessage
  | OrthoUpdateViewsFallbackMessage;

export interface OrthoRoiMetricsResponse {
  type: 'ROI_METRICS';
  requestId: number;
  bounds: OrthoFrustumBounds;
}

export interface OrthoViewsUpdatedFallbackResponse {
  type: 'ORTHO_VIEWS_UPDATED';
  requestId: number;
  roiPositions: Float32Array;
  roiColors: Float32Array;
  bounds: OrthoFrustumBounds;
}

export type OrthoWorkerResponse =
  | OrthoRoiMetricsResponse
  | OrthoViewsUpdatedFallbackResponse;

interface OrthoWorkerGlobalScopeCompat {
  postMessage: (message: OrthoWorkerResponse, transfer?: Transferable[]) => void;
  onmessage: ((event: MessageEvent<OrthoWorkerRequest>) => void) | null;
}

const workerScope = self as unknown as OrthoWorkerGlobalScopeCompat;

// Cached full LiDAR point cloud inside Worker 2
let cachedPositions: Float32Array | null = null;
let cachedColors: Float32Array | null = null;

// Worker-owned OffscreenCanvas Three.js WebGL state
let orthoRenderer: THREE.WebGLRenderer | null = null;
let orthoScene: THREE.Scene | null = null;
let orthoPointsMesh: THREE.Points | null = null;
let orthoBoxMesh: THREE.Mesh | null = null;
let orthoBoxEdges: THREE.LineSegments | null = null;
let orthoGroundRing: THREE.Line | null = null;
let orthoCameras: {
  top: THREE.OrthographicCamera;
  side: THREE.OrthographicCamera;
  front: THREE.OrthographicCamera;
} | null = null;

let canvasWidth = 540;
let canvasHeight = 164;
let currentOrthoRoiMultiplier = 1.6;
let currentCuboid: CuboidParams = {
  x: 10.2,
  y: 2.1,
  z: 0.5,
  l: 5.2,
  w: 2.6,
  h: 2.2,
  yaw: 0.0,
};
let currentActiveColor = '#3b82f6';
let currentIsGroundStripped = false;
let currentZooms: OrthoZoomLevels = { top: 1.0, side: 1.0, front: 1.0 };
let currentBounds: OrthoFrustumBounds = {
  topHalfExtent: 5.5,
  sideHalfExtent: 4.5,
  frontHalfExtent: 3.5,
  roiRadius: 11.5,
  inBoxPointCount: 0,
  totalRoiPointCount: 0,
};

// Track last ROI center/dimensions to avoid redundant full scans on tiny micro-rotations
let lastRoiCuboid: CuboidParams | null = null;
let lastRoiGroundStripped: boolean | null = null;
let cachedRoiRawPositions: Float32Array | null = null;
let cachedRoiRawColors: Float32Array | null = null;

/**
 * Extracts ROI-cropped point cloud subset and orthographic bounds for Top, Side, and Front views.
 */
export function computeOrthoRoiSubset(
  positions: Float32Array,
  colors: Float32Array,
  cuboid: CuboidParams,
  isGroundStripped: boolean,
  roiMultiplier: number = currentOrthoRoiMultiplier
): {
  roiPositions: Float32Array;
  roiColors: Float32Array;
  bounds: OrthoFrustumBounds;
} {
  const { x, y, z, l, w, h, yaw } = cuboid;
  const roiRadius = Math.max(l, w, h) * roiMultiplier + 3.5;

  // Check if we can reuse the spatial ROI crop buffer and only re-classify in-box points
  const needsFullSpatialRecrop =
    !lastRoiCuboid ||
    !cachedRoiRawPositions ||
    !cachedRoiRawColors ||
    lastRoiGroundStripped !== isGroundStripped ||
    Math.hypot(x - lastRoiCuboid.x, y - lastRoiCuboid.y, z - lastRoiCuboid.z) > 0.15 ||
    Math.abs(l - lastRoiCuboid.l) > 0.05 ||
    Math.abs(w - lastRoiCuboid.w) > 0.05 ||
    Math.abs(h - lastRoiCuboid.h) > 0.05;

  if (needsFullSpatialRecrop) {
    const cropMargin = 0.6;
    const cropRadiusSq = (roiRadius + cropMargin) * (roiRadius + cropMargin);
    const minZ = z - h / 2 - 1.5;
    const maxZ = z + h / 2 + 2.0;

    const numPoints = positions.length / 3;
    const rawPts: number[] = [];
    const rawCols: number[] = [];

    for (let i = 0; i < numPoints; i++) {
      const idx = i * 3;
      const px = positions[idx];
      const py = positions[idx + 1];
      const pz = positions[idx + 2];

      if (isGroundStripped && pz <= -1.55) continue;

      const dx = px - x;
      const dy = py - y;
      if (dx * dx + dy * dy > cropRadiusSq) continue;
      if (pz < minZ || pz > maxZ) continue;

      rawPts.push(px, py, pz);
      rawCols.push(colors[idx], colors[idx + 1], colors[idx + 2]);
    }

    cachedRoiRawPositions = new Float32Array(rawPts);
    cachedRoiRawColors = new Float32Array(rawCols);
    lastRoiCuboid = { ...cuboid };
    lastRoiGroundStripped = isGroundStripped;
  }

  const roiPositions = new Float32Array(cachedRoiRawPositions!);
  const rawColors = cachedRoiRawColors!;
  const roiColors = new Float32Array(rawColors.length);

  const cosY = Math.cos(yaw);
  const sinY = Math.sin(yaw);
  const halfL = l / 2;
  const halfW = w / 2;
  const halfH = h / 2;

  const roiCount = roiPositions.length / 3;
  let inBoxPointCount = 0;

  for (let i = 0; i < roiCount; i++) {
    const idx = i * 3;
    const px = roiPositions[idx];
    const py = roiPositions[idx + 1];
    const pz = roiPositions[idx + 2];

    const dx = px - x;
    const dy = py - y;
    const u = dx * cosY + dy * sinY;
    const v = -dx * sinY + dy * cosY;
    const dz = pz - z;

    const insideBox =
      Math.abs(u) <= halfL + 0.08 &&
      Math.abs(v) <= halfW + 0.08 &&
      Math.abs(dz) <= halfH + 0.08;

    if (insideBox) {
      inBoxPointCount++;
      roiColors[idx] = 0.22;
      roiColors[idx + 1] = 0.96;
      roiColors[idx + 2] = 0.88;
    } else if (pz <= -1.55) {
      roiColors[idx] = 0.28;
      roiColors[idx + 1] = 0.36;
      roiColors[idx + 2] = 0.52;
    } else {
      roiColors[idx] = rawColors[idx] * 0.65;
      roiColors[idx + 1] = rawColors[idx + 1] * 0.65;
      roiColors[idx + 2] = rawColors[idx + 2] * 0.65;
    }
  }

  const bounds: OrthoFrustumBounds = {
    topHalfExtent: Math.max(l, w) * 0.85 + 1.4,
    sideHalfExtent: Math.max(l, h) * 0.78 + 1.1,
    frontHalfExtent: Math.max(w, h) * 0.9 + 1.1,
    roiRadius,
    inBoxPointCount,
    totalRoiPointCount: roiCount,
  };

  return {
    roiPositions,
    roiColors,
    bounds,
  };
}

/**
 * Initializes the Three.js WebGLRenderer, Scene, Meshes, and 3 OrthographicCameras
 * directly on the worker's OffscreenCanvas. Safely disposes any previous renderer
 * if the component remounted in React StrictMode.
 */
function initOffscreenThreeRenderer(msg: OrthoInitOffscreenCanvasMessage): void {
  if (orthoRenderer) {
    try {
      orthoRenderer.dispose();
    } catch {
      // Ignore disposal errors from previous OffscreenCanvas
    }
    orthoRenderer = null;
  }

  canvasWidth = msg.width;
  canvasHeight = msg.height;
  currentOrthoRoiMultiplier = msg.orthoRoiMultiplier ?? 1.6;
  currentCuboid = { ...msg.cuboid };
  currentActiveColor = msg.activeColor;
  currentIsGroundStripped = msg.isGroundStripped;
  lastRoiCuboid = null;

  orthoScene = new THREE.Scene();
  orthoScene.background = new THREE.Color(0x090d16);

  orthoRenderer = new THREE.WebGLRenderer({
    canvas: msg.canvas as unknown as HTMLCanvasElement,
    antialias: msg.antialias ?? true,
    alpha: false,
    powerPreference: 'high-performance',
  });
  orthoRenderer.setPixelRatio(Math.min(msg.pixelRatio || 1, 2));
  // Pass updateStyle = false because OffscreenCanvas has no .style property
  orthoRenderer.setSize(canvasWidth, canvasHeight, false);
  orthoRenderer.setScissorTest(true);

  // Reference Grid at road level (z = -1.65)
  const orthoGrid = new THREE.GridHelper(36, 36, 0x1e293b, 0x111827);
  orthoGrid.rotation.x = Math.PI / 2;
  orthoGrid.position.z = -1.65;
  orthoScene.add(orthoGrid);

  // ROI-Cropped Point Cloud Mesh
  const orthoPointsGeo = new THREE.BufferGeometry();
  const orthoPointsMat = new THREE.PointsMaterial({
    size: 2.6,
    sizeAttenuation: false,
    vertexColors: true,
    transparent: true,
    opacity: 0.95,
  });
  orthoPointsMesh = new THREE.Points(orthoPointsGeo, orthoPointsMat);
  orthoScene.add(orthoPointsMesh);

  // 3D Cuboid Mesh & Wireframe Edges
  const orthoBoxGeo = new THREE.BoxGeometry(1, 1, 1);
  const orthoBoxMat = new THREE.MeshBasicMaterial({
    color: new THREE.Color(currentActiveColor),
    transparent: true,
    opacity: 0.18,
    depthWrite: false,
  });
  orthoBoxMesh = new THREE.Mesh(orthoBoxGeo, orthoBoxMat);
  orthoScene.add(orthoBoxMesh);

  const orthoEdgesGeo = new THREE.EdgesGeometry(orthoBoxGeo);
  const orthoEdgesMat = new THREE.LineBasicMaterial({
    color: new THREE.Color(currentActiveColor),
    linewidth: 2,
  });
  orthoBoxEdges = new THREE.LineSegments(orthoEdgesGeo, orthoEdgesMat);
  orthoBoxMesh.add(orthoBoxEdges);

  const orthoArrow = new THREE.ArrowHelper(
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(0, 0, 0),
    0.85,
    0x00ffcc,
    0.25,
    0.18
  );
  orthoBoxMesh.add(orthoArrow);

  // Ground Contact Ring
  const ringGeo = new THREE.BufferGeometry();
  const ringPts: THREE.Vector3[] = [];
  for (let i = 0; i <= 32; i++) {
    const a = (i / 32) * Math.PI * 2;
    ringPts.push(new THREE.Vector3(Math.cos(a), Math.sin(a), 0));
  }
  ringGeo.setFromPoints(ringPts);
  orthoGroundRing = new THREE.Line(
    ringGeo,
    new THREE.LineBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.55 })
  );
  orthoScene.add(orthoGroundRing);

  orthoCameras = {
    top: new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 100),
    side: new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 100),
    front: new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 100),
  };

  if (cachedPositions && cachedColors) {
    recomputeRoiAndRender(0);
  } else {
    syncWorkerOrthoSceneAndRender();
  }
}

/**
 * Updates cuboid mesh, orthographic camera frustums, and renders all 3 viewports
 * (Top BEV, Side Profile, Front Profile) on the worker's OffscreenCanvas.
 */
function syncWorkerOrthoSceneAndRender(): void {
  if (!orthoRenderer || !orthoScene || !orthoCameras) return;

  const c = currentCuboid;
  if (orthoBoxMesh && orthoBoxEdges && orthoGroundRing) {
    orthoBoxMesh.position.set(c.x, c.y, c.z);
    orthoBoxMesh.scale.set(c.l, c.w, c.h);
    orthoBoxMesh.rotation.z = c.yaw;
    (orthoBoxMesh.material as THREE.MeshBasicMaterial).color.set(currentActiveColor);
    (orthoBoxEdges.material as THREE.LineBasicMaterial).color.set(currentActiveColor);

    orthoGroundRing.position.set(c.x, c.y, -1.63);
    orthoGroundRing.scale.set(c.l / 1.6, c.w / 1.6, 1);
  }

  const { top, side, front } = orthoCameras;
  const subW = Math.floor(canvasWidth / 3);
  const subH = canvasHeight;
  const aspect = subW / Math.max(1, subH);

  const cosY = Math.cos(c.yaw);
  const sinY = Math.sin(c.yaw);

  // 1. Top (BEV) Orthographic Camera
  const topH = currentBounds.topHalfExtent / currentZooms.top;
  const topW = topH * aspect;
  top.left = -topW;
  top.right = topW;
  top.top = topH;
  top.bottom = -topH;
  top.position.set(c.x, c.y, c.z + 18);
  top.up.set(cosY, sinY, 0);
  top.lookAt(c.x, c.y, c.z);
  top.updateProjectionMatrix();

  // 2. Side Profile Orthographic Camera
  const sideH = currentBounds.sideHalfExtent / currentZooms.side;
  const sideW = sideH * aspect;
  side.left = -sideW;
  side.right = sideW;
  side.top = sideH;
  side.bottom = -sideH;
  side.position.set(c.x - sinY * 18, c.y + cosY * 18, c.z);
  side.up.set(0, 0, 1);
  side.lookAt(c.x, c.y, c.z);
  side.updateProjectionMatrix();

  // 3. Front Profile Orthographic Camera
  const frontH = currentBounds.frontHalfExtent / currentZooms.front;
  const frontW = frontH * aspect;
  front.left = -frontW;
  front.right = frontW;
  front.top = frontH;
  front.bottom = -frontH;
  front.position.set(c.x + cosY * 18, c.y + sinY * 18, c.z);
  front.up.set(0, 0, 1);
  front.lookAt(c.x, c.y, c.z);
  front.updateProjectionMatrix();

  // Render all 3 viewports on the worker's OffscreenCanvas
  orthoRenderer.setViewport(0, 0, subW, subH);
  orthoRenderer.setScissor(0, 0, subW, subH);
  orthoRenderer.render(orthoScene, top);

  orthoRenderer.setViewport(subW, 0, subW, subH);
  orthoRenderer.setScissor(subW, 0, subW, subH);
  orthoRenderer.render(orthoScene, side);

  orthoRenderer.setViewport(subW * 2, 0, subW, subH);
  orthoRenderer.setScissor(subW * 2, 0, subW, subH);
  orthoRenderer.render(orthoScene, front);
}

/**
 * Recomputes ROI point cloud subset inside Worker 2, updates the worker's THREE.Points
 * geometry, renders to the OffscreenCanvas, and sends lightweight ROI_METRICS to main thread.
 */
function recomputeRoiAndRender(requestId: number): void {
  if (!cachedPositions || !cachedColors) return;

  const { roiPositions, roiColors, bounds } = computeOrthoRoiSubset(
    cachedPositions,
    cachedColors,
    currentCuboid,
    currentIsGroundStripped
  );
  currentBounds = bounds;

  if (orthoPointsMesh) {
    const geom = orthoPointsMesh.geometry;
    geom.setAttribute('position', new THREE.BufferAttribute(roiPositions, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(roiColors, 3));
    geom.computeBoundingSphere();
  }

  syncWorkerOrthoSceneAndRender();

  workerScope.postMessage({
    type: 'ROI_METRICS',
    requestId,
    bounds,
  });
}

workerScope.onmessage = (event: MessageEvent<OrthoWorkerRequest>) => {
  const msg = event.data;

  if (msg.type === 'INIT_OFFSCREEN_CANVAS') {
    initOffscreenThreeRenderer(msg);
  } else if (msg.type === 'SET_POINT_CLOUD') {
    cachedPositions = new Float32Array(msg.positions);
    cachedColors = new Float32Array(msg.colors);
    lastRoiCuboid = null;
    if (orthoRenderer) {
      recomputeRoiAndRender(0);
    }
  } else if (msg.type === 'UPDATE_CUBOID') {
    currentCuboid = { ...msg.cuboid };
    currentActiveColor = msg.activeColor;
    currentIsGroundStripped = msg.isGroundStripped;

    if (orthoRenderer) {
      recomputeRoiAndRender(msg.requestId);
    }
  } else if (msg.type === 'UPDATE_ZOOM') {
    currentZooms = { ...msg.zoomLevels };
    syncWorkerOrthoSceneAndRender();
  } else if (msg.type === 'RESIZE') {
    canvasWidth = msg.width;
    canvasHeight = msg.height;
    if (orthoRenderer) {
      orthoRenderer.setPixelRatio(Math.min(msg.pixelRatio || 1, 2));
      orthoRenderer.setSize(canvasWidth, canvasHeight, false);
      syncWorkerOrthoSceneAndRender();
    }
  } else if (msg.type === 'UPDATE_ORTHO_VIEWS') {
    // Fallback mode when OffscreenCanvas is unavailable
    if (!cachedPositions || !cachedColors) return;
    currentCuboid = { ...msg.cuboid };
    currentIsGroundStripped = msg.isGroundStripped;

    const { roiPositions, roiColors, bounds } = computeOrthoRoiSubset(
      cachedPositions,
      cachedColors,
      msg.cuboid,
      msg.isGroundStripped
    );

    workerScope.postMessage(
      {
        type: 'ORTHO_VIEWS_UPDATED',
        requestId: msg.requestId,
        roiPositions,
        roiColors,
        bounds,
      },
      [roiPositions.buffer, roiColors.buffer]
    );
  }
};
