import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import type { CuboidParams, SnapMetrics } from '../utils/snappingAndOrientation';
import { snapAndOrientCuboid } from '../utils/snappingAndOrientation';
import { DualSceneWorkerController, type OrthoZoomLevels } from '../utils/sceneWorkers';
import { Tooltip, Button, Tag } from 'antd';
import {
  AimOutlined,
  CompressOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  UndoOutlined,
  SwapOutlined,
  FilterOutlined,
  ScissorOutlined,
} from '@ant-design/icons';

export type CameraViewMode = 'perspective' | 'top' | 'side' | 'front';

interface PointCloudCanvasProps {
  cuboid: CuboidParams;
  onCuboidChange: (newCuboid: CuboidParams) => void;
  activeColor: string;
  isGroundStripped?: boolean;
  isObjectIsolated?: boolean;
  onToggleGroundStripping?: () => void;
  onToggleObjectIsolation?: () => void;
  onInvertHeading?: () => void;
  onSnapNotification?: (metrics: SnapMetrics) => void;
  currentViewMode?: CameraViewMode;
  onViewModeChange?: (mode: CameraViewMode) => void;
}

const ORTHO_DOCK_WIDTH = 540;
const ORTHO_DOCK_HEIGHT = 164;

export const PointCloudCanvas: React.FC<PointCloudCanvasProps> = ({
  cuboid,
  onCuboidChange,
  activeColor,
  isGroundStripped = false,
  isObjectIsolated = false,
  onToggleGroundStripping,
  onToggleObjectIsolation,
  onInvertHeading,
  onSnapNotification,
  currentViewMode,
  onViewModeChange,
}) => {
  // Renderer 1 (Main Thread): Main 3D Perspective Scene Refs
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const pointCloudDataRef = useRef<Float32Array | null>(null);
  const originalColorsRef = useRef<Float32Array | null>(null);
  const pointsMeshRef = useRef<THREE.Points | null>(null);
  const boxMeshRef = useRef<THREE.Mesh | null>(null);
  const boxEdgesRef = useRef<THREE.LineSegments | null>(null);
  const arrowHelperRef = useRef<THREE.ArrowHelper | null>(null);
  const groundRingRef = useRef<THREE.Line | null>(null);
  const needsRenderRef = useRef<boolean>(true);

  // Renderer 2 (Worker 2 OffscreenCanvas): Multi-View Orthographic Dock Container & Zoom State
  const orthoMountRef = useRef<HTMLDivElement>(null);
  const orthoZoomsRef = useRef<OrthoZoomLevels>({
    top: 1.0,
    side: 1.0,
    front: 1.0,
  });

  // Dual Web Worker Controller (Worker 1: MainSceneWorker, Worker 2: OffscreenCanvas OrthoViewsWorker)
  const workerControllerRef = useRef<DualSceneWorkerController | null>(null);

  // Latest state refs for non-reactive callbacks inside worker listeners & animation loop
  const cuboidRef = useRef<CuboidParams>(cuboid);
  cuboidRef.current = cuboid;
  const activeColorRef = useRef<string>(activeColor);
  activeColorRef.current = activeColor;
  const isGroundStrippedRef = useRef<boolean>(isGroundStripped);
  isGroundStrippedRef.current = isGroundStripped;
  const onCuboidChangeRef = useRef(onCuboidChange);
  onCuboidChangeRef.current = onCuboidChange;
  const onSnapNotificationRef = useRef(onSnapNotification);
  onSnapNotificationRef.current = onSnapNotification;

  const [lastSnap, setLastSnap] = useState<SnapMetrics | null>(null);
  const [previousCuboid, setPreviousCuboid] = useState<CuboidParams | null>(null);
  const [previousCuboidArray, setPreviousCuboidArray] = useState<number[] | null>(null);
  const [internalViewMode, setInternalViewMode] = useState<CameraViewMode>('perspective');
  const [orthoStats, setOrthoStats] = useState<{ inBox: number; roiTotal: number }>({
    inBox: 0,
    roiTotal: 0,
  });
  const viewMode = currentViewMode ?? internalViewMode;
  const hasActiveFilterRef = useRef<boolean>(false);

  // Camera Orbit State (Target + Smooth Damped Current State)
  const isDraggingRef = useRef(false);
  const dragButtonRef = useRef(0);
  const prevMouseRef = useRef({ x: 0, y: 0 });
  const targetOrbitRef = useRef({
    theta: Math.PI / 4,
    phi: Math.PI / 3.5,
    radius: 28,
    target: new THREE.Vector3(8, 0, -1.0),
  });
  const currentOrbitRef = useRef({
    theta: Math.PI / 4,
    phi: Math.PI / 3.5,
    radius: 28,
    target: new THREE.Vector3(8, 0, -1.0),
  });

  // 1. Initialize Main Three.js WebGLRenderer (Renderer 1) and Dual Web Worker Pipeline
  //    (Worker 1: MainSceneWorker + Worker 2: OffscreenCanvas OrthoViewsWorker).
  //    Note: activeColor is intentionally excluded from dependencies so changing class never recreates WebGL!
  useEffect(() => {
    if (!mountRef.current) return;
    const container = mountRef.current;
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;

    // --- RENDERER 1 (Main Thread): Main 3D Perspective Scene ---
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0c0f17);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 500);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Ambient & Directional Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(20, 20, 30);
    scene.add(dirLight);

    // Coordinate Grid at ground (z = -1.65)
    const grid = new THREE.GridHelper(80, 80, 0x223355, 0x141b2a);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = -1.65;
    scene.add(grid);

    // Main Point Cloud Mesh (populated asynchronously by Worker 1: MainSceneWorker)
    const geometry = new THREE.BufferGeometry();
    const material = new THREE.PointsMaterial({
      size: 0.12,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
    });
    const pointsMesh = new THREE.Points(geometry, material);
    scene.add(pointsMesh);
    pointsMeshRef.current = pointsMesh;

    // 3D Bounding Box (Cuboid)
    const boxGeo = new THREE.BoxGeometry(1, 1, 1);
    const boxMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(activeColorRef.current),
      transparent: true,
      opacity: 0.28,
      depthWrite: false,
    });
    const boxMesh = new THREE.Mesh(boxGeo, boxMat);
    scene.add(boxMesh);
    boxMeshRef.current = boxMesh;

    // Wireframe Edges
    const edgesGeo = new THREE.EdgesGeometry(boxGeo);
    const edgesMat = new THREE.LineBasicMaterial({
      color: new THREE.Color(activeColorRef.current),
      linewidth: 2,
    });
    const boxEdges = new THREE.LineSegments(edgesGeo, edgesMat);
    boxMesh.add(boxEdges);
    boxEdgesRef.current = boxEdges;

    // Front Orientation Arrow Helper
    const arrowDir = new THREE.Vector3(1, 0, 0);
    const arrowHelper = new THREE.ArrowHelper(
      arrowDir,
      new THREE.Vector3(0, 0, 0),
      2.8,
      0x00ffcc,
      0.6,
      0.35
    );
    boxMesh.add(arrowHelper);
    arrowHelperRef.current = arrowHelper;

    // Ground Contact Ring / Shadow
    const ringGeo = new THREE.BufferGeometry();
    const ringPts: THREE.Vector3[] = [];
    for (let i = 0; i <= 32; i++) {
      const a = (i / 32) * Math.PI * 2;
      ringPts.push(new THREE.Vector3(Math.cos(a), Math.sin(a), 0));
    }
    ringGeo.setFromPoints(ringPts);
    const ringMat = new THREE.LineBasicMaterial({
      color: 0x00ff88,
      transparent: true,
      opacity: 0.6,
    });
    const groundRing = new THREE.Line(ringGeo, ringMat);
    scene.add(groundRing);
    groundRingRef.current = groundRing;

    // --- Initialize Dual-Worker Pipeline (Worker 1 + Worker 2 OffscreenCanvas) ---
    const workerController = new DualSceneWorkerController({
      onMainSceneInitialized: (positions, colors) => {
        pointCloudDataRef.current = positions;
        originalColorsRef.current = new Float32Array(colors);

        if (pointsMeshRef.current) {
          pointsMeshRef.current.geometry.setAttribute(
            'position',
            new THREE.BufferAttribute(positions, 3)
          );
          pointsMeshRef.current.geometry.setAttribute(
            'color',
            new THREE.BufferAttribute(new Float32Array(colors), 3)
          );
          pointsMeshRef.current.geometry.computeBoundingSphere();
          needsRenderRef.current = true;
        }

        // Trigger initial ROI update & render on Worker 2 (OrthoViewsWorker)
        workerController.requestOrthoCuboidUpdate(
          cuboidRef.current,
          activeColorRef.current,
          isGroundStrippedRef.current
        );
      },
      onMainPointCloudFiltered: (filteredColors) => {
        if (!pointsMeshRef.current) return;
        const geom = pointsMeshRef.current.geometry;
        geom.setAttribute('color', new THREE.BufferAttribute(filteredColors, 3));
        needsRenderRef.current = true;
      },
      onMainSnapResult: (result) => {
        setPreviousCuboid(result.previousCuboid);
        setPreviousCuboidArray(result.previousCuboidArray);
        onCuboidChangeRef.current(result.cuboid);
        setLastSnap(result.metrics);
        if (onSnapNotificationRef.current) {
          onSnapNotificationRef.current(result.metrics);
        }
      },
      onOrthoRoiMetrics: (bounds) => {
        setOrthoStats({
          inBox: bounds.inBoxPointCount,
          roiTotal: bounds.totalRoiPointCount,
        });
      },
    });
    workerControllerRef.current = workerController;

    // --- RENDERER 2 (Worker 2 Thread): Transfer OffscreenCanvas to OrthoViewsWorker ---
    const orthoContainer = orthoMountRef.current;
    let orthoCanvasEl: HTMLCanvasElement | null = null;
    if (orthoContainer) {
      // Create a fresh <canvas> dynamically so React StrictMode double-mount never fails on transferControlToOffscreen()
      orthoCanvasEl = document.createElement('canvas');
      orthoCanvasEl.width = ORTHO_DOCK_WIDTH;
      orthoCanvasEl.height = ORTHO_DOCK_HEIGHT;
      orthoCanvasEl.style.width = `${ORTHO_DOCK_WIDTH}px`;
      orthoCanvasEl.style.height = `${ORTHO_DOCK_HEIGHT}px`;
      orthoCanvasEl.style.display = 'block';
      orthoContainer.appendChild(orthoCanvasEl);

      if (typeof orthoCanvasEl.transferControlToOffscreen === 'function') {
        const offscreen = orthoCanvasEl.transferControlToOffscreen();
        workerController.initOffscreenCanvas(
          offscreen,
          ORTHO_DOCK_WIDTH,
          ORTHO_DOCK_HEIGHT,
          Math.min(window.devicePixelRatio || 1, 2),
          cuboidRef.current,
          activeColorRef.current,
          isGroundStrippedRef.current
        );
      }
    }

    workerController.initScene();

    // --- Demand-Driven + Smoothly Damped Animation Loop for Main 3D Scene (Renderer 1) ---
    let animId: number;
    const damping = 0.25;

    const animate = () => {
      animId = requestAnimationFrame(animate);

      // Smoothly damp camera orbit towards targetOrbitRef
      const tgt = targetOrbitRef.current;
      const cur = currentOrbitRef.current;

      const dTheta = tgt.theta - cur.theta;
      const dPhi = tgt.phi - cur.phi;
      const dRadius = tgt.radius - cur.radius;
      const dTx = tgt.target.x - cur.target.x;
      const dTy = tgt.target.y - cur.target.y;
      const dTz = tgt.target.z - cur.target.z;

      const isMoving =
        Math.abs(dTheta) > 1e-4 ||
        Math.abs(dPhi) > 1e-4 ||
        Math.abs(dRadius) > 1e-3 ||
        Math.abs(dTx) > 1e-4 ||
        Math.abs(dTy) > 1e-4 ||
        Math.abs(dTz) > 1e-4;

      if (isMoving) {
        cur.theta += dTheta * damping;
        cur.phi += dPhi * damping;
        cur.radius += dRadius * damping;
        cur.target.x += dTx * damping;
        cur.target.y += dTy * damping;
        cur.target.z += dTz * damping;
        needsRenderRef.current = true;
      }

      // Render Main 3D Scene (Renderer 1) only when dirty
      if (needsRenderRef.current) {
        camera.position.x = cur.target.x + cur.radius * Math.sin(cur.phi) * Math.cos(cur.theta);
        camera.position.y = cur.target.y + cur.radius * Math.sin(cur.phi) * Math.sin(cur.theta);
        camera.position.z = cur.target.z + cur.radius * Math.cos(cur.phi);
        camera.lookAt(cur.target);

        renderer.render(scene, camera);
        needsRenderRef.current = false;
      }
    };
    animate();

    // Window Resize Handler
    const handleResize = () => {
      if (!container || !camera || !renderer) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      needsRenderRef.current = true;
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);
      workerController.dispose();
      workerControllerRef.current = null;
      if (renderer.domElement && container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      renderer.dispose();
      if (orthoCanvasEl && orthoContainer && orthoContainer.contains(orthoCanvasEl)) {
        orthoContainer.removeChild(orthoCanvasEl);
      }
    };
  }, []);

  // 2. Update Cuboid Mesh in Main 3D Scene (Renderer 1) and Offload Ortho Scene Render to Worker 2
  useEffect(() => {
    if (boxMeshRef.current && boxEdgesRef.current && groundRingRef.current) {
      const mesh = boxMeshRef.current;
      mesh.position.set(cuboid.x, cuboid.y, cuboid.z);
      mesh.scale.set(cuboid.l, cuboid.w, cuboid.h);
      mesh.rotation.z = cuboid.yaw;

      (mesh.material as THREE.MeshBasicMaterial).color.set(activeColor);
      (boxEdgesRef.current.material as THREE.LineBasicMaterial).color.set(activeColor);

      const groundZ = -1.65;
      groundRingRef.current.position.set(cuboid.x, cuboid.y, groundZ + 0.02);
      groundRingRef.current.scale.set(cuboid.l / 1.8, cuboid.w / 1.8, 1);
      needsRenderRef.current = true;
    }

    // Offload ROI point cloud crop & OffscreenCanvas Three.js rendering to Worker 2 (OrthoViewsWorker)
    workerControllerRef.current?.requestOrthoCuboidUpdate(
      cuboid,
      activeColor,
      isGroundStripped
    );
  }, [cuboid, activeColor, isGroundStripped]);

  // 3. Mouse Event Listeners for Smooth Main 3D Camera Navigation
  const handleMouseDown = (e: React.MouseEvent) => {
    isDraggingRef.current = true;
    dragButtonRef.current = e.button;
    prevMouseRef.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDraggingRef.current) return;
    const dx = e.clientX - prevMouseRef.current.x;
    const dy = e.clientY - prevMouseRef.current.y;
    prevMouseRef.current = { x: e.clientX, y: e.clientY };

    if (dragButtonRef.current === 0) {
      // Left button: Orbit rotation
      targetOrbitRef.current.theta -= dx * 0.008;
      targetOrbitRef.current.phi = Math.max(
        0.04,
        Math.min(Math.PI / 2 - 0.03, targetOrbitRef.current.phi - dy * 0.008)
      );
    } else if (dragButtonRef.current === 2 || dragButtonRef.current === 1) {
      // Right/Middle button: Pan
      const panSpeed = targetOrbitRef.current.radius * 0.0012;
      const cosT = Math.cos(targetOrbitRef.current.theta);
      const sinT = Math.sin(targetOrbitRef.current.theta);
      targetOrbitRef.current.target.x += (-dx * sinT - dy * cosT) * panSpeed;
      targetOrbitRef.current.target.y += (dx * cosT - dy * sinT) * panSpeed;
    }
    needsRenderRef.current = true;
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
  };

  const handleWheel = (e: React.WheelEvent) => {
    targetOrbitRef.current.radius = Math.max(
      4,
      Math.min(120, targetOrbitRef.current.radius + e.deltaY * 0.03)
    );
    needsRenderRef.current = true;
  };

  const handleOrthoSubViewWheel = (
    e: React.WheelEvent,
    subView: 'top' | 'side' | 'front'
  ) => {
    e.stopPropagation();
    const factor = e.deltaY < 0 ? 1.12 : 0.9;
    orthoZoomsRef.current = {
      ...orthoZoomsRef.current,
      [subView]: Math.max(0.4, Math.min(4.0, orthoZoomsRef.current[subView] * factor)),
    };
    workerControllerRef.current?.requestOrthoZoomUpdate(orthoZoomsRef.current);
  };

  // 4. Snapping & Orientation Action (Offloaded to Worker 1 with synchronous fallback)
  const executeSnapToRoadAndOrient = useCallback(
    (options?: { snapGround?: boolean; autoOrient?: boolean }) => {
      if (workerControllerRef.current) {
        workerControllerRef.current.requestSnapAndOrient(cuboid, {
          snapGround: options?.snapGround ?? true,
          autoOrient: options?.autoOrient ?? true,
        });
        return;
      }

      if (!pointCloudDataRef.current) return;
      const result = snapAndOrientCuboid(cuboid, pointCloudDataRef.current, {
        snapGround: options?.snapGround ?? true,
        autoOrient: options?.autoOrient ?? true,
      });

      setPreviousCuboid(result.previousCuboid);
      setPreviousCuboidArray(result.previousCuboidArray);
      onCuboidChange(result.cuboid);
      setLastSnap(result.metrics);

      if (onSnapNotification) {
        onSnapNotification(result.metrics);
      }
    },
    [cuboid, onCuboidChange, onSnapNotification]
  );

  const revertSnap = useCallback(() => {
    if (previousCuboidArray && previousCuboidArray.length >= 9) {
      onCuboidChange({
        x: previousCuboidArray[0],
        y: previousCuboidArray[1],
        z: previousCuboidArray[2],
        yaw: previousCuboidArray[5],
        l: previousCuboidArray[6],
        w: previousCuboidArray[7],
        h: previousCuboidArray[8],
      });
      setPreviousCuboid(null);
      setPreviousCuboidArray(null);
      setLastSnap(null);
    } else if (previousCuboid) {
      onCuboidChange(previousCuboid);
      setPreviousCuboid(null);
      setPreviousCuboidArray(null);
      setLastSnap(null);
    }
  }, [previousCuboid, previousCuboidArray, onCuboidChange]);

  // 5. Offload Dynamic Point Cloud Filtering (Ground Stripping & Object Isolation) to Worker 1
  useEffect(() => {
    if (!workerControllerRef.current) return;

    if (!isGroundStripped && !isObjectIsolated) {
      if (!hasActiveFilterRef.current) return;
      hasActiveFilterRef.current = false;
    } else {
      hasActiveFilterRef.current = true;
    }

    workerControllerRef.current.requestFilterPointCloud(
      cuboid,
      isGroundStripped,
      isObjectIsolated
    );
  }, [isGroundStripped, isObjectIsolated, cuboid]);

  // View Presets: Top, Side, Front, and Perspective
  const setPresetView = useCallback(
    (mode: CameraViewMode) => {
      if (onViewModeChange) {
        onViewModeChange(mode);
      } else {
        setInternalViewMode(mode);
      }

      const { x, y, z, yaw, l, w, h } = cuboid;
      const targetZ = z || -0.9;

      if (mode === 'top') {
        targetOrbitRef.current.phi = 0.04;
        targetOrbitRef.current.theta = yaw - Math.PI / 2;
        targetOrbitRef.current.radius = Math.max(l, w) * 2.8 + 6;
        targetOrbitRef.current.target.set(x, y, targetZ);
      } else if (mode === 'side') {
        targetOrbitRef.current.phi = Math.PI / 2 - 0.05;
        targetOrbitRef.current.theta = yaw;
        targetOrbitRef.current.radius = Math.max(l, h) * 2.2 + 4;
        targetOrbitRef.current.target.set(x, y, targetZ);
      } else if (mode === 'front') {
        targetOrbitRef.current.phi = Math.PI / 2 - 0.05;
        targetOrbitRef.current.theta = yaw + Math.PI / 2;
        targetOrbitRef.current.radius = Math.max(w, h) * 2.5 + 4;
        targetOrbitRef.current.target.set(x, y, targetZ);
      } else {
        targetOrbitRef.current.phi = Math.PI / 3.5;
        targetOrbitRef.current.theta = Math.PI / 4;
        targetOrbitRef.current.radius = 28;
        targetOrbitRef.current.target.set(x, y, targetZ);
      }
      needsRenderRef.current = true;
    },
    [cuboid, onViewModeChange]
  );

  // Keyboard shortcut listener: micro-nudges, snap, revert, heading inversion, modes
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) return;

      const stepDist = e.shiftKey ? 0.2 : 0.04;
      const stepAngle = (e.shiftKey ? 5 : 1) * (Math.PI / 180);

      if (e.key === 'ArrowUp') {
        e.preventDefault();
        onCuboidChange({ ...cuboid, x: parseFloat((cuboid.x + stepDist).toFixed(3)) });
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        onCuboidChange({ ...cuboid, x: parseFloat((cuboid.x - stepDist).toFixed(3)) });
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        onCuboidChange({ ...cuboid, y: parseFloat((cuboid.y + stepDist).toFixed(3)) });
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        onCuboidChange({ ...cuboid, y: parseFloat((cuboid.y - stepDist).toFixed(3)) });
      } else if (e.key === 'PageUp') {
        e.preventDefault();
        onCuboidChange({ ...cuboid, z: parseFloat((cuboid.z + 0.02).toFixed(3)) });
      } else if (e.key === 'PageDown') {
        e.preventDefault();
        onCuboidChange({ ...cuboid, z: parseFloat((cuboid.z - 0.02).toFixed(3)) });
      } else if (e.key === '[' || e.key === '{') {
        e.preventDefault();
        onCuboidChange({ ...cuboid, yaw: parseFloat((cuboid.yaw - stepAngle).toFixed(4)) });
      } else if (e.key === ']' || e.key === '}') {
        e.preventDefault();
        onCuboidChange({ ...cuboid, yaw: parseFloat((cuboid.yaw + stepAngle).toFixed(4)) });
      } else if (e.key === 'h' || e.key === 'H') {
        e.preventDefault();
        onInvertHeading?.();
      } else if (e.key === 'g' || e.key === 'G') {
        e.preventDefault();
        onToggleGroundStripping?.();
      } else if ((e.key === 'i' || e.key === 'I') && e.shiftKey) {
        e.preventDefault();
        onToggleObjectIsolation?.();
      } else if (e.key === '1') {
        e.preventDefault();
        setPresetView('perspective');
      } else if (e.key === '2') {
        e.preventDefault();
        setPresetView('top');
      } else if (e.key === '3') {
        e.preventDefault();
        setPresetView('side');
      } else if (e.key === '4') {
        e.preventDefault();
        setPresetView('front');
      } else if ((e.key === 'f' || e.key === 'F') && e.shiftKey) {
        e.preventDefault();
        revertSnap();
      } else if (e.key === 'f' || e.key === 'F') {
        e.preventDefault();
        executeSnapToRoadAndOrient();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [
    executeSnapToRoadAndOrient,
    revertSnap,
    cuboid,
    onCuboidChange,
    onInvertHeading,
    onToggleGroundStripping,
    onToggleObjectIsolation,
    setPresetView,
  ]);

  useEffect(() => {
    if (currentViewMode) {
      setPresetView(currentViewMode);
    }
  }, [currentViewMode, setPresetView]);

  const orthoSubViews: Array<{
    key: 'top' | 'side' | 'front';
    title: string;
    shortcut: string;
    subtitle: string;
  }> = [
    { key: 'top', title: 'TOP (BEV)', shortcut: '2', subtitle: 'Yaw & L×W Fit' },
    { key: 'side', title: 'SIDE PROFILE', shortcut: '3', subtitle: 'Ground Z & Pitch' },
    { key: 'front', title: 'FRONT PROFILE', shortcut: '4', subtitle: 'Roll & Width' },
  ];

  return (
    <div
      className="point-cloud-canvas-wrapper"
      style={{ position: 'relative', width: '100%', height: '100%', overflow: 'hidden' }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onWheel={handleWheel}
      onContextMenu={(e) => e.preventDefault()}
    >
      {/* Main 3D Perspective WebGL Mount (Renderer 1) */}
      <div ref={mountRef} style={{ width: '100%', height: '100%' }} />

      {/* Top Floating Control Bar */}
      <div
        style={{
          position: 'absolute',
          top: 16,
          left: 16,
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          background: 'rgba(15, 23, 42, 0.85)',
          padding: '8px 14px',
          borderRadius: '10px',
          border: '1px solid rgba(255, 255, 255, 0.12)',
          backdropFilter: 'blur(8px)',
          boxShadow: '0 8px 24px rgba(0, 0, 0, 0.4)',
        }}
      >
        <Tooltip title="1-Click Snap to Road & Align Orientation (Shortcut: F)">
          <Button
            type="primary"
            icon={<ThunderboltOutlined />}
            onClick={() => executeSnapToRoadAndOrient()}
            style={{
              background: 'linear-gradient(135deg, #3b82f6 0%, #10b981 100%)',
              borderColor: 'transparent',
              fontWeight: 600,
            }}
          >
            Auto-Snap & Orient (F)
          </Button>
        </Tooltip>

        <Tooltip title="Snap Base to Ground Plane Only">
          <Button
            icon={<AimOutlined />}
            onClick={() => executeSnapToRoadAndOrient({ snapGround: true, autoOrient: false })}
          >
            Snap Ground (Z)
          </Button>
        </Tooltip>

        <Tooltip title="Auto-Orient Yaw & Tighten Bounding Box Only">
          <Button
            icon={<CompressOutlined />}
            onClick={() => executeSnapToRoadAndOrient({ snapGround: false, autoOrient: true })}
          >
            Orient Heading
          </Button>
        </Tooltip>

        {previousCuboid && (
          <Tooltip title="Revert to Pre-Snap Position & Extents (Shift+F)">
            <Button
              icon={<UndoOutlined />}
              onClick={revertSnap}
              style={{
                borderColor: '#f59e0b',
                color: '#f59e0b',
                fontWeight: 600,
              }}
            >
              Revert Snap (Shift+F)
            </Button>
          </Tooltip>
        )}

        <Tooltip title="Invert Vehicle Heading (180°) (Shortcut: H)">
          <Button icon={<SwapOutlined />} onClick={onInvertHeading}>
            Invert (180°)
          </Button>
        </Tooltip>

        <Tooltip title="Toggle Ground Stripping: Filter Road Surface (Shortcut: G)">
          <Button
            type={isGroundStripped ? 'primary' : 'default'}
            icon={<FilterOutlined />}
            onClick={onToggleGroundStripping}
            style={isGroundStripped ? { background: '#8b5cf6', borderColor: '#8b5cf6' } : undefined}
          >
            Strip Road (G)
          </Button>
        </Tooltip>

        <Tooltip title="Toggle Object Isolation: Dim Scene Clutter (Shortcut: Shift+I)">
          <Button
            type={isObjectIsolated ? 'primary' : 'default'}
            icon={<ScissorOutlined />}
            onClick={onToggleObjectIsolation}
            style={isObjectIsolated ? { background: '#06b6d4', borderColor: '#06b6d4' } : undefined}
          >
            Isolate (Shift+I)
          </Button>
        </Tooltip>

        <div style={{ width: 1, height: 20, background: 'rgba(255,255,255,0.15)', margin: '0 4px' }} />

        <div
          style={{
            display: 'flex',
            background: 'rgba(255,255,255,0.06)',
            borderRadius: 6,
            padding: 2,
            gap: 2,
          }}
        >
          <Button
            size="small"
            type={viewMode === 'perspective' ? 'primary' : 'text'}
            onClick={() => setPresetView('perspective')}
            style={{ fontSize: 11 }}
          >
            3D (1)
          </Button>
          <Button
            size="small"
            type={viewMode === 'top' ? 'primary' : 'text'}
            onClick={() => setPresetView('top')}
            style={{ fontSize: 11 }}
          >
            Top BEV (2)
          </Button>
          <Button
            size="small"
            type={viewMode === 'side' ? 'primary' : 'text'}
            onClick={() => setPresetView('side')}
            style={{ fontSize: 11 }}
          >
            Side (3)
          </Button>
          <Button
            size="small"
            type={viewMode === 'front' ? 'primary' : 'text'}
            onClick={() => setPresetView('front')}
            style={{ fontSize: 11 }}
          >
            Front (4)
          </Button>
        </div>
      </div>

      {/* OffscreenCanvas Multi-View Orthographic Dock (Worker 2 + Dedicated Worker WebGLRenderer) */}
      <div
        onMouseDown={(e) => e.stopPropagation()}
        onMouseMove={(e) => e.stopPropagation()}
        style={{
          position: 'absolute',
          bottom: 54,
          right: 20,
          width: ORTHO_DOCK_WIDTH,
          background: 'rgba(9, 13, 22, 0.92)',
          border: '1px solid rgba(56, 189, 248, 0.28)',
          borderRadius: 10,
          overflow: 'hidden',
          boxShadow: '0 10px 30px rgba(0, 0, 0, 0.55)',
          backdropFilter: 'blur(8px)',
        }}
      >
        {/* Multi-View Dock Status Header */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '5px 10px',
            background: 'rgba(15, 23, 42, 0.95)',
            borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
            fontSize: 11,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: '50%',
                background: '#10b981',
                display: 'inline-block',
              }}
            />
            <span style={{ fontWeight: 600, color: '#e2e8f0' }}>
              Multi-View Orthographic Dock
            </span>
            <Tag color="geekblue" style={{ margin: 0, fontSize: 10, lineHeight: '16px' }}>
              Worker 1: Main 3D
            </Tag>
            <Tag color="cyan" style={{ margin: 0, fontSize: 10, lineHeight: '16px' }}>
              Worker 2: OffscreenCanvas WebGL
            </Tag>
          </div>
          <span style={{ color: '#94a3b8', fontSize: 10 }}>
            In-Box: <strong style={{ color: '#38bdf8' }}>{orthoStats.inBox}</strong> / ROI:{' '}
            {orthoStats.roiTotal} pts
          </span>
        </div>

        {/* OffscreenCanvas Mount + Interactive Sub-View Overlays */}
        <div style={{ position: 'relative', width: ORTHO_DOCK_WIDTH, height: ORTHO_DOCK_HEIGHT }}>
          <div
            ref={orthoMountRef}
            style={{ width: ORTHO_DOCK_WIDTH, height: ORTHO_DOCK_HEIGHT }}
          />

          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              pointerEvents: 'auto',
            }}
          >
            {orthoSubViews.map((sv, idx) => {
              const isSelected = viewMode === sv.key;
              return (
                <div
                  key={sv.key}
                  onClick={() => setPresetView(sv.key)}
                  onWheel={(e) => handleOrthoSubViewWheel(e, sv.key)}
                  title={`Click to sync main camera to ${sv.title} (${sv.shortcut}) | Scroll to zoom orthographic view`}
                  style={{
                    position: 'relative',
                    cursor: 'pointer',
                    borderRight:
                      idx < 2 ? '1px solid rgba(255, 255, 255, 0.12)' : undefined,
                    boxShadow: isSelected
                      ? 'inset 0 0 0 2px #38bdf8'
                      : 'inset 0 0 0 1px transparent',
                    transition: 'box-shadow 0.15s ease',
                  }}
                >
                  <div
                    style={{
                      position: 'absolute',
                      top: 6,
                      left: 8,
                      right: 8,
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      pointerEvents: 'none',
                    }}
                  >
                    <span
                      style={{
                        fontSize: 10,
                        fontWeight: 700,
                        color: isSelected ? '#38bdf8' : '#cbd5e1',
                        background: 'rgba(15, 23, 42, 0.78)',
                        padding: '1px 6px',
                        borderRadius: 4,
                      }}
                    >
                      {sv.title}
                    </span>
                    <span
                      style={{
                        fontSize: 9,
                        color: '#64748b',
                        background: 'rgba(15, 23, 42, 0.75)',
                        padding: '1px 5px',
                        borderRadius: 4,
                      }}
                    >
                      Key {sv.shortcut}
                    </span>
                  </div>
                  <div
                    style={{
                      position: 'absolute',
                      bottom: 5,
                      left: 8,
                      fontSize: 9,
                      color: 'rgba(148, 163, 184, 0.85)',
                      background: 'rgba(15, 23, 42, 0.7)',
                      padding: '1px 5px',
                      borderRadius: 3,
                      pointerEvents: 'none',
                    }}
                  >
                    {sv.subtitle}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Snap Feedback Badge */}
      {lastSnap && (
        <div
          style={{
            position: 'absolute',
            bottom: 20,
            left: 20,
            background: 'rgba(16, 185, 129, 0.15)',
            border: '1px solid rgba(16, 185, 129, 0.4)',
            padding: '10px 16px',
            borderRadius: '8px',
            color: '#10b981',
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            backdropFilter: 'blur(6px)',
          }}
        >
          <CheckCircleOutlined style={{ fontSize: 18 }} />
          <div>
            <div style={{ fontWeight: 600, fontSize: 13 }}>Road Snapped & Oriented</div>
            <div style={{ fontSize: 11, color: 'rgba(255, 255, 255, 0.75)' }}>
              Road Elevation: {lastSnap.groundZ}m | ΔZ:{' '}
              {lastSnap.deltaZ > 0 ? `+${lastSnap.deltaZ}` : lastSnap.deltaZ}m | Yaw Adjustment:{' '}
              {lastSnap.deltaYawDeg > 0 ? `+${lastSnap.deltaYawDeg}` : lastSnap.deltaYawDeg}° |
              Points: {lastSnap.pointsCount}
            </div>
          </div>
          {previousCuboid && (
            <Button
              size="small"
              type="primary"
              ghost
              icon={<UndoOutlined />}
              onClick={revertSnap}
              style={{ marginLeft: 8 }}
            >
              Revert
            </Button>
          )}
        </div>
      )}

      {/* Legend / Tips at bottom right */}
      <div
        style={{
          position: 'absolute',
          bottom: 14,
          right: 20,
          background: 'rgba(15, 23, 42, 0.75)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          padding: '5px 10px',
          borderRadius: '6px',
          fontSize: '11px',
          color: 'rgba(255, 255, 255, 0.5)',
          pointerEvents: 'none',
        }}
      >
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
          <span>Navigate: LMB/RMB/Scroll</span>
          <Tag color="cyan">1-4: Views (Top/Side/Front/3D)</Tag>
          <Tag color="green">F: Snap</Tag>
          <Tag color="purple">G: Strip Road</Tag>
          <Tag color="orange">H: Invert 180°</Tag>
          <Tag color="blue">Arrows: Micro-Nudge</Tag>
          <Tag color="geekblue">PgUp/Dn: Z</Tag>
          <Tag color="gold">[/]: Yaw</Tag>
        </div>
      </div>
    </div>
  );
};
