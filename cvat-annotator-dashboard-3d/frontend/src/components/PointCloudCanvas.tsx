import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import type { CuboidParams, SnapMetrics } from '../utils/snappingAndOrientation';
import { snapAndOrientCuboid } from '../utils/snappingAndOrientation';
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

  const [lastSnap, setLastSnap] = useState<SnapMetrics | null>(null);
  const [previousCuboid, setPreviousCuboid] = useState<CuboidParams | null>(null);
  const [previousCuboidArray, setPreviousCuboidArray] = useState<number[] | null>(null);
  const [internalViewMode, setInternalViewMode] = useState<CameraViewMode>('perspective');
  const viewMode = currentViewMode ?? internalViewMode;
  const hasActiveFilterRef = useRef<boolean>(false);

  // Camera Orbit State
  const isDraggingRef = useRef(false);
  const dragButtonRef = useRef(0);
  const prevMouseRef = useRef({ x: 0, y: 0 });
  const orbitRef = useRef({
    theta: Math.PI / 4,
    phi: Math.PI / 3.5,
    radius: 28,
    target: new THREE.Vector3(8, 0, -1.0),
  });

  // 1. Generate Synthetic 64-beam LiDAR Point Cloud with target vehicle
  const generateLiDARScene = useCallback(() => {
    const pts: number[] = [];
    const colors: number[] = [];

    // Ground Plane & Road Markings
    const groundZ = -1.65;
    for (let r = 2; r < 55; r += 0.4) {
      const numTheta = Math.floor(r * 24);
      for (let t = 0; t < numTheta; t++) {
        const angle = (t / numTheta) * Math.PI * 2;
        const x = r * Math.cos(angle);
        const y = r * Math.sin(angle);
        const z = groundZ + (Math.random() - 0.5) * 0.02;

        pts.push(x, y, z);

        // Lane markings intensity (y = +-1.75, +-5.25)
        const isLane = Math.abs(Math.abs(y) - 1.75) < 0.12 || Math.abs(Math.abs(y) - 5.25) < 0.12;
        if (isLane) {
          colors.push(0.95, 0.9, 0.3); // Bright yellow lane markings
        } else {
          const depthShade = Math.max(0.15, 0.55 - r / 100);
          colors.push(depthShade * 0.5, depthShade * 0.7, depthShade);
        }
      }
    }

    // Target vehicle point cluster around x=10.0, y=1.8 (heading ~15 degrees)
    const vehCenterX = 10.0;
    const vehCenterY = 1.8;
    const vehYaw = THREE.MathUtils.degToRad(14.5);
    const cosY = Math.cos(vehYaw);
    const sinY = Math.sin(vehYaw);

    const vehL = 4.4;
    const vehW = 1.9;
    const vehH = 1.5;

    for (let i = 0; i < 950; i++) {
      // Points on exterior faces of car
      const sign = Math.random() > 0.5 ? 1 : -1;
      let lx: number, ly: number, lz: number;

      const face = Math.floor(Math.random() * 3);
      if (face === 0) {
        // Side face
        ly = (sign * vehW) / 2 + (Math.random() - 0.5) * 0.05;
        lx = (Math.random() - 0.5) * vehL;
        lz = Math.random() * vehH;
      } else if (face === 1) {
        // Front/Rear face
        lx = (sign * vehL) / 2 + (Math.random() - 0.5) * 0.05;
        ly = (Math.random() - 0.5) * vehW;
        lz = Math.random() * vehH;
      } else {
        // Roof
        lz = vehH + (Math.random() - 0.5) * 0.04;
        lx = (Math.random() - 0.5) * vehL * 0.7;
        ly = (Math.random() - 0.5) * vehW * 0.85;
      }

      // Rotate to world
      const wx = vehCenterX + (lx * cosY - ly * sinY);
      const wy = vehCenterY + (lx * sinY + ly * cosY);
      const wz = groundZ + lz;

      pts.push(wx, wy, wz);
      // High vehicle reflectance
      colors.push(0.3, 0.85, 1.0);
    }

    return {
      positions: new Float32Array(pts),
      colors: new Float32Array(colors),
    };
  }, []);

  // 2. Initialize Three.js WebGL Scene
  useEffect(() => {
    if (!mountRef.current) return;
    const container = mountRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0c0f17);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 500);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
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

    // LiDAR Point Cloud
    const lidarData = generateLiDARScene();
    pointCloudDataRef.current = lidarData.positions;
    originalColorsRef.current = new Float32Array(lidarData.colors);

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(lidarData.positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(lidarData.colors, 3));

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
      color: new THREE.Color(activeColor),
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
      color: new THREE.Color(activeColor),
      linewidth: 2,
    });
    const boxEdges = new THREE.LineSegments(edgesGeo, edgesMat);
    boxMesh.add(boxEdges);
    boxEdgesRef.current = boxEdges;

    // Front Orientation Arrow Helper
    const arrowDir = new THREE.Vector3(1, 0, 0);
    const arrowHelper = new THREE.ArrowHelper(arrowDir, new THREE.Vector3(0, 0, 0), 2.8, 0x00ffcc, 0.6, 0.35);
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
    const ringMat = new THREE.LineBasicMaterial({ color: 0x00ff88, transparent: true, opacity: 0.6 });
    const groundRing = new THREE.Line(ringGeo, ringMat);
    scene.add(groundRing);
    groundRingRef.current = groundRing;

    // Animation Loop
    let animId: number;
    const animate = () => {
      animId = requestAnimationFrame(animate);

      // Update camera position from orbit parameters
      const { theta, phi, radius, target } = orbitRef.current;
      camera.position.x = target.x + radius * Math.sin(phi) * Math.cos(theta);
      camera.position.y = target.y + radius * Math.sin(phi) * Math.sin(theta);
      camera.position.z = target.z + radius * Math.cos(phi);
      camera.lookAt(target);

      renderer.render(scene, camera);
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
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);
      if (renderer.domElement && container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [activeColor, generateLiDARScene]);

  // 3. Update Cuboid Mesh in 3D Scene when props change
  useEffect(() => {
    if (!boxMeshRef.current || !boxEdgesRef.current || !arrowHelperRef.current || !groundRingRef.current) return;

    const mesh = boxMeshRef.current;
    mesh.position.set(cuboid.x, cuboid.y, cuboid.z);
    mesh.scale.set(cuboid.l, cuboid.w, cuboid.h);
    mesh.rotation.z = cuboid.yaw;

    // Update color
    (mesh.material as THREE.MeshBasicMaterial).color.set(activeColor);
    (boxEdgesRef.current.material as THREE.LineBasicMaterial).color.set(activeColor);

    // Update Ground Ring position under the box
    const groundZ = -1.65;
    groundRingRef.current.position.set(cuboid.x, cuboid.y, groundZ + 0.02);
    groundRingRef.current.scale.set(cuboid.l / 1.8, cuboid.w / 1.8, 1);
  }, [cuboid, activeColor]);

  // 4. Mouse Event Listeners for 3D Navigation
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
      orbitRef.current.theta -= dx * 0.008;
      orbitRef.current.phi = Math.max(0.1, Math.min(Math.PI / 2 - 0.05, orbitRef.current.phi - dy * 0.008));
    } else if (dragButtonRef.current === 2 || dragButtonRef.current === 1) {
      // Right/Middle button: Pan
      const panSpeed = orbitRef.current.radius * 0.0012;
      const cosT = Math.cos(orbitRef.current.theta);
      const sinT = Math.sin(orbitRef.current.theta);
      orbitRef.current.target.x += (-dx * sinT - dy * cosT) * panSpeed;
      orbitRef.current.target.y += (dx * cosT - dy * sinT) * panSpeed;
    }
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    orbitRef.current.radius = Math.max(4, Math.min(120, orbitRef.current.radius + e.deltaY * 0.03));
  };

  // 5. The Core Feature: Snapping & Orientation Action
  const executeSnapToRoadAndOrient = useCallback(
    (options?: { snapGround?: boolean; autoOrient?: boolean }) => {
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

  // Dynamic point cloud filtering (Ground Stripping & Object Isolation)
  useEffect(() => {
    if (!pointsMeshRef.current || !pointCloudDataRef.current || !originalColorsRef.current) return;

    // Fast-path: If neither filter is active and previous state had no filter, skip 90k point iteration
    if (!isGroundStripped && !isObjectIsolated) {
      if (!hasActiveFilterRef.current) return;
      hasActiveFilterRef.current = false;
    } else {
      hasActiveFilterRef.current = true;
    }

    const geometry = pointsMeshRef.current.geometry;
    const colorAttr = geometry.getAttribute('color') as THREE.BufferAttribute;
    const positions = pointCloudDataRef.current;
    const orig = originalColorsRef.current;
    const count = positions.length / 3;

    const cx = cuboid.x;
    const cy = cuboid.y;
    const halfL = cuboid.l / 2 + 0.35;
    const halfW = cuboid.w / 2 + 0.35;
    const cosY = Math.cos(cuboid.yaw);
    const sinY = Math.sin(cuboid.yaw);

    for (let i = 0; i < count; i++) {
      const px = positions[i * 3];
      const py = positions[i * 3 + 1];
      const pz = positions[i * 3 + 2];

      let r = orig[i * 3];
      let g = orig[i * 3 + 1];
      let b = orig[i * 3 + 2];

      if (isGroundStripped && pz <= -1.55) {
        r = 0.03;
        g = 0.04;
        b = 0.06;
      }

      if (isObjectIsolated) {
        const dx = px - cx;
        const dy = py - cy;
        const u = dx * cosY + dy * sinY;
        const v = -dx * sinY + dy * cosY;
        const inside = Math.abs(u) <= halfL && Math.abs(v) <= halfW && pz >= cuboid.z - cuboid.h / 2 - 0.1 && pz <= cuboid.z + cuboid.h / 2 + 0.2;

        if (inside) {
          r = 0.2;
          g = 0.95;
          b = 1.0;
        } else {
          r *= 0.15;
          g *= 0.15;
          b *= 0.15;
        }
      }

      colorAttr.setXYZ(i, r, g, b);
    }
    colorAttr.needsUpdate = true;
  }, [isGroundStripped, isObjectIsolated, cuboid]);

  // Keyboard shortcut listener: micro-nudges, snap, revert, heading inversion, modes
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Avoid triggering when focused on input fields
      if (['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) return;

      const stepDist = e.shiftKey ? 0.20 : 0.04;
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
    onToggleObjectIsolation
  ]);

  // View Presets: Top, Side, Front, and Perspective
  const setPresetView = useCallback((mode: CameraViewMode) => {
    if (onViewModeChange) {
      onViewModeChange(mode);
    } else {
      setInternalViewMode(mode);
    }

    const { x, y, z, yaw, l, w, h } = cuboid;
    const targetZ = z || -0.9;

    if (mode === 'top') {
      // Orthographic top-down BEV: phi near 0, yaw aligned
      orbitRef.current.phi = 0.04;
      orbitRef.current.theta = yaw - Math.PI / 2;
      orbitRef.current.radius = Math.max(l, w) * 2.8 + 6;
      orbitRef.current.target.set(x, y, targetZ);
    } else if (mode === 'side') {
      // Side camera view: lateral profile (doors / wheels)
      orbitRef.current.phi = Math.PI / 2 - 0.05;
      orbitRef.current.theta = yaw;
      orbitRef.current.radius = Math.max(l, h) * 2.2 + 4;
      orbitRef.current.target.set(x, y, targetZ);
    } else if (mode === 'front') {
      // Front camera view: front bumper & windshield
      orbitRef.current.phi = Math.PI / 2 - 0.05;
      orbitRef.current.theta = yaw + Math.PI / 2;
      orbitRef.current.radius = Math.max(w, h) * 2.5 + 4;
      orbitRef.current.target.set(x, y, targetZ);
    } else {
      // 3D Perspective orbit
      orbitRef.current.phi = Math.PI / 3.5;
      orbitRef.current.theta = Math.PI / 4;
      orbitRef.current.radius = 28;
      orbitRef.current.target.set(x, y, targetZ);
    }
  }, [cuboid, onViewModeChange]);

  useEffect(() => {
    if (currentViewMode) {
      setPresetView(currentViewMode);
    }
  }, [currentViewMode, setPresetView]);

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
          <Button
            icon={<SwapOutlined />}
            onClick={onInvertHeading}
          >
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

        <div style={{ display: 'flex', background: 'rgba(255,255,255,0.06)', borderRadius: 6, padding: 2, gap: 2 }}>
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
            <div style={{ fontWeight: 600, fontSize: 13 }}>
              Road Snapped & Oriented
            </div>
            <div style={{ fontSize: 11, color: 'rgba(255, 255, 255, 0.75)' }}>
              Road Elevation: {lastSnap.groundZ}m | ΔZ: {lastSnap.deltaZ > 0 ? `+${lastSnap.deltaZ}` : lastSnap.deltaZ}m |
              Yaw Adjustment: {lastSnap.deltaYawDeg > 0 ? `+${lastSnap.deltaYawDeg}` : lastSnap.deltaYawDeg}° | Points: {lastSnap.pointsCount}
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
          bottom: 20,
          right: 20,
          background: 'rgba(15, 23, 42, 0.75)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          padding: '8px 12px',
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
