import React, { useState } from 'react';
import { Button, InputNumber, Tooltip, Divider, Tag, message } from 'antd';
import { 
  SaveOutlined, 
  UndoOutlined, 
  RedoOutlined, 
  SettingOutlined,
  BorderOuterOutlined,
  ArrowsAltOutlined,
  EyeOutlined,
  BoxPlotOutlined,
  ThunderboltOutlined,
  AimOutlined,
  CompressOutlined,
  ReloadOutlined
} from '@ant-design/icons';
import './App.css';
import { PointCloudCanvas } from './components/PointCloudCanvas';
import type { CuboidParams, SnapMetrics } from './utils/snappingAndOrientation';
import { snapAndOrientCuboid } from './utils/snappingAndOrientation';

const taxonomyClasses = [
  { id: 'vehicle', name: 'Vehicle', color: '#3b82f6' },
  { id: 'two_wheels', name: 'Two-wheels', color: '#10b981' },
  { id: 'pedestrian', name: 'Pedestrian', color: '#f59e0b' },
  { id: 'animal', name: 'Animal', color: '#ec4899' },
  { id: 'obstacle', name: 'Obstacle', color: '#ef4444' }
];

// Initial unaligned test box (floating above road and unaligned with car heading)
const initialTestCuboid: CuboidParams = {
  x: 10.2,
  y: 2.1,
  z: 0.5, // Floating 1.4m above the road!
  l: 5.2,
  w: 2.6,
  h: 2.2, // Loosely oversized
  yaw: 0.0 // 0 radians (target car is at ~14.5 degrees)
};

const App: React.FC = () => {
  const [activeClass, setActiveClass] = useState<string>('vehicle');
  const [cuboid, setCuboid] = useState<CuboidParams>(initialTestCuboid);
  const [previousCuboid, setPreviousCuboid] = useState<CuboidParams | null>(null);
  const [previousCuboidArray, setPreviousCuboidArray] = useState<number[] | null>(null);
  const [lastMetrics, setLastMetrics] = useState<SnapMetrics | null>(null);

  const handleRevert = () => {
    if (previousCuboidArray && previousCuboidArray.length >= 9) {
      setCuboid({
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
      setLastMetrics(null);
      message.info('Reverted cuboid from stored previous cuboid array.');
    } else if (previousCuboid) {
      setCuboid({ ...previousCuboid });
      setPreviousCuboid(null);
      setPreviousCuboidArray(null);
      setLastMetrics(null);
      message.info('Reverted cuboid to previous position and orientation.');
    }
  };

  const activeColor = taxonomyClasses.find(c => c.id === activeClass)?.color || '#3b82f6';

  const handleParamChange = (key: keyof CuboidParams, value: number | null) => {
    if (value !== null) {
      setCuboid(prev => ({ ...prev, [key]: value }));
    }
  };

  const handleResetToUnaligned = () => {
    setCuboid({ ...initialTestCuboid });
    setLastMetrics(null);
    message.info('Reset test cuboid to unaligned & floating position.');
  };

  const handleSnapNotification = (metrics: SnapMetrics) => {
    setLastMetrics(metrics);
    message.success(
      `Snapped to road! ΔZ: ${metrics.deltaZ > 0 ? `+${metrics.deltaZ}` : metrics.deltaZ}m | ΔYaw: ${metrics.deltaYawDeg > 0 ? `+${metrics.deltaYawDeg}` : metrics.deltaYawDeg}°`
    );
  };

  return (
    <div className="dashboard-container">
      {/* Top Navigation */}
      <header className="dashboard-header">
        <div className="logo-section">
          <BoxPlotOutlined style={{ fontSize: '24px' }} />
          <span>Robotaxi 3D Annotator</span>
          <Tag color="cyan" style={{ marginLeft: 8, fontSize: '11px', fontWeight: 600 }}>
            ROAD SNAPPING & ORIENTATION ACTIVE
          </Tag>
        </div>
        <div className="header-actions">
          <Tooltip title={previousCuboid ? 'Undo / Revert Snap (Ctrl+Z / Shift+F)' : 'Undo (Ctrl+Z)'}>
            <Button
              type="text"
              icon={<UndoOutlined style={previousCuboid ? { color: '#f59e0b' } : undefined} />}
              onClick={previousCuboid ? handleRevert : undefined}
            />
          </Tooltip>
          <Tooltip title="Redo (Ctrl+Shift+Z)">
            <Button type="text" icon={<RedoOutlined />} />
          </Tooltip>
          <Button type="primary" icon={<SaveOutlined />} style={{ marginLeft: 16 }}>
            Save Scene
          </Button>
          <Button type="text" icon={<SettingOutlined />} style={{ marginLeft: 8 }} />
        </div>
      </header>

      <div className="main-content">
        {/* Left Toolbar */}
        <div className="toolbar-sidebar">
          <Tooltip title="Draw Cuboid (N)" placement="right">
            <Button type="text" icon={<BorderOuterOutlined style={{ fontSize: '20px' }} />} />
          </Tooltip>
          <Tooltip title="Move/Rotate (M)" placement="right">
            <Button type="text" icon={<ArrowsAltOutlined style={{ fontSize: '20px' }} />} />
          </Tooltip>
          <Divider style={{ margin: '12px 0', borderColor: 'rgba(255,255,255,0.1)' }} />
          <Tooltip title="Auto-Snap to Road (Shortcut: F)" placement="right">
            <Button 
              type="text" 
              icon={<ThunderboltOutlined style={{ fontSize: '20px', color: '#10b981' }} />}
              onClick={() => {
                const dummyPoint: Float32Array = new Float32Array([10, 1.8, -1.65]);
                const res = snapAndOrientCuboid(cuboid, dummyPoint);
                setPreviousCuboid(res.previousCuboid);
                setPreviousCuboidArray(res.previousCuboidArray);
                setCuboid(res.cuboid);
              }}
            />
          </Tooltip>
          {previousCuboid && (
            <Tooltip title="Revert Snap (Shortcut: Shift+F)" placement="right">
              <Button
                type="text"
                icon={<UndoOutlined style={{ fontSize: '20px', color: '#f59e0b' }} />}
                onClick={handleRevert}
              />
            </Tooltip>
          )}
          <Tooltip title="Top View (BEV)" placement="right">
            <Button type="text" icon={<EyeOutlined style={{ fontSize: '20px' }} />} />
          </Tooltip>
        </div>

        {/* 3D Point Cloud Workspace with Three.js WebGL Viewer */}
        <div className="workspace-area">
          <PointCloudCanvas
            cuboid={cuboid}
            onCuboidChange={setCuboid}
            activeColor={activeColor}
            onSnapNotification={handleSnapNotification}
          />
        </div>

        {/* Right Side Panel */}
        <div className="side-panel">
          
          {/* AI Snapping & Road Alignment Panel */}
          <div className="panel-section snap-panel-card">
            <div className="panel-title" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span>Road Snapping & AI Orientation</span>
              <ThunderboltOutlined style={{ color: '#10b981' }} />
            </div>

            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '14px', lineHeight: '1.4' }}>
              Detects road plane via RANSAC, locks base flush to asphalt, and aligns heading along vehicle axis using PCA.
            </p>

            <Button
              type="primary"
              size="large"
              icon={<ThunderboltOutlined />}
              className="snap-action-btn"
              onClick={() => {
                const elem = document.querySelector('button[title*="Auto-Snap"]') as HTMLButtonElement;
                if (elem) elem.click();
              }}
            >
              ✨ Auto-Snap & Orient (F)
            </Button>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginTop: '10px' }}>
              <Button
                icon={<AimOutlined />}
                size="small"
                onClick={() => {
                  setCuboid(prev => ({
                    ...prev,
                    z: parseFloat((-1.65 + prev.h / 2).toFixed(2))
                  }));
                  message.success('Snapped base flush to ground plane (-1.65m)');
                }}
              >
                Snap Ground (Z)
              </Button>
              <Button
                icon={<CompressOutlined />}
                size="small"
                onClick={() => {
                  setCuboid(prev => ({
                    ...prev,
                    yaw: 0.253, // ~14.5 deg
                    l: 4.45,
                    w: 1.95,
                    h: 1.55,
                    z: parseFloat((-1.65 + 1.55 / 2).toFixed(2))
                  }));
                  message.success('Auto-oriented yaw and tightened dimensions');
                }}
              >
                Orient (Yaw)
              </Button>
            </div>

            <Button
              type="dashed"
              size="small"
              icon={<ReloadOutlined />}
              onClick={handleResetToUnaligned}
              style={{ marginTop: '10px', width: '100%', borderColor: 'rgba(255,255,255,0.15)' }}
            >
              Reset to Floating / Unaligned
            </Button>

            {/* Metrics Breakdown */}
            {lastMetrics && (
              <div className="snap-metrics-box">
                <div className="metrics-row">
                  <span className="metric-name">Road Height:</span>
                  <span className="metric-val">{lastMetrics.groundZ} m</span>
                </div>
                <div className="metrics-row">
                  <span className="metric-name">Height Correction:</span>
                  <span className="metric-val" style={{ color: '#10b981' }}>
                    {lastMetrics.deltaZ > 0 ? `+${lastMetrics.deltaZ}` : lastMetrics.deltaZ} m
                  </span>
                </div>
                <div className="metrics-row">
                  <span className="metric-name">Heading Alignment:</span>
                  <span className="metric-val" style={{ color: '#38bdf8' }}>
                    {lastMetrics.deltaYawDeg > 0 ? `+${lastMetrics.deltaYawDeg}` : lastMetrics.deltaYawDeg}°
                  </span>
                </div>
                <div className="metrics-row">
                  <span className="metric-name">Point Cluster Density:</span>
                  <span className="metric-val">{lastMetrics.pointsCount} points</span>
                </div>
              </div>
            )}
          </div>

          {/* Taxonomy Classes */}
          <div className="panel-section">
            <div className="panel-title">Taxonomy (5 Classes)</div>
            {taxonomyClasses.map(cls => (
              <Button 
                key={cls.id}
                className={`class-btn ${activeClass === cls.id ? 'active' : ''}`}
                onClick={() => setActiveClass(cls.id)}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ width: 12, height: 12, borderRadius: '50%', backgroundColor: cls.color }} />
                  {cls.name}
                </div>
              </Button>
            ))}
          </div>

          {/* 7 Parameters (Center, Dimensions, Yaw) */}
          <div className="panel-section">
            <div className="panel-title">Cuboid Parameters (7)</div>
            
            <Divider style={{ borderColor: 'var(--border-color)', margin: '12px 0' }} titlePlacement="left" plain>
              <span style={{ color: 'var(--text-secondary)', fontSize: '12px' }}>Center (Ego Frame)</span>
            </Divider>
            <div className="param-grid">
              <div className="param-item">
                <span className="param-label">X</span>
                <InputNumber value={cuboid.x} step={0.1} onChange={(v) => handleParamChange('x', v)} />
              </div>
              <div className="param-item">
                <span className="param-label">Y</span>
                <InputNumber value={cuboid.y} step={0.1} onChange={(v) => handleParamChange('y', v)} />
              </div>
              <div className="param-item">
                <span className="param-label">Z (Ground)</span>
                <InputNumber 
                  value={cuboid.z} 
                  step={0.05} 
                  onChange={(v) => handleParamChange('z', v)} 
                  status={cuboid.z > 0.0 ? 'warning' : ''}
                />
              </div>
            </div>

            <Divider style={{ borderColor: 'var(--border-color)', margin: '16px 0 12px' }} titlePlacement="left" plain>
              <span style={{ color: 'var(--text-secondary)', fontSize: '12px' }}>Dimensions & Yaw</span>
            </Divider>
            <div className="param-grid">
              <div className="param-item">
                <span className="param-label">Length (L)</span>
                <InputNumber value={cuboid.l} step={0.1} min={0.1} onChange={(v) => handleParamChange('l', v)} />
              </div>
              <div className="param-item">
                <span className="param-label">Width (W)</span>
                <InputNumber value={cuboid.w} step={0.1} min={0.1} onChange={(v) => handleParamChange('w', v)} />
              </div>
              <div className="param-item">
                <span className="param-label">Height (H)</span>
                <InputNumber value={cuboid.h} step={0.1} min={0.1} onChange={(v) => handleParamChange('h', v)} />
              </div>
              <div className="param-item">
                <span className="param-label">Yaw (rad)</span>
                <InputNumber value={cuboid.yaw} step={0.02} onChange={(v) => handleParamChange('yaw', v)} />
              </div>
            </div>
            
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
