// Copyright (C) 2026 CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useState, useEffect } from 'react';
import Icon from '@ant-design/icons';
import Dropdown from 'antd/lib/dropdown';
import notification from 'antd/lib/notification';
import Spin from 'antd/lib/spin';

import { Canvas3d as Canvas } from 'cvat-canvas3d-wrapper';
import CVATTooltip from 'components/common/cvat-tooltip';

export interface Props {
    canvasInstance: Canvas;
    jobInstance: any;
    disabled?: boolean;
}

const LayersStitchIcon = (): JSX.Element => (
    <svg viewBox='0 0 1024 1024' width='1em' height='1em' fill='currentColor'>
        <path d='M512 64l448 208-448 208L64 272 512 64zm0 304l320-148.6L512 110.6 192 219.4 512 368z' />
        <path d='M960 480l-64-29.7-384 178.3-384-178.3L64 480l448 208 448-208z' />
        <path d='M960 688l-64-29.7-384 178.3-384-178.3L64 688l448 208 448-208z' />
    </svg>
);

function SceneStitchControl(props: Props): JSX.Element {
    const { canvasInstance, jobInstance, disabled } = props;
    const [isStitched, setIsStitched] = useState<boolean>(false);
    const [loading, setLoading] = useState<boolean>(false);
    const [windowRadius, setWindowRadius] = useState<number>(() => {
        const saved = localStorage.getItem('cvat-stitch-window');
        return saved ? parseInt(saved, 10) : 1;
    });
    const [voxelSize, setVoxelSize] = useState<number>(() => {
        const saved = localStorage.getItem('cvat-stitch-voxel');
        return saved ? parseFloat(saved) : 0.05;
    });

    useEffect(() => {
        const checkStatus = (): void => {
            if (canvasInstance && typeof (canvasInstance as any).isSceneStitched === 'function') {
                setIsStitched((canvasInstance as any).isSceneStitched());
            }
        };

        checkStatus();
        const container = window.document.querySelector('.cvat-canvas3d-perspective');
        if (container) {
            container.addEventListener('canvas.sceneStitched', checkStatus);
            container.addEventListener('canvas.setup', checkStatus);
        }

        return () => {
            if (container) {
                container.removeEventListener('canvas.sceneStitched', checkStatus);
                container.removeEventListener('canvas.setup', checkStatus);
            }
        };
    }, [canvasInstance]);

    const handleToggle = async (): Promise<void> => {
        if (disabled || loading || !canvasInstance || typeof (canvasInstance as any).toggleSceneStitching !== 'function') {
            return;
        }

        setLoading(true);
        try {
            const notifKey = 'scene_stitch_progress';
            if (!isStitched) {
                notification.open({
                    key: notifKey,
                    message: 'Stitching Point Cloud Scene',
                    description: `Accumulating neighbor frames (±${windowRadius}) and applying Voxel Smoothing (voxel: ${voxelSize * 100}cm)...`,
                    duration: 0,
                });
            }

            const res = await (canvasInstance as any).toggleSceneStitching(
                jobInstance,
                windowRadius,
                voxelSize,
                (msg: string, pct: number) => {
                    notification.open({
                        key: notifKey,
                        message: `Stitching Scene (${pct}%)`,
                        description: msg,
                        duration: 0,
                    });
                },
            );

            notification.destroy(notifKey);
            setIsStitched(res.stitched);

            if (res.stitched) {
                notification.success({
                    key: `stitch_success_${Date.now()}`,
                    message: 'Scene Stitching: ENABLED',
                    description: `Accumulated ${res.pointsCount.toLocaleString()} points (smoothed from ${res.originalCount.toLocaleString()} points with ${voxelSize * 100}cm voxel grid & outlier filtering).`,
                    duration: 3.5,
                });
            } else {
                notification.info({
                    key: `stitch_revert_${Date.now()}`,
                    message: 'Scene Stitching: REVERTED',
                    description: `Restored single-frame point cloud (${res.pointsCount.toLocaleString()} points).`,
                    duration: 2.5,
                });
            }
        } catch (err: any) {
            notification.destroy('scene_stitch_progress');
            notification.error({
                key: `stitch_error_${Date.now()}`,
                message: 'Scene Stitching Failed',
                description: err?.message || 'Error occurred while stitching scene frames',
                duration: 4,
            });
        } finally {
            setLoading(false);
        }
    };

    const updateRadius = (r: number): void => {
        setWindowRadius(r);
        localStorage.setItem('cvat-stitch-window', r.toString());
        notification.info({
            message: `Stitch Window: ±${r} Frame${r > 1 ? 's' : ''}`,
            description: `Accumulating ${r * 2 + 1} frames total (${r} past + current + ${r} future).`,
            duration: 2,
        });
    };

    const updateVoxel = (v: number): void => {
        setVoxelSize(v);
        localStorage.setItem('cvat-stitch-voxel', v.toString());
        notification.info({
            message: `Voxel Smoothing: ${v * 100}cm`,
            description: `Point cloud density will be balanced to ${v * 100}cm cubic resolution.`,
            duration: 2,
        });
    };

    const menuItems = [
        {
            key: 'status',
            label: isStitched ? '● Scene Stitching: ACTIVE' : '○ Scene Stitching: INACTIVE',
            onClick: handleToggle,
        },
        {
            type: 'divider' as const,
        },
        {
            key: 'radius_1',
            label: windowRadius === 1 ? '● Window: ±1 Frame (Fast / 3 frames total)' : '○ Window: ±1 Frame (Fast / 3 frames total)',
            onClick: () => updateRadius(1),
        },
        {
            key: 'radius_2',
            label: windowRadius === 2 ? '● Window: ±2 Frames (Dense / 5 frames total)' : '○ Window: ±2 Frames (Dense / 5 frames total)',
            onClick: () => updateRadius(2),
        },
        {
            type: 'divider' as const,
        },
        {
            key: 'voxel_003',
            label: voxelSize === 0.03 ? '● Voxel: 3cm (Ultra Dense)' : '○ Voxel: 3cm (Ultra Dense)',
            onClick: () => updateVoxel(0.03),
        },
        {
            key: 'voxel_005',
            label: voxelSize === 0.05 ? '● Voxel: 5cm (Recommended Balanced)' : '○ Voxel: 5cm (Recommended Balanced)',
            onClick: () => updateVoxel(0.05),
        },
        {
            key: 'voxel_008',
            label: voxelSize === 0.08 ? '● Voxel: 8cm (High FPS / Performance)' : '○ Voxel: 8cm (High FPS / Performance)',
            onClick: () => updateVoxel(0.08),
        },
    ];

    if (disabled) {
        return (
            <Icon
                className='cvat-stitch-scene-control cvat-disabled-canvas-control'
                component={LayersStitchIcon}
            />
        );
    }

    const tooltipTitle = isStitched ?
        'Scene Stitching: Active (Click to Revert, Right-click to Configure)' :
        `Stitch & Smooth Scene (±${windowRadius} frames, ${voxelSize * 100}cm voxel) • Right-click to configure`;

    return (
        <Dropdown menu={{ items: menuItems }} trigger={['contextMenu']}>
            <span className='cvat-stitch-scene-control-wrapper' style={{ display: 'inline-block' }}>
                <CVATTooltip title={tooltipTitle} placement='right'>
                    {loading ? (
                        <Spin size='small' style={{ margin: '8px' }} />
                    ) : (
                        <Icon
                            className={`cvat-stitch-scene-control ${isStitched ? 'cvat-stitch-scene-active' : ''}`}
                            component={LayersStitchIcon}
                            onClick={handleToggle}
                            style={{
                                color: isStitched ? '#1890ff' : undefined,
                                filter: isStitched ? 'drop-shadow(0px 0px 4px rgba(24, 144, 255, 0.8))' : undefined,
                            }}
                        />
                    )}
                </CVATTooltip>
            </span>
        </Dropdown>
    );
}

Object.assign(SceneStitchControl, { displayName: 'SceneStitchControl' });
export default React.memo(SceneStitchControl);
