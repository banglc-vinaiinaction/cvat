// Copyright (C) 2026 CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React from 'react';
import { useSelector } from 'react-redux';
import Icon from '@ant-design/icons';
import Dropdown from 'antd/lib/dropdown';
import notification from 'antd/lib/notification';

import { Canvas3d } from 'cvat-canvas3d-wrapper';
import { CombinedState } from 'reducers';
import { SnapToolsIcon } from 'icons';
import CVATTooltip from 'components/common/cvat-tooltip';

export interface Props {
    canvasInstance: Canvas3d;
    disabled?: boolean;
    jobInstance?: any;
    onOpenStandardizeModal?: () => void;
}

function SnapToRoadControl(props: Props): JSX.Element {
    const {
        canvasInstance, disabled, jobInstance, onOpenStandardizeModal,
    } = props;
    const { selectedStatesID, activatedStateID } = useSelector(
        (state: CombinedState) => ({
            selectedStatesID: state.annotation.annotations.selectedStatesID,
            activatedStateID: state.annotation.annotations.activatedStateID,
        }),
    );

    const canvasActiveID = canvasInstance && typeof (canvasInstance as any).getActiveElement === 'function' ?
        (canvasInstance as any).getActiveElement() : null;
    let targetID: number | null = canvasActiveID;
    if (targetID === null) {
        if (activatedStateID !== null) {
            targetID = activatedStateID;
        } else if (selectedStatesID.length) {
            [targetID] = selectedStatesID;
        }
    }

    const [autoSnap, setAutoSnap] = React.useState<boolean>(true);
    const [snapMode, setSnapMode] = React.useState<'ground' | 'full'>(() => {
        const saved = localStorage.getItem('cvat-snap-mode');
        return (saved === 'full' || saved === 'ground') ? saved : 'ground';
    });

    const isMountedRef = React.useRef<boolean>(true);
    React.useEffect(() => {
        isMountedRef.current = true;
        return () => {
            isMountedRef.current = false;
            notification.destroy('track_snap_progress');
        };
    }, []);

    React.useEffect(() => {
        if (canvasInstance && typeof (canvasInstance as any).configure === 'function') {
            (canvasInstance as any).configure({ snapMode, autoRoadSnap: autoSnap });
        }
    }, [canvasInstance, snapMode, autoSnap]);

    const setMode = (mode: 'ground' | 'full'): void => {
        setSnapMode(mode);
        try {
            localStorage.setItem('cvat-snap-mode', mode);
        } catch (_) {
            // ignore localStorage errors
        }
        if (canvasInstance && typeof (canvasInstance as any).configure === 'function') {
            (canvasInstance as any).configure({ snapMode: mode });
        }
        notification.info({
            message: `Snap Mode: ${mode === 'ground' ? 'Ground Only (Safe)' : 'Full Fit (Scratch)'}`,
            description: mode === 'ground' ?
                'Snaps Z flush to road pavement. Preserves bounding box LWH, centroid, and yaw.' :
                'Fits centroid, aligns yaw to points, and snaps to road pavement.',
            duration: 2.5,
        });
    };

    const toggleAutoSnap = (): void => {
        const next = !autoSnap;
        setAutoSnap(next);
        if (canvasInstance && typeof (canvasInstance as any).configure === 'function') {
            (canvasInstance as any).configure({ autoRoadSnap: next });
        }
        notification.info({
            message: `Auto-Snap on Move: ${next ? 'ENABLED' : 'DISABLED'}`,
            description: next ?
                'Cuboids will automatically snap flush to the road pavement when translated.' :
                'Automatic road snapping disabled. Press Alt+S to snap manually.',
            duration: 2.5,
        });
    };

    const onClick = (): void => {
        if (!disabled) {
            canvasInstance.snapToRoad(targetID);
        }
    };

    const menuItems = [
        {
            key: 'mode_ground',
            label: snapMode === 'ground' ? '● Mode: Ground Only (Safe / Pre-labels)' : '○ Mode: Ground Only (Safe / Pre-labels)',
            onClick: () => setMode('ground'),
        },
        {
            key: 'mode_full',
            label: snapMode === 'full' ? '● Mode: Full Fit (Scratch Annotation)' : '○ Mode: Full Fit (Scratch Annotation)',
            onClick: () => setMode('full'),
        },
        {
            type: 'divider' as const,
        },
        {
            key: 'auto_snap_toggle',
            label: autoSnap ? '✓ Auto-Snap on Move (Active)' : 'Auto-Snap on Move (Disabled)',
            onClick: toggleAutoSnap,
        },
        {
            type: 'divider' as const,
        },
        {
            key: 'snap_track',
            label: 'Snap Track to Road (Alt+T)',
            onClick: async () => {
                if (targetID === null || !jobInstance) return;
                try {
                    notification.open({
                        key: 'track_snap_progress',
                        message: `Snapping Track #${targetID}`,
                        description: 'Starting track road snapping...',
                        duration: 0,
                    });
                    const res = await canvasInstance.snapTrackToRoad(
                        jobInstance,
                        targetID,
                        (msg: string, pct: number) => {
                            if (!isMountedRef.current) return;
                            notification.open({
                                key: 'track_snap_progress',
                                message: `Snapping Track #${targetID} (${pct}%)`,
                                description: msg,
                                duration: 0,
                            });
                        },
                    );
                    notification.destroy('track_snap_progress');
                    if (!isMountedRef.current) return;
                    notification.success({
                        key: `track_snap_success_${Date.now()}`,
                        message: `Track #${targetID} Snapped`,
                        description: `Successfully aligned and smoothed ${res.processedFrames} frames along the road.`,
                        duration: 3,
                    });
                } catch (err: any) {
                    notification.destroy('track_snap_progress');
                    if (!isMountedRef.current) return;
                    notification.error({
                        key: `track_snap_error_${Date.now()}`,
                        message: 'Track Snap Failed',
                        description: err?.message || 'Error occurred during track snapping',
                        duration: 3,
                    });
                }
            },
        },
        {
            key: 'revert',
            label: 'Revert Snap (Alt+Shift+S)',
            onClick: () => {
                if (targetID !== null) {
                    canvasInstance.revertSnap(targetID);
                }
            },
        },
        {
            key: 'ground_strip',
            label: 'Toggle Ground Stripping (G)',
            onClick: () => {
                if (canvasInstance?.toggleGroundStripping) {
                    canvasInstance.toggleGroundStripping();
                }
            },
        },
        {
            key: 'isolate',
            label: 'Toggle Object Isolation (Shift+I)',
            onClick: () => {
                if (canvasInstance?.toggleObjectIsolation) {
                    canvasInstance.toggleObjectIsolation(targetID);
                }
            },
        },
        ...(onOpenStandardizeModal ? [
            {
                type: 'divider' as const,
            },
            {
                key: 'standardize',
                label: 'Standardize Dimensions (En-Masse)...',
                onClick: onOpenStandardizeModal,
            },
        ] : []),
        {
            type: 'divider' as const,
        },
        {
            key: 'perf_hud',
            label: 'Toggle 3D GPU & Performance HUD',
            onClick: () => {
                const enabled = (canvasInstance as any)?.togglePerformanceHud?.();
                const stats = (canvasInstance as any)?.getPerformanceStats?.();
                const gpuDesc = stats?.gpuTier ?
                    ` (${String(stats.gpuTier).toUpperCase()} • ${stats.renderingMethod || '4-View WebGL2'})` :
                    '';
                notification.info({
                    message: `3D GPU & Performance HUD: ${enabled === false ? 'HIDDEN' : 'VISIBLE'}`,
                    description: `Live GPU tier, scene worker status, and frame timing overlay${gpuDesc}.`,
                    duration: 2.5,
                });
            },
        },
    ];

    if (disabled) {
        return (
            <Icon
                className='cvat-snap-to-road-control cvat-disabled-canvas-control'
                component={SnapToolsIcon}
            />
        );
    }

    const modeLabel = snapMode === 'ground' ? 'Ground Only' : 'Full Fit';

    return (
        <Dropdown menu={{ items: menuItems }} trigger={['contextMenu']}>
            <span className='cvat-snap-to-road-control-wrapper'>
                <CVATTooltip
                    title={`Snap (Alt+S): ${modeLabel} • Auto-Snap: ${autoSnap ? 'ON' : 'OFF'} (Right-click to configure)`}
                    placement='right'
                >
                    <Icon
                        className='cvat-snap-to-road-control'
                        component={SnapToolsIcon}
                        onClick={onClick}
                    />
                </CVATTooltip>
            </span>
        </Dropdown>
    );
}

Object.assign(SnapToRoadControl, { displayName: 'SnapToRoadControl' });
export default React.memo(SnapToRoadControl);
