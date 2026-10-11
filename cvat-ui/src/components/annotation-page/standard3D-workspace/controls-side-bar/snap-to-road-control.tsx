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
}

function SnapToRoadControl(props: Props): JSX.Element {
    const { canvasInstance, disabled, jobInstance } = props;
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

    const toggleAutoSnap = (): void => {
        const next = !autoSnap;
        setAutoSnap(next);
        if (canvasInstance && typeof (canvasInstance as any).configure === 'function') {
            (canvasInstance as any).configure({ autoRoadSnap: next });
        }
        notification.info({
            message: `Auto-Snap to Road: ${next ? 'ENABLED' : 'DISABLED'}`,
            description: next ?
                'Cuboids will automatically snap flush to the road pavement when translated.' :
                'Automatic road snapping disabled. Press F to snap manually.',
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
            key: 'auto_snap_toggle',
            label: autoSnap ? '✓ Auto-Snap on Move (Active)' : 'Auto-Snap on Move (Disabled)',
            onClick: toggleAutoSnap,
        },
        {
            type: 'divider' as const,
        },
        {
            key: 'snap',
            label: 'Snap to Road & Orient (F)',
            onClick: () => {
                if (targetID !== null) {
                    canvasInstance.snapToRoad(targetID);
                }
            },
        },
        {
            key: 'invert',
            label: 'Invert Heading 180° (H)',
            onClick: () => {
                if (targetID !== null) {
                    canvasInstance.invertHeading(targetID);
                }
            },
        },
        {
            key: 'snap_track',
            label: 'Snap Track to Road (Alt+F)',
            onClick: async () => {
                if (!targetID || !jobInstance) return;
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
                            notification.open({
                                key: 'track_snap_progress',
                                message: `Snapping Track #${targetID} (${pct}%)`,
                                description: msg,
                                duration: 0,
                            });
                        },
                    );
                    notification.destroy('track_snap_progress');
                    notification.success({
                        key: `track_snap_success_${Date.now()}`,
                        message: `Track #${targetID} Snapped`,
                        description: `Successfully aligned and smoothed ${res.processedFrames} frames along the road.`,
                        duration: 3,
                    });
                } catch (err: any) {
                    notification.destroy('track_snap_progress');
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
            label: 'Revert Snap (Shift+F)',
            onClick: () => {
                if (targetID !== null) {
                    canvasInstance.revertSnap(targetID);
                }
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

    return (
        <Dropdown menu={{ items: menuItems }} trigger={['contextMenu']}>
            <span className='cvat-snap-to-road-control-wrapper'>
                <CVATTooltip title={`Snap to Road (F) • Auto-Snap on Move: ${autoSnap ? 'ACTIVE' : 'OFF'} (Right-click menu)`} placement='right'>
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
