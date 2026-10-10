// Copyright (C) 2026 CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React from 'react';
import { useSelector } from 'react-redux';
import Icon from '@ant-design/icons';

import { Canvas3d } from 'cvat-canvas3d-wrapper';
import { CombinedState } from 'reducers';
import { SnapToolsIcon } from 'icons';
import CVATTooltip from 'components/common/cvat-tooltip';

export interface Props {
    canvasInstance: Canvas3d;
    disabled?: boolean;
}

function SnapToRoadControl(props: Props): JSX.Element {
    const { canvasInstance, disabled } = props;
    const selectedStatesID = useSelector(
        (state: CombinedState) => state.annotation.annotations.selectedStatesID,
    );

    const onClick = (): void => {
        if (!disabled) {
            const targetID = selectedStatesID.length ? selectedStatesID[0] : null;
            canvasInstance.snapToRoad(targetID);
        }
    };

    return disabled ? (
        <Icon
            className='cvat-snap-to-road-control cvat-disabled-canvas-control'
            component={SnapToolsIcon}
        />
    ) : (
        <CVATTooltip title='Snap to Road & Orient (F)' placement='right'>
            <Icon
                className='cvat-snap-to-road-control'
                component={SnapToolsIcon}
                onClick={onClick}
            />
        </CVATTooltip>
    );
}

Object.assign(SnapToRoadControl, { displayName: 'SnapToRoadControl' });
export default React.memo(SnapToRoadControl);
