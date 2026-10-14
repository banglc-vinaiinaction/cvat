// Copyright (C) 2021-2022 Intel Corporation
// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React from 'react';
import { useSelector } from 'react-redux';
import Layout from 'antd/lib/layout';
import notification from 'antd/lib/notification';
import { ActiveControl, CombinedState } from 'reducers';
import { Label, LabelType } from 'cvat-core-wrapper';
import { Canvas3d as Canvas } from 'cvat-canvas3d-wrapper';
import MoveControl, {
    Props as MoveControlProps,
} from 'components/annotation-page/standard-workspace/controls-side-bar/move-control';
import CursorControl, {
    Props as CursorControlProps,
} from 'components/annotation-page/standard-workspace/controls-side-bar/cursor-control';
import DrawCuboidControl, {
    Props as DrawCuboidControlProps,
} from 'components/annotation-page/standard-workspace/controls-side-bar/draw-cuboid-control';
import GroupControl, {
    Props as GroupControlProps,
} from 'components/annotation-page/standard-workspace/controls-side-bar/group-control';
import MergeControl, {
    Props as MergeControlProps,
} from 'components/annotation-page/standard-workspace/controls-side-bar/merge-control';
import SplitControl, {
    Props as SplitControlProps,
} from 'components/annotation-page/standard-workspace/controls-side-bar/split-control';
import SnapToRoadControl, {
    Props as SnapToRoadControlProps,
} from './snap-to-road-control';
import SceneStitchControl, {
    Props as SceneStitchControlProps,
} from './scene-stitch-control';
import GlobalHotKeys, { KeyMap, KeyMapItem } from 'utils/mousetrap-react';
import ControlVisibilityObserver from 'components/annotation-page/standard-workspace/controls-side-bar/control-visibility-observer';
import { filterApplicableForType } from 'utils/filter-applicable-labels';
import { subKeyMap } from 'utils/component-subkeymap';
import { ShortcutScope } from 'utils/enums';
import { registerComponentShortcuts } from 'actions/shortcuts-actions';

interface Props {
    keyMap: KeyMap;
    canvasInstance: Canvas;
    activeControl: ActiveControl;
    normalizedKeyMap: Record<string, string>;
    labels: Label[];
    jobInstance: any;
    repeatDrawShape(): void;
    redrawShape(): void;
    pasteShape(): void;
    resetGroup(): void;
    updateActiveControl(activeControl: ActiveControl): void;
}

const componentShortcuts: Record<string, KeyMapItem> = {
    PASTE_SHAPE: {
        name: 'Paste shape',
        description: 'Paste a shape from internal CVAT clipboard',
        sequences: ['ctrl+v', 'command+v'],
        scope: ShortcutScope.OBJECTS_SIDEBAR,
    },
    SWITCH_DRAW_MODE_STANDARD_3D_CONTROLS: {
        name: 'Draw mode',
        description:
            'Repeat the latest procedure of drawing with the same parameters',
        sequences: ['n'],
        scope: ShortcutScope['3D_ANNOTATION_WORKSPACE_CONTROLS'],
    },
    SWITCH_REDRAW_MODE_STANDARD_3D_CONTROLS: {
        name: 'Redraw shape',
        description: 'Remove selected shape and redraw it from scratch',
        sequences: ['shift+n'],
        scope: ShortcutScope['3D_ANNOTATION_WORKSPACE_CONTROLS'],
    },
    SWITCH_GROUP_MODE_STANDARD_3D_CONTROLS: {
        name: 'Group mode',
        description: 'Activate or deactivate mode to grouping shapes',
        sequences: ['g'],
        scope: ShortcutScope['3D_ANNOTATION_WORKSPACE_CONTROLS'],
    },
    RESET_GROUP_STANDARD_3D_CONTROLS: {
        name: 'Reset group',
        description: 'Reset group for selected shapes (in group mode)',
        sequences: ['shift+g'],
        scope: ShortcutScope['3D_ANNOTATION_WORKSPACE_CONTROLS'],
    },
    SWITCH_MERGE_MODE_STANDARD_3D_CONTROLS: {
        name: 'Merge mode',
        description: 'Activate or deactivate mode to merging shapes',
        sequences: ['m'],
        scope: ShortcutScope['3D_ANNOTATION_WORKSPACE_CONTROLS'],
    },
    SWITCH_SPLIT_MODE_STANDARD_3D_CONTROLS: {
        name: 'Split mode',
        description: 'Activate or deactivate mode to splitting shapes',
        sequences: ['alt+m'],
        scope: ShortcutScope['3D_ANNOTATION_WORKSPACE_CONTROLS'],
    },
    INVERT_HEADING_3D_CONTROLS: {
        name: 'Invert Heading (180°)',
        description: 'Invert selected 3D cuboid heading by 180 degrees',
        sequences: ['alt+h'],
        scope: ShortcutScope['3D_ANNOTATION_WORKSPACE_CONTROLS'],
    },
    ISOLATE_OBJECT_3D_CONTROLS: {
        name: 'Toggle Object Isolation',
        description: 'Clip point cloud to local volume around active cuboid and dim scene clutter',
        sequences: ['shift+i'],
        scope: ShortcutScope['3D_ANNOTATION_WORKSPACE_CONTROLS'],
    },
    TOGGLE_SCENE_STITCHING_3D_CONTROLS: {
        name: 'Toggle Scene Stitching',
        description: 'Accumulate and smooth point cloud from neighbor frames',
        sequences: ['alt+m'],
        scope: ShortcutScope['3D_ANNOTATION_WORKSPACE_CONTROLS'],
    },
};

registerComponentShortcuts(componentShortcuts);

const ObservedCursorControl = ControlVisibilityObserver<CursorControlProps>(CursorControl, 'CursorControl');
const ObservedMoveControl = ControlVisibilityObserver<MoveControlProps>(MoveControl, 'MoveControl');
const ObservedDrawCuboidControl = ControlVisibilityObserver<DrawCuboidControlProps>(DrawCuboidControl, 'DrawCuboidControl');
const ObservedSnapToRoadControl = ControlVisibilityObserver<SnapToRoadControlProps>(SnapToRoadControl, 'SnapToRoadControl');
const ObservedSceneStitchControl = ControlVisibilityObserver<SceneStitchControlProps>(SceneStitchControl, 'SceneStitchControl');
const ObservedGroupControl = ControlVisibilityObserver<GroupControlProps>(GroupControl, 'GroupControl');
const ObservedMergeControl = ControlVisibilityObserver<MergeControlProps>(MergeControl, 'MergeControl');
const ObservedSplitControl = ControlVisibilityObserver<SplitControlProps>(SplitControl, 'SplitControl');

export default function ControlsSideBarComponent(props: Props): JSX.Element {
    const {
        canvasInstance,
        pasteShape,
        activeControl,
        normalizedKeyMap,
        keyMap,
        labels,
        redrawShape,
        repeatDrawShape,
        updateActiveControl,
        resetGroup,
    } = props;

    const { selectedStatesID, activatedStateID } = useSelector(
        (state: CombinedState) => ({
            selectedStatesID: state.annotation.annotations.selectedStatesID,
            activatedStateID: state.annotation.annotations.activatedStateID,
        }),
    );
    const canvasActiveID = canvasInstance && typeof (canvasInstance as any).getActiveElement === 'function' ?
        (canvasInstance as any).getActiveElement() : null;
    let activeTargetID: number | null = canvasActiveID;
    if (activeTargetID === null) {
        if (activatedStateID !== null) {
            activeTargetID = activatedStateID;
        } else if (selectedStatesID.length) {
            [activeTargetID] = selectedStatesID;
        }
    }

    const applicableLabels = filterApplicableForType(LabelType.CUBOID, labels);
    const preventDefault = (event: KeyboardEvent | undefined): void => {
        if (event) {
            event.preventDefault();
        }
    };

    const handleDrawMode = (event: KeyboardEvent | undefined, action: 'draw' | 'redraw'): void => {
        preventDefault(event);
        const drawing = [ActiveControl.DRAW_CUBOID].includes(activeControl);
        if (!drawing) {
            canvasInstance.cancel();
            if (action === 'draw') {
                repeatDrawShape();
            } else {
                redrawShape();
            }
        } else {
            canvasInstance.draw({ enabled: false });
        }
    };

    const dynamicMergeIconProps =
        activeControl === ActiveControl.MERGE ?
            {
                className: 'cvat-merge-control cvat-active-canvas-control',
                onClick: (): void => {
                    canvasInstance.merge({ enabled: false });
                    updateActiveControl(ActiveControl.CURSOR);
                },
            } :
            {
                className: 'cvat-merge-control',
                onClick: (): void => {
                    canvasInstance.cancel();
                    canvasInstance.merge({ enabled: true });
                    updateActiveControl(ActiveControl.MERGE);
                },
            };

    const dynamicGroupIconProps =
        activeControl === ActiveControl.GROUP ?
            {
                className: 'cvat-group-control cvat-active-canvas-control',
                onClick: (): void => {
                    canvasInstance.group({ enabled: false });
                    updateActiveControl(ActiveControl.CURSOR);
                },
            } :
            {
                className: 'cvat-group-control',
                onClick: (): void => {
                    canvasInstance.cancel();
                    canvasInstance.group({ enabled: true });
                    updateActiveControl(ActiveControl.GROUP);
                },
            };

    const dynamicTrackIconProps = activeControl === ActiveControl.SPLIT ?
        {
            className: 'cvat-split-track-control cvat-active-canvas-control',
            onClick: (): void => {
                canvasInstance.split({ enabled: false });
            },
        } :
        {
            className: 'cvat-split-track-control',
            onClick: (): void => {
                canvasInstance.cancel();
                canvasInstance.split({ enabled: true });
                updateActiveControl(ActiveControl.SPLIT);
            },
        };

    let handlers: any = {
        SWITCH_GROUP_MODE_STANDARD_3D_CONTROLS: (event: KeyboardEvent | undefined): void => {
            if (event) event.preventDefault();
            dynamicGroupIconProps.onClick();
        },
        RESET_GROUP_STANDARD_3D_CONTROLS: (event: KeyboardEvent | undefined): void => {
            if (event) event.preventDefault();
            const grouping = activeControl === ActiveControl.GROUP;
            if (!grouping) {
                return;
            }
            resetGroup();
            canvasInstance.group({ enabled: false });
            updateActiveControl(ActiveControl.CURSOR);
        },
        SWITCH_MERGE_MODE_STANDARD_3D_CONTROLS: (event: KeyboardEvent | undefined): void => {
            if (event) event.preventDefault();
            dynamicMergeIconProps.onClick();
        },
        SWITCH_SPLIT_MODE_STANDARD_3D_CONTROLS: (event: KeyboardEvent | undefined) => {
            if (event) event.preventDefault();
            dynamicTrackIconProps.onClick();
        },
        INVERT_HEADING_3D_CONTROLS: (event: KeyboardEvent | undefined) => {
            if (event) event.preventDefault();
            const targetID = activeTargetID;
            if (targetID === null) {
                notification.warning({ message: 'No 3D object selected', duration: 2 });
                return;
            }
            canvasInstance.invertHeading(targetID);
        },
        ISOLATE_OBJECT_3D_CONTROLS: (event: KeyboardEvent | undefined) => {
            if (event) event.preventDefault();
            const targetID = activeTargetID;
            if (targetID === null) {
                notification.warning({ message: 'No 3D object selected', duration: 2 });
                return;
            }
            if (canvasInstance?.toggleObjectIsolation) {
                canvasInstance.toggleObjectIsolation(targetID);
            }
        },
        TOGGLE_SCENE_STITCHING_3D_CONTROLS: (event: KeyboardEvent | undefined) => {
            if (event) event.preventDefault();
            if (canvasInstance && typeof (canvasInstance as any).toggleSceneStitching === 'function') {
                (canvasInstance as any).toggleSceneStitching(jobInstance);
            }
        },
    };

    handlers = applicableLabels.length ? {
        ...handlers,
        PASTE_SHAPE: (event: KeyboardEvent | undefined) => {
            preventDefault(event);
            canvasInstance.cancel();
            pasteShape();
        },
        SWITCH_DRAW_MODE_STANDARD_3D_CONTROLS: (event: KeyboardEvent | undefined) => {
            handleDrawMode(event, 'draw');
        },
        SWITCH_REDRAW_MODE_STANDARD_3D_CONTROLS: (event: KeyboardEvent | undefined) => {
            handleDrawMode(event, 'redraw');
        },
    } : { ...handlers };

    const controlsDisabled = !applicableLabels.length;
    return (
        <Layout.Sider className='cvat-canvas-controls-sidebar' theme='light' width={44}>
            <GlobalHotKeys
                keyMap={applicableLabels.length ? subKeyMap(componentShortcuts, keyMap) : {}}
                handlers={handlers}
            />
            <ObservedCursorControl
                cursorShortkey={normalizedKeyMap.CANCEL}
                canvasInstance={canvasInstance}
                activeControl={activeControl}
            />
            <ObservedMoveControl canvasInstance={canvasInstance} activeControl={activeControl} />
            <ObservedDrawCuboidControl
                canvasInstance={canvasInstance}
                isDrawing={activeControl === ActiveControl.DRAW_CUBOID}
                disabled={controlsDisabled}
            />
            <ObservedSnapToRoadControl
                canvasInstance={canvasInstance}
                jobInstance={jobInstance}
                disabled={controlsDisabled}
            />
            <ObservedSceneStitchControl
                canvasInstance={canvasInstance}
                jobInstance={jobInstance}
                disabled={controlsDisabled}
            />

            <hr />

            <ObservedMergeControl
                canvasInstance={canvasInstance}
                dynamicIconProps={dynamicMergeIconProps}
                disabled={controlsDisabled}
            />
            <ObservedGroupControl
                canvasInstance={canvasInstance}
                dynamicIconProps={dynamicGroupIconProps}
                disabled={controlsDisabled}
            />
            <ObservedSplitControl
                canvasInstance={canvasInstance}
                dynamicIconProps={dynamicTrackIconProps}
                disabled={controlsDisabled}
            />
        </Layout.Sider>
    );
}
