// Copyright (C) 2021-2022 Intel Corporation
// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useState } from 'react';
import Layout from 'antd/lib/layout';
import { useSelector } from 'react-redux';

import { DimensionType } from 'cvat-core-wrapper';
import ControlsSideBarContainer from 'containers/annotation-page/standard3D-workspace/controls-side-bar/controls-side-bar';
import ObjectSideBarComponent from 'components/annotation-page/standard-workspace/objects-side-bar/objects-side-bar';
import ObjectsListContainer from 'containers/annotation-page/standard-workspace/objects-side-bar/objects-list';
import CanvasContextMenuContainer from 'containers/annotation-page/canvas/canvas-context-menu';
import CanvasLayout from 'components/annotation-page/canvas/grid-layout/canvas-layout';
import CanvasPointContextMenuComponent from 'components/annotation-page/canvas/views/canvas2d/canvas-point-context-menu';
import RemoveConfirmComponent from 'components/annotation-page/standard-workspace/remove-confirm';
import PropagateConfirmComponent from 'components/annotation-page/standard-workspace/propagate-confirm';
import { CombinedState } from 'reducers';
import StandardizeCuboidsModal from './standardize-cuboids-modal';

export default function StandardWorkspace3DComponent(): JSX.Element {
    const [standardizeModalVisible, setStandardizeModalVisible] = useState<boolean>(false);
    const labels = useSelector((state: CombinedState) => state.annotation.job.labels);
    const activatedStateID = useSelector((state: CombinedState) => state.annotation.annotations.activatedStateID);
    const states = useSelector((state: CombinedState) => state.annotation.annotations.states);

    const activeState = typeof activatedStateID === 'number' ?
        states.find((s) => s.clientID === activatedStateID) : null;

    return (
        <Layout hasSider className='cvat-standard-workspace'>
            <ControlsSideBarContainer onOpenStandardizeModal={() => setStandardizeModalVisible(true)} />
            <CanvasLayout type={DimensionType.DIMENSION_3D} />
            <ObjectSideBarComponent objectsList={<ObjectsListContainer />} />
            <PropagateConfirmComponent />
            <CanvasContextMenuContainer />
            <CanvasPointContextMenuComponent />
            <RemoveConfirmComponent />
            <StandardizeCuboidsModal
                visible={standardizeModalVisible}
                onClose={() => setStandardizeModalVisible(false)}
                applicableLabels={labels}
                activeLabelId={activeState?.label?.id || null}
            />
        </Layout>
    );
}
