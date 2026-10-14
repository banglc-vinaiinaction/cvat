// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import Modal from 'antd/lib/modal';
import Select from 'antd/lib/select';
import InputNumber from 'antd/lib/input-number';
import Radio, { RadioChangeEvent } from 'antd/lib/radio';
import Checkbox, { CheckboxChangeEvent } from 'antd/lib/checkbox';
import notification from 'antd/lib/notification';
import { Row, Col } from 'antd/lib/grid';
import Text from 'antd/lib/typography/Text';
import Button from 'antd/lib/button';
import Alert from 'antd/lib/alert';
import { serverProxy, Label } from 'cvat-core-wrapper';
import { getClassDimensionPrior } from 'cvat-canvas3d/src/typescript/consts';
import { fetchAnnotationsAsync } from 'actions/annotation-actions';
import { CombinedState } from 'reducers';

interface Props {
    visible: boolean;
    onClose: () => void;
    applicableLabels: Label[];
    activeLabelId?: number | null;
    activeTrack?: { serverID: number | null; clientID: number } | null;
}

export default function StandardizeCuboidsModal(props: Props): JSX.Element {
    const {
        visible, onClose, applicableLabels, activeLabelId, activeTrack,
    } = props;
    const dispatch = useDispatch();

    const jobInstance = useSelector((state: CombinedState) => state.annotation.job.instance);
    const currentFrame = useSelector((state: CombinedState) => state.annotation.player.frame.number);

    const defaultLabel = (activeLabelId && applicableLabels.find((l) => l.id === activeLabelId)) ||
        applicableLabels[0] || null;

    const [selectedLabelId, setSelectedLabelId] = useState<number | null>(defaultLabel?.id || null);
    const [dx, setDx] = useState<number>(4.8);
    const [dy, setDy] = useState<number>(1.9);
    const [dz, setDz] = useState<number>(1.6);
    const [scope, setScope] = useState<'selected_track' | 'job' | 'current_frame' | 'range' | 'task'>('job');
    const [frameFrom, setFrameFrom] = useState<number>(jobInstance?.startFrame || 0);
    const [frameTo, setFrameTo] = useState<number>(jobInstance?.stopFrame || 0);
    const [reground, setReground] = useState<boolean>(true);
    const [loading, setLoading] = useState<boolean>(false);

    // Sync default dimensions whenever selected label changes
    useEffect(() => {
        if (!selectedLabelId) return;
        const labelObj = applicableLabels.find((l) => l.id === selectedLabelId);
        if (labelObj) {
            const [pDx, pDy, pDz] = getClassDimensionPrior(labelObj);
            setDx(pDx);
            setDy(pDy);
            setDz(pDz);
        }
    }, [selectedLabelId, applicableLabels]);

    // Update active label and default scope when modal opens
    useEffect(() => {
        if (visible) {
            if (activeTrack) {
                setScope('selected_track');
            } else {
                setScope('job');
            }
            if (defaultLabel) {
                setSelectedLabelId(defaultLabel.id);
                const [pDx, pDy, pDz] = getClassDimensionPrior(defaultLabel);
                setDx(pDx);
                setDy(pDy);
                setDz(pDz);
            }
            setFrameFrom(jobInstance?.startFrame || 0);
            setFrameTo(jobInstance?.stopFrame || 0);
        }
    }, [visible]);

    const executeStandardize = async (): Promise<void> => {
        if (!jobInstance || !selectedLabelId) return;

        setLoading(true);
        try {
            if (jobInstance.annotations?.hasUnsavedChanges?.()) {
                await jobInstance.annotations.save();
            }

            const isTaskScope = scope === 'task' && jobInstance.taskId;
            const instanceType = isTaskScope ? 'task' : 'job';
            const instanceId = isTaskScope ? jobInstance.taskId : jobInstance.id;

            let fFrom: number | null = null;
            let fTo: number | null = null;
            if (scope === 'current_frame') {
                fFrom = currentFrame;
                fTo = currentFrame;
            } else if (scope === 'range') {
                fFrom = frameFrom;
                fTo = frameTo;
            }

            const labelObj = applicableLabels.find((l) => l.id === selectedLabelId);
            const payload: any = {
                label_id: selectedLabelId,
                dimensions: [dx, dy, dz],
                reground,
                frame_from: fFrom,
                frame_to: fTo,
            };

            if (scope === 'selected_track') {
                let trackServerId = activeTrack?.serverID ?? null;
                if (!trackServerId && activeTrack?.clientID !== undefined) {
                    const currentStates = await jobInstance.annotations.get(currentFrame);
                    const matched = currentStates?.find((s: any) => s.clientID === activeTrack.clientID);
                    if (matched?.serverID) {
                        trackServerId = matched.serverID;
                    }
                }
                if (trackServerId) {
                    payload.track_id = trackServerId;
                }
            }

            const result = await (serverProxy.annotations as any).standardizeCuboids(
                instanceType,
                instanceId,
                payload,
            );

            const targetName = labelObj?.name || selectedLabelId;
            const scopeDesc = scope === 'selected_track' ?
                `Track #${activeTrack?.clientID}` : `label "${targetName}"`;

            notification.success({
                message: 'Standardization Complete',
                description: `Standardized ${result.total} cuboid(s) (${result.updated_shapes} shapes, ` +
                    `${result.updated_tracks} track frames) for ${scopeDesc} ` +
                    `to ${dx}m × ${dy}m × ${dz}m.`,
                duration: 4.5,
            });

            // Clear cvat-core in-memory session cache and reload fresh annotations from server
            await jobInstance.annotations.clear({ reload: true });
            dispatch(fetchAnnotationsAsync());
            onClose();
        } catch (error: any) {
            notification.error({
                message: 'Standardization Failed',
                description: error.message || error.toString(),
            });
        } finally {
            setLoading(false);
        }
    };

    const handleApply = (): void => {
        if (!jobInstance || !selectedLabelId) return;

        if (scope === 'selected_track') {
            if (!activeTrack?.serverID && !jobInstance.annotations?.hasUnsavedChanges?.()) {
                notification.error({
                    message: 'Unsaved Track',
                    description: `Track #${activeTrack?.clientID} is not saved to the server yet. ` +
                        'Please save annotations (Ctrl + S) first.',
                });
                return;
            }
        }

        executeStandardize();
    };

    return (
        <Modal
            title='Standardize Cuboid Dimensions (En-Masse)'
            visible={visible}
            onCancel={onClose}
            footer={[
                <Button key='cancel' onClick={onClose} disabled={loading}>
                    Cancel
                </Button>,
                <Button key='submit' type='primary' loading={loading} onClick={handleApply}>
                    Apply to Database
                </Button>,
            ]}
            width={520}
            destroyOnClose
        >
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                <div>
                    <Text strong>Target Label</Text>
                    <Select
                        style={{ width: '100%', marginTop: 6 }}
                        value={selectedLabelId}
                        onChange={(val: number) => setSelectedLabelId(val)}
                        placeholder='Select label'
                    >
                        {applicableLabels.map((lbl) => (
                            <Select.Option key={lbl.id} value={lbl.id}>
                                <span style={{ color: lbl.color }}>● </span>
                                {lbl.name}
                            </Select.Option>
                        ))}
                    </Select>
                </div>

                <div>
                    <Text strong>Quick Vehicle Class Presets</Text>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
                        {[
                            { key: 'compact', label: 'Compact Car', dx: 4.2, dy: 1.8, dz: 1.45 },
                            { key: 'sedan', label: 'Standard Sedan', dx: 4.65, dy: 1.85, dz: 1.50 },
                            { key: 'suv', label: 'Full-size SUV', dx: 4.95, dy: 2.0, dz: 1.75 },
                            { key: 'truck', label: 'Box Truck', dx: 8.5, dy: 2.5, dz: 3.20 },
                            { key: 'bus', label: 'City Bus', dx: 12.0, dy: 2.55, dz: 3.40 },
                            { key: 'moto', label: 'Motorcycle', dx: 2.1, dy: 0.85, dz: 1.25 },
                            { key: 'ped', label: 'Pedestrian', dx: 0.6, dy: 0.6, dz: 1.75 },
                        ].map((preset) => (
                            <Button
                                key={preset.key}
                                size='small'
                                onClick={() => {
                                    setDx(preset.dx);
                                    setDy(preset.dy);
                                    setDz(preset.dz);
                                }}
                            >
                                {`${preset.label} (${preset.dx}×${preset.dy}×${preset.dz}m)`}
                            </Button>
                        ))}
                    </div>
                </div>

                <div>
                    <Text strong>Target Dimensions (meters)</Text>
                    <Row gutter={12} style={{ marginTop: 6 }}>
                        <Col span={8}>
                            <Text type='secondary'>Length (dx)</Text>
                            <InputNumber
                                style={{ width: '100%', marginTop: 4 }}
                                min={0.1}
                                max={50}
                                step={0.05}
                                value={dx}
                                onChange={(val: number | null) => val && setDx(val)}
                            />
                        </Col>
                        <Col span={8}>
                            <Text type='secondary'>Width (dy)</Text>
                            <InputNumber
                                style={{ width: '100%', marginTop: 4 }}
                                min={0.1}
                                max={50}
                                step={0.05}
                                value={dy}
                                onChange={(val: number | null) => val && setDy(val)}
                            />
                        </Col>
                        <Col span={8}>
                            <Text type='secondary'>Height (dz)</Text>
                            <InputNumber
                                style={{ width: '100%', marginTop: 4 }}
                                min={0.1}
                                max={50}
                                step={0.05}
                                value={dz}
                                onChange={(val: number | null) => val && setDz(val)}
                            />
                        </Col>
                    </Row>
                </div>

                <div>
                    <Text strong>Scope</Text>
                    <Radio.Group
                        style={{
                            display: 'flex', flexDirection: 'column', gap: 8, marginTop: 6,
                        }}
                        value={scope}
                        onChange={(e: RadioChangeEvent) => setScope(e.target.value)}
                    >
                        {activeTrack && (
                            <Radio value='selected_track'>
                                <Text strong>Selected Track #{activeTrack.clientID}</Text>
                                {activeTrack.serverID ? (
                                    <Text type='secondary'>
                                        {` (Server ID: ${activeTrack.serverID}) `}
                                        {'— updates all frames of this track only'}
                                    </Text>
                                ) : (
                                    <Text type='danger'> (Unsaved — save annotations with Ctrl+S first)</Text>
                                )}
                            </Radio>
                        )}
                        <Radio value='job'>
                            {`Entire Job (${jobInstance ?
                                `Job #${jobInstance.id}, frames ${jobInstance.startFrame}–${jobInstance.stopFrame}` :
                                'All frames'})`}
                        </Radio>
                        <Radio value='current_frame'>
                            Current Frame (Frame #{currentFrame})
                        </Radio>
                        <Radio value='range'>
                            Frame Range
                        </Radio>
                        {jobInstance?.taskId && (
                            <Radio value='task'>
                                {`Entire Task (All jobs across Task #${jobInstance.taskId})`}
                            </Radio>
                        )}
                    </Radio.Group>
                    {scope === 'selected_track' && !activeTrack?.serverID && (
                        <Alert
                            style={{ marginTop: 8 }}
                            type='warning'
                            showIcon
                            message='Unsaved Track'
                            description={
                                'This track was created recently and has not been committed to the server. ' +
                                'Save annotations (Ctrl + S) first before running standardization.'
                            }
                        />
                    )}
                    {scope === 'range' && (
                        <Row gutter={8} style={{ marginTop: 8, marginLeft: 24 }}>
                            <Col span={10}>
                                <InputNumber
                                    style={{ width: '100%' }}
                                    min={jobInstance?.startFrame || 0}
                                    max={frameTo}
                                    value={frameFrom}
                                    onChange={(val: number | null) => val !== null && setFrameFrom(val)}
                                    placeholder='From'
                                />
                            </Col>
                            <Col span={2} style={{ textAlign: 'center', lineHeight: '32px' }}>to</Col>
                            <Col span={10}>
                                <InputNumber
                                    style={{ width: '100%' }}
                                    min={frameFrom}
                                    max={jobInstance?.stopFrame || 0}
                                    value={frameTo}
                                    onChange={(val: number | null) => val !== null && setFrameTo(val)}
                                    placeholder='To'
                                />
                            </Col>
                        </Row>
                    )}
                </div>

                <div>
                    <Checkbox
                        checked={reground}
                        onChange={(e: CheckboxChangeEvent) => setReground(e.target.checked)}
                    >
                        Preserve Road Contact (Keep bottom surface flush with ground)
                    </Checkbox>
                    <div>
                        <Text type='secondary' style={{ fontSize: 12 }}>
                            Automatically recalculates vertical center so bottom elevation ($z - dz/2$)
                            remains fixed to pavement when height changes.
                        </Text>
                    </div>
                </div>
            </div>
        </Modal>
    );
}
