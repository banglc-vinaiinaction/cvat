// Copyright (C) CVAT.ai Corporation
// SPDX-License-Identifier: MIT

import React, {
    useCallback, useEffect, useRef, useState,
} from 'react';
import Axios, { AxiosError } from 'axios';
import { AuditOutlined, LoadingOutlined } from '@ant-design/icons';
import Alert from 'antd/lib/alert';
import Button from 'antd/lib/button';
import Modal from 'antd/lib/modal';
import notification from 'antd/lib/notification';
import Space from 'antd/lib/space';
import Switch from 'antd/lib/switch';
import Table from 'antd/lib/table';
import Tag from 'antd/lib/tag';
import Typography from 'antd/lib/typography';

import { Job, ObjectState } from 'cvat-core-wrapper';
import { Canvas } from 'cvat-canvas-wrapper';
import { Canvas3d } from 'cvat-canvas3d-wrapper';

interface Risk {
    id: number;
    frame: number;
    type: 'drop' | 'flip' | 'shift' | 'spawn';
    status: 'pending' | 'in_review' | 'confirmed' | 'rejected' | 'resolved';
    score: number;
    iou: number | null;
    human_object_id: string | null;
    human_label: string | null;
    referee_label: string | null;
    human_bbox: number[] | null;
    referee_bbox: number[] | null;
}

interface QueueResponse {
    share: number;
    pending_total: number;
    visible_count: number;
    selected_frame_count?: number;
    total_frame_count?: number;
    risks: Risk[];
}

interface Props {
    jobInstance: Job;
    currentFrame: number;
    saving: boolean;
    canvasInstance: Canvas | Canvas3d;
    onSelectFrame(frame: number): void;
}

const FOCUS_RISK_KEY = 'cvat.qaqc.focus-risk';

function errorDescription(error: unknown): string {
    if (error instanceof AxiosError) {
        const detail = error.response?.data?.detail;
        return typeof detail === 'string' ? detail : error.message;
    }
    return error instanceof Error ? error.message : 'Unknown error';
}

function riskColor(type: Risk['type']): string {
    if (type === 'drop') return 'red';
    if (type === 'flip') return 'orange';
    return 'gold';
}

function ModelQCAssistant(props: Props): JSX.Element {
    const {
        jobInstance, currentFrame, saving, canvasInstance, onSelectFrame,
    } = props;
    const [open, setOpen] = useState(false);
    const [loading, setLoading] = useState(false);
    const [autoEvaluate, setAutoEvaluate] = useState(false);
    const [queue, setQueue] = useState<QueueResponse>({
        share: 0.15, pending_total: 0, visible_count: 0, risks: [],
    });
    const previousSaving = useRef(saving);
    const [focusedRisk, setFocusedRisk] = useState<Risk | null>(() => {
        try {
            const stored = sessionStorage.getItem(FOCUS_RISK_KEY);
            if (!stored) return null;
            sessionStorage.removeItem(FOCUS_RISK_KEY);
            const parsed = JSON.parse(stored);
            return parsed.job_id === jobInstance.id ? parsed as Risk : null;
        } catch (_error: unknown) {
            return null;
        }
    });
    const endpoint = `/api/model-assisted-qc/jobs/${jobInstance.id}`;

    const loadQueue = useCallback(async (): Promise<void> => {
        const response = await Axios.get(`${endpoint}/risks`, { params: { share: 0.15 } });
        setQueue(response.data);
    }, [endpoint]);

    const evaluateFrame = useCallback(async (silent = false): Promise<void> => {
        setLoading(true);
        try {
            const states = await jobInstance.annotations.get(currentFrame, false, []) as ObjectState[];
            const annotations = states.filter((state) => (
                state.frame === currentFrame &&
                state.shapeType === 'rectangle' &&
                !state.outside &&
                Array.isArray(state.points) &&
                state.points.length === 4
            )).map((state) => ({
                id: String(state.serverID ?? state.clientID),
                label: state.label.name,
                bbox: state.points as number[],
            }));
            await Axios.post(`${endpoint}/frames/${currentFrame}/evaluate`, {
                annotations,
                include_spawn: false,
            });
            await loadQueue();
            if (!silent) {
                notification.success({
                    message: `QC completed for frame ${currentFrame}`,
                    description: 'The top 15% risk queue has been refreshed.',
                });
            }
        } catch (error: unknown) {
            if (!silent || !(error instanceof AxiosError) || error.response?.status !== 409) {
                notification.error({
                    message: 'Could not evaluate this frame',
                    description: errorDescription(error),
                });
            }
        } finally {
            setLoading(false);
        }
    }, [currentFrame, endpoint, jobInstance, loadQueue]);

    useEffect(() => {
        if (previousSaving.current && !saving && autoEvaluate) {
            evaluateFrame(true);
        }
        previousSaving.current = saving;
    }, [saving, autoEvaluate, evaluateFrame]);

    useEffect(() => {
        if (!focusedRisk || focusedRisk.frame !== currentFrame || !(canvasInstance instanceof Canvas)) return undefined;
        const bbox = focusedRisk.type === 'drop' ? focusedRisk.referee_bbox :
            (focusedRisk.human_bbox || focusedRisk.referee_bbox);
        if (!bbox || bbox.length !== 4) return undefined;

        const timer = window.setTimeout(() => {
            canvasInstance.setupIssueRegions({
                [-1000000 - focusedRisk.id]: { hidden: false, points: bbox },
            });
            notification.info({
                message: `${focusedRisk.type.toUpperCase()} risk #${focusedRisk.id}`,
                description: 'Vùng kẻ sọc đỏ trên ảnh là vị trí lỗi cần kiểm tra.',
                duration: 3,
            });
        }, 150);
        return () => window.clearTimeout(timer);
    }, [canvasInstance, currentFrame, focusedRisk]);

    const locateRisk = (risk: Risk): void => {
        setFocusedRisk(risk);
        setOpen(false);
        onSelectFrame(risk.frame);
    };

    const updateStatus = async (risk: Risk, status: Risk['status']): Promise<void> => {
        await Axios.patch(`${endpoint}/risks/${risk.id}`, { status });
        await loadQueue();
    };

    return (
        <>
            <Button
                type='link'
                className='cvat-annotation-header-qc-button cvat-annotation-header-button'
                onClick={(): void => {
                    setOpen(true);
                    evaluateFrame();
                }}
            >
                {loading ? <LoadingOutlined /> : <AuditOutlined />}
                {`QA/QC${queue.visible_count ? ` (${queue.visible_count})` : ''}`}
            </Button>
            <Modal
                open={open}
                width={1050}
                title='QA/QC — top 15% frames theo điểm PD'
                footer={null}
                destroyOnClose={false}
                onCancel={(): void => setOpen(false)}
            >
                <Space direction='vertical' style={{ width: '100%' }} size='middle'>
                    <Alert
                        type='info'
                        showIcon
                        message={`${queue.selected_frame_count || 0}/${queue.total_frame_count || 0} frame được chọn · ${queue.visible_count} risk hiển thị`}
                        description='Mỗi box được chấm bằng PD; điểm frame là tổng các risk ≥ 0,5. Danh sách lấy top 15% frame, không phải top 15% risk.'
                    />
                    <Space wrap>
                        <Button loading={loading} type='primary' onClick={(): void => { evaluateFrame(); }}>
                            Kiểm tra lại frame {currentFrame}
                        </Button>
                        <Typography.Text>Tự kiểm tra sau khi Save</Typography.Text>
                        <Switch checked={autoEvaluate} onChange={setAutoEvaluate} />
                        <Button onClick={(): void => { loadQueue(); }}>Làm mới danh sách</Button>
                    </Space>
                    <Table<Risk>
                        rowKey='id'
                        size='small'
                        pagination={false}
                        dataSource={queue.risks}
                        columns={[
                            { title: 'Rank', render: (_value, _risk, index): number => index + 1, width: 65 },
                            {
                                title: 'Frame', dataIndex: 'frame', width: 85,
                                render: (frame: number, risk) => <Button type='link' onClick={(): void => locateRisk(risk)}>{frame}</Button>,
                            },
                            {
                                title: 'Risk', dataIndex: 'type', width: 90,
                                render: (type: Risk['type']) => <Tag color={riskColor(type)}>{type.toUpperCase()}</Tag>,
                            },
                            { title: 'Score', dataIndex: 'score', width: 85, render: (score: number) => score.toFixed(3) },
                            { title: 'Human', dataIndex: 'human_label', render: (value: string | null) => value || '—' },
                            { title: 'Referee', dataIndex: 'referee_label', render: (value: string | null) => value || '—' },
                            { title: 'IoU', dataIndex: 'iou', width: 75, render: (value: number | null) => value === null ? '—' : value.toFixed(3) },
                            {
                                title: 'Vị trí', width: 85,
                                render: (_value, risk) => <Button size='small' onClick={(): void => locateRisk(risk)}>Chỉ lỗi</Button>,
                            },
                            {
                                title: 'Review', width: 210,
                                render: (_value, risk) => (
                                    <Space>
                                        <Button size='small' type='primary' onClick={(): Promise<void> => updateStatus(risk, 'confirmed')}>Confirm</Button>
                                        <Button size='small' onClick={(): Promise<void> => updateStatus(risk, 'rejected')}>False alarm</Button>
                                    </Space>
                                ),
                            },
                        ]}
                    />
                </Space>
            </Modal>
        </>
    );
}

export default React.memo(ModelQCAssistant);
