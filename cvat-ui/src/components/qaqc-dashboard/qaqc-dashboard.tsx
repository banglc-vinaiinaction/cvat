// Copyright (C) CVAT.ai Corporation
// SPDX-License-Identifier: MIT

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useHistory } from 'react-router';
import Axios, { AxiosError } from 'axios';
import { ReloadOutlined } from '@ant-design/icons';
import Alert from 'antd/lib/alert';
import Button from 'antd/lib/button';
import Card from 'antd/lib/card';
import { Col, Row } from 'antd/lib/grid';
import notification from 'antd/lib/notification';
import Select from 'antd/lib/select';
import Space from 'antd/lib/space';
import Statistic from 'antd/lib/statistic';
import Table from 'antd/lib/table';
import Tag from 'antd/lib/tag';
import Title from 'antd/lib/typography/Title';
import Text from 'antd/lib/typography/Text';

type RiskStatus = 'pending' | 'in_review' | 'confirmed' | 'rejected' | 'resolved';
type RiskType = 'drop' | 'flip' | 'shift' | 'spawn';

interface DashboardRisk {
    id: number;
    task_id: number;
    task_name: string;
    job_id: number;
    frame: number;
    type: RiskType;
    status: RiskStatus;
    score: number;
    iou: number | null;
    human_label: string | null;
    referee_label: string | null;
    human_bbox: number[] | null;
    referee_bbox: number[] | null;
    updated_date: string;
}

interface DashboardResponse {
    total: number;
    status_counts: Partial<Record<RiskStatus, number>>;
    type_counts: Partial<Record<RiskType, number>>;
    risks: DashboardRisk[];
}

const statusLabels: Record<RiskStatus, string> = {
    pending: 'Pending', in_review: 'In review', confirmed: 'Confirmed',
    rejected: 'False alarm', resolved: 'Resolved',
};

function statusColor(status: RiskStatus): string {
    if (status === 'confirmed') return 'red';
    if (status === 'rejected') return 'green';
    if (status === 'resolved') return 'blue';
    if (status === 'in_review') return 'orange';
    return 'gold';
}

function riskColor(type: RiskType): string {
    if (type === 'drop') return 'red';
    if (type === 'flip') return 'orange';
    if (type === 'shift') return 'gold';
    return 'purple';
}

export default function QAQCDashboard(): JSX.Element {
    const history = useHistory();
    const [data, setData] = useState<DashboardResponse>({
        total: 0, status_counts: {}, type_counts: {}, risks: [],
    });
    const [loading, setLoading] = useState(false);
    const [statusFilter, setStatusFilter] = useState<string>('all');
    const [typeFilter, setTypeFilter] = useState<string>('all');

    const load = useCallback(async (): Promise<void> => {
        setLoading(true);
        try {
            const response = await Axios.get('/api/model-assisted-qc/dashboard');
            setData(response.data);
        } catch (error: unknown) {
            const detail = error instanceof AxiosError ? error.response?.data?.detail || error.message : String(error);
            notification.error({ message: 'Không thể tải lịch sử QA/QC', description: detail });
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => { load(); }, [load]);

    const filtered = useMemo(() => data.risks.filter((risk) => (
        (statusFilter === 'all' || risk.status === statusFilter) &&
        (typeFilter === 'all' || risk.type === typeFilter)
    )), [data.risks, statusFilter, typeFilter]);

    const confirmed = data.status_counts.confirmed || 0;
    const falseAlarms = data.status_counts.rejected || 0;
    const reviewed = confirmed + falseAlarms + (data.status_counts.resolved || 0);

    const locateRisk = (risk: DashboardRisk): void => {
        sessionStorage.setItem('cvat.qaqc.focus-risk', JSON.stringify(risk));
        history.push(`/tasks/${risk.task_id}/jobs/${risk.job_id}?frame=${risk.frame}&qaqcRisk=${risk.id}`);
    };

    return (
        <div style={{ padding: '32px 48px', maxWidth: 1500, margin: '0 auto', width: '100%' }}>
            <Row justify='space-between' align='middle' style={{ marginBottom: 20 }}>
                <Col>
                    <Title level={2} style={{ marginBottom: 2 }}>QA/QC Dashboard</Title>
                    <Text type='secondary'>Lịch sử object-risk từ model trọng tài và quyết định của reviewer</Text>
                </Col>
                <Col><Button icon={<ReloadOutlined />} loading={loading} onClick={load}>Làm mới</Button></Col>
            </Row>

            <Row gutter={[16, 16]} style={{ marginBottom: 20 }}>
                <Col xs={12} md={6}><Card><Statistic title='Tổng risk' value={data.total} /></Card></Col>
                <Col xs={12} md={6}><Card><Statistic title='Đang chờ' value={data.status_counts.pending || 0} valueStyle={{ color: '#d48806' }} /></Card></Col>
                <Col xs={12} md={6}><Card><Statistic title='Đã review' value={reviewed} valueStyle={{ color: '#1769d2' }} /></Card></Col>
                <Col xs={12} md={6}><Card><Statistic title='Lỗi thật / False alarm' value={`${confirmed} / ${falseAlarms}`} /></Card></Col>
            </Row>

            <Alert
                style={{ marginBottom: 18 }} type='info' showIcon
                message={`DROP ${data.type_counts.drop || 0} · FLIP ${data.type_counts.flip || 0} · SHIFT ${data.type_counts.shift || 0}`}
                description='Pending là risk còn trong hàng đợi. Confirmed là lỗi annotation thật. False alarm là cảnh báo model sai.'
            />

            <Card>
                <Space wrap style={{ marginBottom: 16 }}>
                    <Text strong>Trạng thái</Text>
                    <Select value={statusFilter} onChange={setStatusFilter} style={{ width: 150 }}>
                        <Select.Option value='all'>Tất cả</Select.Option>
                        <Select.Option value='pending'>Pending</Select.Option>
                        <Select.Option value='confirmed'>Confirmed</Select.Option>
                        <Select.Option value='rejected'>False alarm</Select.Option>
                        <Select.Option value='resolved'>Resolved</Select.Option>
                    </Select>
                    <Text strong>Loại risk</Text>
                    <Select value={typeFilter} onChange={setTypeFilter} style={{ width: 135 }}>
                        <Select.Option value='all'>Tất cả</Select.Option>
                        <Select.Option value='drop'>DROP</Select.Option>
                        <Select.Option value='flip'>FLIP</Select.Option>
                        <Select.Option value='shift'>SHIFT</Select.Option>
                        <Select.Option value='spawn'>SPAWN</Select.Option>
                    </Select>
                    <Text type='secondary'>{filtered.length} bản ghi</Text>
                </Space>
                <Table<DashboardRisk>
                    rowKey='id' loading={loading} dataSource={filtered} size='middle'
                    pagination={{ defaultPageSize: 20, showSizeChanger: true, pageSizeOptions: ['20', '50', '100'] }}
                    columns={[
                        { title: 'Task', dataIndex: 'task_name', ellipsis: true },
                        { title: 'Job', dataIndex: 'job_id', width: 75 },
                        {
                            title: 'Frame', dataIndex: 'frame', width: 85,
                            render: (frame: number, risk) => (
                                <Button type='link' onClick={(): void => locateRisk(risk)}>{frame}</Button>
                            ),
                        },
                        { title: 'Risk', dataIndex: 'type', width: 90, render: (type: RiskType) => <Tag color={riskColor(type)}>{type.toUpperCase()}</Tag> },
                        { title: 'Score', dataIndex: 'score', width: 85, sorter: (a, b) => a.score - b.score, render: (score: number) => score.toFixed(3) },
                        { title: 'Human', dataIndex: 'human_label', width: 125, render: (value: string | null) => value || '—' },
                        { title: 'Referee', dataIndex: 'referee_label', width: 125, render: (value: string | null) => value || '—' },
                        { title: 'IoU', dataIndex: 'iou', width: 75, render: (value: number | null) => value === null ? '—' : value.toFixed(3) },
                        { title: 'Trạng thái', dataIndex: 'status', width: 120, render: (status: RiskStatus) => <Tag color={statusColor(status)}>{statusLabels[status]}</Tag> },
                        { title: 'Cập nhật', dataIndex: 'updated_date', width: 175, render: (value: string) => new Date(value).toLocaleString('vi-VN') },
                        { title: 'Vị trí', width: 85, render: (_value, risk) => <Button size='small' onClick={(): void => locateRisk(risk)}>Chỉ lỗi</Button> },
                    ]}
                />
            </Card>
        </div>
    );
}
