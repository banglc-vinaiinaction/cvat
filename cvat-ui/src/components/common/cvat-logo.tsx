// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React from 'react';

function CVATLogo(): JSX.Element {
    return (
        <div className='cvat-logo-icon'>
            <span className='cvat-logo-text'>CVAT-QA/QC Tool</span>
        </div>
    );
}

export default React.memo(CVATLogo);
