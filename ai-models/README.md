### AI Models in CVAT

#### 1. Serverless Auto-Annotation Models (Nuclio Web UI)
In modern CVAT, AI models for web UI auto-annotation (Detectors, Interactors, Trackers) are deployed via **Nuclio Serverless functions** located under the [`../serverless/`](../serverless/) directory.

- Newly configured: **Florence-2 Base FT (Microsoft)** vision foundation model:
  [`../serverless/pytorch/microsoft/florence-2-base-ft/nuclio/`](../serverless/pytorch/microsoft/florence-2-base-ft/nuclio/)
  - GPU Configuration: [`function-gpu.yaml`](../serverless/pytorch/microsoft/florence-2-base-ft/nuclio/function-gpu.yaml)
  - CPU Configuration: [`function.yaml`](../serverless/pytorch/microsoft/florence-2-base-ft/nuclio/function.yaml)
  - Model Handler: [`model_handler.py`](../serverless/pytorch/microsoft/florence-2-base-ft/nuclio/model_handler.py)

#### 2. Standalone Auto-Annotation Functions (CVAT SDK & CLI)
The standalone python agents were moved to an independent repository:
- GitHub: <https://github.com/cvat-ai/cvat-models.git>
