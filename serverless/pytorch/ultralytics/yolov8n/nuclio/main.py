import base64
import io
import json
import os
from PIL import Image
import torch
from ultralytics import YOLO

COCO_TO_BDD = {
    "person": "pedestrian",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "train": "train",
    "truck": "truck",
    "traffic light": "traffic light",
    "stop sign": "traffic sign",
}


class YOLOv8nPrelabelHandler:
    def __init__(self):
        weights = os.environ.get("MODEL_WEIGHTS", "yolov8n.pt")
        self.device = "0" if torch.cuda.is_available() and os.environ.get("TORCH_DEVICE", "cuda") == "cuda" else "cpu"
        self.model = YOLO(weights)

    def infer(self, image: Image.Image, threshold: float = 0.25):
        results = self.model.predict(source=image.convert("RGB"), conf=threshold, device=self.device, verbose=False)[0]
        names = results.names
        detections = []
        if results.boxes is None:
            return detections
        for box in results.boxes:
            cls_id = int(box.cls.item())
            raw_name = str(names.get(cls_id, cls_id)).lower().strip()
            mapped = COCO_TO_BDD.get(raw_name, raw_name if raw_name in COCO_TO_BDD.values() else None)
            if mapped is None:
                continue
            conf = round(float(box.conf.item()), 4)
            x1, y1, x2, y2 = [round(float(v), 2) for v in box.xyxy[0].tolist()]
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({
                "confidence": str(conf),
                "label": mapped,
                "points": [x1, y1, x2, y2],
                "type": "rectangle",
            })
        return detections


def init_context(context):
    context.logger.info("Initializing YOLOv8n Pre-labeler context...")
    context.user_data.model_handler = YOLOv8nPrelabelHandler()
    context.logger.info("YOLOv8n Pre-labeler ready.")


def handler(context, event):
    try:
        if isinstance(event.body, (bytes, bytearray)):
            data = json.loads(event.body.decode("utf-8"))
        elif isinstance(event.body, str):
            data = json.loads(event.body)
        else:
            data = event.body
        image = Image.open(io.BytesIO(base64.b64decode(data["image"])))
        threshold = float(data.get("threshold", 0.25))
        results = context.user_data.model_handler.infer(image=image, threshold=threshold)
        return context.Response(
            body=json.dumps(results),
            headers={},
            content_type="application/json",
            status_code=200,
        )
    except Exception as exc:
        context.logger.error(f"YOLOv8n inference error: {exc}", exc_info=True)
        return context.Response(
            body=json.dumps({"error": str(exc)}),
            headers={},
            content_type="application/json",
            status_code=500,
        )
