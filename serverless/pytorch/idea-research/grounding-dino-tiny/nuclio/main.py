import base64
import io
import json
import os
from PIL import Image
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

BDD100K_CLASSES = [
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
]

PROMPT_TO_BDD = {
    "pedestrian": "pedestrian",
    "person": "pedestrian",
    "rider": "rider",
    "bicyclist": "rider",
    "motorcyclist": "rider",
    "car": "car",
    "truck": "truck",
    "pickup truck": "truck",
    "van": "truck",
    "bus": "bus",
    "train": "train",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "traffic light": "traffic light",
    "traffic sign": "traffic sign",
    "stop sign": "traffic sign",
}


class GroundingDinoModelHandler:
    def __init__(self):
        self.model_id = os.environ.get("MODEL_ID", "IDEA-Research/grounding-dino-tiny")
        self.device = "cuda" if torch.cuda.is_available() and os.environ.get("TORCH_DEVICE", "cuda") == "cuda" else "cpu"
        if self.device == "cuda":
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if total_vram_gb > 4.5:
                torch.cuda.set_per_process_memory_fraction(min(1.0, 4.5 / total_vram_gb), device=0)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        self.default_prompt = ". ".join(BDD100K_CLASSES) + "."

    def infer(self, image: Image.Image, threshold: float = 0.25):
        image_rgb = image.convert("RGB")
        w, h = image_rgb.size
        inputs = self.processor(images=image_rgb, text=self.default_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=min(threshold, 0.20),
            target_sizes=[(h, w)],
        )[0]
        raw_labels = results.get("text_labels", results.get("labels", []))
        detections = []
        for box, score, raw_lbl in zip(results["boxes"], results["scores"], raw_labels):
            clean = str(raw_lbl).strip().lower()
            mapped = PROMPT_TO_BDD.get(clean)
            if mapped is None:
                for key, target in PROMPT_TO_BDD.items():
                    if key in clean:
                        mapped = target
                        break
            if mapped is None:
                continue
            x1, y1, x2, y2 = [round(float(c), 2) for c in box.tolist()]
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({
                "confidence": str(round(float(score.item()), 4)),
                "label": mapped,
                "points": [x1, y1, x2, y2],
                "type": "rectangle",
            })
        return detections


def init_context(context):
    context.logger.info("Initializing Grounding DINO Tiny context...")
    context.user_data.model_handler = GroundingDinoModelHandler()
    context.logger.info("Grounding DINO Tiny ready.")


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
        context.logger.error(f"Grounding DINO inference error: {exc}", exc_info=True)
        return context.Response(
            body=json.dumps({"error": str(exc)}),
            headers={},
            content_type="application/json",
            status_code=500,
        )
