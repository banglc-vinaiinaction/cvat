import base64
import io
import json
import os
import re
import urllib.request

try:
    from PIL import Image
except ImportError:
    Image = None

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

CANDIDATE_OLLAMA_URLS = [
    "http://host.docker.internal:11435",
    "http://172.18.0.1:11435",
    "http://172.17.0.1:11435",
    "http://host.docker.internal:11434",
    "http://172.18.0.1:11434",
    "http://172.17.0.1:11434",
    "http://localhost:11434",
]


def _resolve_ollama_url() -> str:
    env_url = os.environ.get("OLLAMA_URL")
    candidates = ([env_url.rstrip("/")] if env_url else []) + CANDIDATE_OLLAMA_URLS
    for url in candidates:
        try:
            req = urllib.request.Request(f"{url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    return url
        except Exception:
            continue
    return candidates[0]


def _clean_json(text: str):
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE | re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE | re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise


class Qwen35RefereeHandler:
    def __init__(self):
        self.model_id = os.environ.get("OLLAMA_MODEL", "qwen3.5:2b")
        self.ollama_url = _resolve_ollama_url()

    def _chat(self, prompt: str, img_b64: str) -> dict:
        payload = json.dumps({
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [img_b64],
                }
            ],
            "stream": False,
            "think": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "seed": 42,
                "num_ctx": 2048,
                "num_predict": 256,
            },
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.ollama_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=45.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            content = body.get("message", {}).get("content", "{}")
            return _clean_json(content)

    def infer_detector(self, image_b64: str, threshold: float = 0.25):
        w, h = 1280, 720
        if Image is not None:
            try:
                image_rgb = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
                w, h = image_rgb.size
                buf = io.BytesIO()
                image_rgb.save(buf, format="JPEG", quality=85)
                image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception:
                pass

        prompt = (
            "You are a BDD100K Autonomous Driving Object Detector and Semantic Referee. "
            f"Image size is {w}x{h}. Detect visible objects from classes: {BDD100K_CLASSES}. "
            'Return JSON: {"detections": [{"label": "<class>", "confidence": 0.9, "points": [x1, y1, x2, y2]}]}'
        )
        try:
            parsed = self._chat(prompt, image_b64)
            raw_list = parsed.get("detections", []) if isinstance(parsed, dict) else parsed
            detections = []
            for item in raw_list:
                lbl = str(item.get("label", "")).strip().lower()
                if lbl not in BDD100K_CLASSES:
                    continue
                conf = float(item.get("confidence", 0.85))
                if conf < threshold:
                    continue
                pts = item.get("points", item.get("bbox", [0, 0, w, h]))
                if len(pts) == 4:
                    x1, y1, x2, y2 = [round(float(v), 2) for v in pts]
                    detections.append({
                        "confidence": str(round(conf, 4)),
                        "label": lbl,
                        "points": [x1, y1, x2, y2],
                        "type": "rectangle",
                    })
            return detections
        except Exception:
            return []


def init_context(context):
    context.logger.info("Initializing Qwen3.5-2B VLM Referee context...")
    context.user_data.model_handler = Qwen35RefereeHandler()
    context.logger.info(
        f"Qwen3.5-2B VLM Referee ready (model={context.user_data.model_handler.model_id}, "
        f"ollama_url={context.user_data.model_handler.ollama_url})."
    )


def handler(context, event):
    try:
        if isinstance(event.body, (bytes, bytearray)):
            data = json.loads(event.body.decode("utf-8"))
        elif isinstance(event.body, str):
            data = json.loads(event.body)
        else:
            data = event.body or {}

        if not hasattr(context.user_data, "model_handler"):
            context.user_data.model_handler = Qwen35RefereeHandler()

        image_b64 = data.get("image") or data.get("crop_base64", "")
        threshold = float(data.get("threshold", 0.25))
        results = context.user_data.model_handler.infer_detector(image_b64=image_b64, threshold=threshold)
        return context.Response(
            body=json.dumps(results),
            headers={},
            content_type="application/json",
            status_code=200,
        )
    except Exception as exc:
        context.logger.error(f"Qwen3.5-2B inference error: {exc}", exc_info=True)
        return context.Response(
            body=json.dumps({"error": str(exc)}),
            headers={},
            content_type="application/json",
            status_code=500,
        )
