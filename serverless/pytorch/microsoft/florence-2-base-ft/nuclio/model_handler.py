# Copyright (C) CVAT.ai Corporation & Microsoft Corporation
#
# SPDX-License-Identifier: MIT

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

logger = logging.getLogger(__name__)


class Florence2ModelHandler:
    """
    Handler for Microsoft Florence-2-base-ft vision-language foundation model.
    Supports Object Detection (<OD>), Dense Region Captioning (<DENSE_REGION_CAPTION>),
    Phrase Grounding (<CAPTION_TO_PHRASE_GROUNDING>), and Open-Vocabulary Detection.
    """

    def __init__(self, model_id_or_path: Optional[str] = None, device: Optional[str] = None):
        # Determine device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Model source (local path or HuggingFace repo ID)
        self.model_path = (
            model_id_or_path
            or os.environ.get("MODEL_PATH")
            or os.environ.get("MODEL_ID")
            or "microsoft/Florence-2-base-ft"
        )

        logger.info(f"Loading Florence-2 model from '{self.model_path}' on device '{self.device}'...")

        # Set torch dtype: fp16 on CUDA for performance and lower VRAM; fp32 on CPU
        if self.device == "cuda":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

        # Florence-2 requires trust_remote_code=True because model definition is on HuggingFace Hub
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        ).to(self.device)

        self.model.eval()
        logger.info(f"Florence-2-base-ft initialized successfully on {self.device} ({self.torch_dtype}).")

    def infer(
        self,
        image: Image.Image,
        threshold: float = 0.5,
        task_prompt: str = "<OD>",
        text_input: Optional[str] = None,
        allowed_labels: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a PIL image and return CVAT-compatible annotations.

        Args:
            image: PIL.Image in RGB mode.
            threshold: Confidence threshold for predictions (0.0 to 1.0).
            task_prompt: '<OD>' for general detection, or '<CAPTION_TO_PHRASE_GROUNDING>' for text queries.
            text_input: Optional text description / comma-separated class queries for phrase grounding.
            allowed_labels: Optional filter for output labels.

        Returns:
            List of CVAT annotation dicts:
            [
                {
                    "confidence": "0.95",
                    "label": "car",
                    "points": [xtl, ytl, xbr, ybr],
                    "type": "rectangle"
                },
                ...
            ]
        """
        # Ensure image is in RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        w, h = image.size

        # Build full prompt
        if text_input:
            full_prompt = f"{task_prompt}{text_input}"
        else:
            full_prompt = task_prompt

        # Tokenize / preprocess
        inputs = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device, self.torch_dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(w, h)
        )

        results: List[Dict[str, Any]] = []

        # Extract bboxes and labels from parsed output
        # Florence-2 returns {task_prompt: {'bboxes': [...], 'labels': [...]}}
        task_result = parsed_answer.get(task_prompt, {})
        bboxes = task_result.get("bboxes", [])
        labels = task_result.get("labels", [])

        # If allowed_labels set, normalize for matching
        allowed_set = set(l.lower().strip() for l in allowed_labels) if allowed_labels else None

        for idx, (bbox, label) in enumerate(zip(bboxes, labels)):
            raw_label = str(label).strip()
            norm_label = raw_label.lower()

            if allowed_set and norm_label not in allowed_set:
                # Optionally check if subphrase matches
                matched = False
                for allowed in allowed_set:
                    if allowed in norm_label or norm_label in allowed:
                        raw_label = allowed
                        matched = True
                        break
                if not matched:
                    continue

            # Bounding box is [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            xtl = max(0.0, min(float(x1), float(x2)))
            ytl = max(0.0, min(float(y1), float(y2)))
            xbr = min(float(w), max(float(x1), float(x2)))
            ybr = min(float(h), max(float(y1), float(y2)))

            # Skip degenerate boxes
            if xbr <= xtl or ybr <= ytl:
                continue

            # Florence-2 does not output class logit confidence out of the box, default to 1.0
            confidence = 1.0
            if confidence < threshold:
                continue

            results.append({
                "confidence": str(round(confidence, 2)),
                "label": raw_label,
                "points": [round(xtl, 2), round(ytl, 2), round(xbr, 2), round(ybr, 2)],
                "type": "rectangle",
            })

        return results
