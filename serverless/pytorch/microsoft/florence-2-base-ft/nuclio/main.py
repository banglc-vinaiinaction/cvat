# Copyright (C) CVAT.ai Corporation & Microsoft Corporation
#
# SPDX-License-Identifier: MIT

import base64
import io
import json
import logging
import os
from PIL import Image
import yaml
from model_handler import Florence2ModelHandler


def init_context(context):
    context.logger.info("Initializing Florence-2 Base FT context... 0%")

    # Parse labels specification from function.yaml if present
    labels = []
    function_yaml_path = "/opt/nuclio/function.yaml"
    if os.path.exists(function_yaml_path):
        try:
            with open(function_yaml_path, "r") as f:
                function_config = yaml.safe_load(f)
            spec_raw = function_config.get("metadata", {}).get("annotations", {}).get("spec", "[]")
            if isinstance(spec_raw, str):
                spec_json = json.loads(spec_raw)
            else:
                spec_json = spec_raw
            labels = [item["name"] for item in spec_json if "name" in item]
            context.logger.info(f"Loaded {len(labels)} label categories from function.yaml spec.")
        except Exception as e:
            context.logger.warning(f"Could not parse labels from function.yaml: {e}")

    # Initialize model handler
    model_handler = Florence2ModelHandler()
    context.user_data.model_handler = model_handler
    context.user_data.labels = labels

    context.logger.info("Initializing Florence-2 Base FT context... 100% Ready.")


def handler(context, event):
    context.logger.info("Handling Florence-2 inference request...")
    try:
        # Event body can be dict, bytes or string
        if isinstance(event.body, (bytes, bytearray)):
            data = json.loads(event.body.decode("utf-8"))
        elif isinstance(event.body, str):
            data = json.loads(event.body)
        else:
            data = event.body

        # Decode base64 image
        image_bytes = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(image_bytes))

        # Parameters
        threshold = float(data.get("threshold", 0.5))
        task_prompt = data.get("task_prompt", "<OD>")
        text_input = data.get("text_input", None)
        prompt = data.get("prompt", None)

        if prompt and not text_input:
            text_input = prompt

        # If user supplies custom task labels in request or task mapping
        custom_labels = data.get("labels", None)
        allowed_labels = custom_labels or context.user_data.labels or None

        # Execute inference
        results = context.user_data.model_handler.infer(
            image=image,
            threshold=threshold,
            task_prompt=task_prompt,
            text_input=text_input,
            allowed_labels=allowed_labels,
        )

        context.logger.info(f"Inference completed. Detected {len(results)} objects.")

        return context.Response(
            body=json.dumps(results),
            headers={},
            content_type="application/json",
            status_code=200,
        )

    except Exception as e:
        context.logger.error(f"Error during Florence-2 inference: {e}", exc_info=True)
        return context.Response(
            body=json.dumps({"error": str(e)}),
            headers={},
            content_type="application/json",
            status_code=500,
        )
