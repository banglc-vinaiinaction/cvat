#!/usr/bin/env python3
"""
test_florence2.py - Standalone test script for Florence-2-base-ft CVAT model handler.
Verifies loading, forward pass, and CVAT-compatible annotation generation.
"""

import argparse
import json
import os
import sys
from PIL import Image, ImageDraw

def create_synthetic_test_image(path: str = "sample_test_image.jpg") -> str:
    img = Image.new("RGB", (640, 480), color=(240, 240, 245))
    draw = ImageDraw.Draw(img)

    # Draw dummy road & car
    draw.rectangle([50, 300, 590, 450], fill=(80, 80, 80)) # Road
    draw.rectangle([150, 250, 350, 380], fill=(200, 30, 30)) # Red car
    draw.ellipse([420, 260, 470, 390], fill=(30, 100, 220)) # Person / pedestrian

    img.save(path)
    print(f"Created synthetic test image: {path}")
    return path

def main():
    parser = argparse.ArgumentParser(description="Test Florence-2 model handler.")
    parser.add_argument("--image", type=str, default=None, help="Path to input test image")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (auto if None)")
    parser.add_argument("--prompt", type=str, default="<OD>", help="Task prompt: <OD>, <CAPTION_TO_PHRASE_GROUNDING>, <DENSE_REGION_CAPTION>")
    parser.add_argument("--text", type=str, default=None, help="Text input for phrase grounding (e.g. 'car. pedestrian.')")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()

    # Import handler
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from model_handler import Florence2ModelHandler

    # Resolve image
    img_path = args.image
    if not img_path or not os.path.exists(img_path):
        img_path = create_synthetic_test_image()

    image = Image.open(img_path)
    print(f"Loaded image: {img_path} (size: {image.size})")

    # Initialize handler
    print("Initializing Florence2ModelHandler...")
    handler = Florence2ModelHandler(device=args.device)

    # Run inference
    print(f"Running inference with task='{args.prompt}', text_input='{args.text}'...")
    annotations = handler.infer(
        image=image,
        threshold=args.threshold,
        task_prompt=args.prompt,
        text_input=args.text
    )

    print("\n--- CVAT Formatted Annotations Output ---")
    print(json.dumps(annotations, indent=2))
    print(f"\nTotal detections: {len(annotations)}")

if __name__ == "__main__":
    main()
