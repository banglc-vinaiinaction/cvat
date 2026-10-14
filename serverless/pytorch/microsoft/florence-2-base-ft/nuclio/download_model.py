#!/usr/bin/env python3
"""
download_model.py - Helper script to pre-download microsoft/Florence-2-base-ft weights.
Can download directly to a specified directory or Hugging Face cache.
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Download Florence-2-base-ft model weights.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="microsoft/Florence-2-base-ft",
        help="HuggingFace model ID (default: microsoft/Florence-2-base-ft)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional local output directory to save snapshot weights"
    )
    args = parser.parse_args()

    print(f"Downloading model '{args.model_id}'...")

    try:
        from huggingface_hub import snapshot_download
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            path = snapshot_download(
                repo_id=args.model_id,
                local_dir=args.output_dir,
                local_dir_use_symlinks=False
            )
            print(f"Model successfully saved to: {path}")
            return
    except ImportError:
        pass

    # Fallback to transformers AutoModel
    from transformers import AutoProcessor, AutoModelForCausalLM

    print("Using transformers to download and verify model artifacts...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        processor.save_pretrained(args.output_dir)
        model.save_pretrained(args.output_dir)
        print(f"Model and processor successfully saved to: {args.output_dir}")
    else:
        print("Model downloaded and cached successfully in default Hugging Face cache.")

if __name__ == "__main__":
    main()
