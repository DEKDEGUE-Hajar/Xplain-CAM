from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import torch
import torchvision.models as models
from xplaincam.utils import load_image, sample_images_flat



# Main 
def main(args):
    from xplaincam.cam import (
        GradCAM, GradCAMpp, XGradCAM, GroupCAM, ScoreCAM, RISE, ISCAM, SSCAM, UnionCAM, FusionCAM
    )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load model dynamically
    if not hasattr(models, args.model):
        raise ValueError(f"Model '{args.model}' is not available in torchvision.models")
    model_cls = getattr(models, args.model)
    model = model_cls(pretrained=True).eval().to(device)

    # CAM registry
    cams_dict = {
        "GradCAM": {"cam": GradCAM, "args": {"target_layer": args.target_layer}},
        "GradCAM++": {"cam": GradCAMpp, "args": {"target_layer": args.target_layer}},
        "XGradCAM": {"cam": XGradCAM, "args": {"target_layer": args.target_layer}},
        "ScoreCAM": {"cam": ScoreCAM, "args": {"target_layer": args.target_layer}},
        "GroupCAM": {"cam": GroupCAM, "args": {"target_layer": args.target_layer}},
        "RISE": {"cam": RISE, "args": {"input_shape": (3, 224, 224)}},
        "ISCAM": {"cam": ISCAM, "args": {"target_layer": args.target_layer}},
        "UnionCAM": {"cam": UnionCAM, "args": {"target_layer": args.target_layer}},
        "FusionCAM": {"cam": FusionCAM, "args": {"target_layer": args.target_layer, "grad_cam": GradCAM, "region_cam": ScoreCAM}},
    }

    if args.methods is not None:
        cams_dict = {k: v for k, v in cams_dict.items() if k in args.methods}

    # Sample images
    image_paths = sample_images_flat(Path(args.data_path), args.num_images, args.seed)

    # CSV setup
    save_csv = bool(args.output_csv)
    if save_csv:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["CAM Method", "Average Latency (s)"])

    # Metrics accumulation
    latency_acc = {cam_name: [] for cam_name in cams_dict.keys()}

    # Evaluation loop
    for img_path in image_paths:
        input_tensor = load_image(img_path, args.image_size).to(device)
        input_tensor.requires_grad_(True)

        for cam_name, cam_info in cams_dict.items():
            cam_cls = cam_info["cam"]
            cam_args = cam_info["args"]

            start_time = time.time()
            with cam_cls(model, **cam_args) as cam_extractor:
                _ = cam_extractor(input_tensor)
            elapsed = time.time() - start_time
            latency_acc[cam_name].append(elapsed)

    # Display and optionally save
    print("\n Latency evaluation completed\n")
    print(f"{'CAM Method':<15} {'Average Latency (s)':<20}")
    print("-"*35)

    for cam_name, times in latency_acc.items():
        avg_latency = sum(times) / len(times)
        avg_latency_fmt = f"{avg_latency:.4f}"
        print(f"{cam_name:<15} {avg_latency_fmt:<20}")

        if save_csv:
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([cam_name, avg_latency_fmt])

    if save_csv:
        print(f"\n Average latency results saved to {csv_path}")

# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure CAM generation latency")

    parser.add_argument("data_path", type=str, help="Path to the ImageNet validation dataset")
    parser.add_argument("--model", type=str, default="vgg16", help="Torchvision model to use")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional CSV filename to save results")
    parser.add_argument("--num-images", type=int, default=50, help="Number of images to sample")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for image sampling")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--methods", nargs="+", default=None, help="Subset of CAM methods to evaluate")
    parser.add_argument("--target-layer", type=str, default=None, help="Target layer for CAM")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu or cuda)")

    args = parser.parse_args()
    main(args)
