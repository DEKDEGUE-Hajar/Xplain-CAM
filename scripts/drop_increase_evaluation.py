from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torchvision.models as models
from xplaincam.utils import load_image, sample_images_flat


# Main 
def main(args):

    from xplaincam.metrics import AvgDropConf
    from xplaincam.cam import (
        GradCAM, GradCAMpp, XGradCAM, GroupCAM, ScoreCAM, LayerCAM, AblationCAM, ISCAM, UnionCAM, FusionCAM
    )

    # Device
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Load model 
    model_name = args.model
    if not hasattr(models, model_name):
        raise ValueError(f"Model '{model_name}' is not available in torchvision.models")
    model_cls = getattr(models, model_name)
    model = model_cls(pretrained=True).eval().to(device)

    # Define CAM methods
    cams_dict = {
        "GradCAM": {"cam": GradCAM, "args": {"target_layer": args.target_layer}},
        "GradCAM++": {"cam": GradCAMpp, "args": {"target_layer": args.target_layer}},
        "XGradCAM": {"cam": XGradCAM, "args": {"target_layer": args.target_layer}},
        "ScoreCAM": {"cam": ScoreCAM, "args": {"target_layer": args.target_layer}},
        "GroupCAM": {"cam": GroupCAM, "args": {"target_layer": args.target_layer}},
        # "LayerCAM": {"cam": LayerCAM,"args": {"target_layer": args.target_layer}},
        "AblationCAM": {"cam": AblationCAM,"args": {"target_layer": args.target_layer}},
        # "RISE": {"cam": RISE, "args": {"input_shape": (3, 224, 224)}},
        "ISCAM": {"cam": ISCAM, "args": {"target_layer": args.target_layer}},
        "UnionCAM": {"cam": UnionCAM, "args": {"target_layer": args.target_layer}},
        "FusionCAM": {"cam": FusionCAM, "args": {"target_layer": args.target_layer, "grad_cam": GradCAM, "region_cam": ScoreCAM}},
    }

    if args.methods is not None:
        cams_dict = {k: v for k, v in cams_dict.items() if k in args.methods}

    # ----------------------------
    # Sample images
    # ----------------------------
    image_paths = sample_images_flat(
        Path(args.data_path),
        args.num_images,
        args.seed,
    )


    # Optional class names
    # class_names = load_class_index(Path(args.class_index)) if args.class_index else None

    # CSV saving setup
    save_csv = bool(args.output_csv)
    if save_csv:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["CAM Method", "Model", "Average Drop", "Increase in Confidence"])

    # Metrics accumulation
    metrics_acc = {cam_name: {"avg_drop": [], "increase_conf": []} for cam_name in cams_dict.keys()}

    # Evaluation loop
    for img_path in image_paths:
        input_tensor = load_image(img_path, args.image_size).to(device)
        input_tensor.requires_grad_(True)

        logits = model(input_tensor)
        pred_class = logits.argmax(dim=1).item()

        for cam_name, cam_info in cams_dict.items():
            cam_cls = cam_info["cam"]
            cam_args = cam_info["args"]

            with cam_cls(model, **cam_args) as cam_extractor:
                cam = cam_extractor(input_tensor, class_idx=pred_class)
                metric = AvgDropConf(model, heatmap=cam, input_tensor=input_tensor, class_idx=pred_class)

                metrics_acc[cam_name]["avg_drop"].append(metric.avg_drop)
                metrics_acc[cam_name]["increase_conf"].append(metric.avg_increase)

    # Display results and optionally save
    print("\n Evaluation completed\n")
    print(f"Average results over {len(image_paths)} images:\n")
    print(f"{'CAM Method':<15} {'Model':<10} {'Avg Drop':<12} {'Increase Conf':<15}")
    print("-"*55)

    for cam_name, vals in metrics_acc.items():
        avg_drop_mean = sum(vals["avg_drop"]) / len(vals["avg_drop"])
        increase_conf_mean = sum(vals["increase_conf"]) / len(vals["increase_conf"])

        # Format to 4 decimals
        avg_drop_fmt = f"{avg_drop_mean:.4f}"
        increase_conf_fmt = f"{increase_conf_mean:.4f}"

        print(f"{cam_name:<15} {model_name:<10} {avg_drop_fmt:<12} {increase_conf_fmt:<15}")

        if save_csv:
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([cam_name, model_name, avg_drop_fmt, increase_conf_fmt])

    if save_csv:
        print(f"\n Average results saved to {csv_path}")

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CAM methods on flat ImageNet validation set")

    parser.add_argument("data_path", type=str, help="Path to the ImageNet validation dataset")
    parser.add_argument("--model", type=str, default="vgg16", help="Name of the torchvision model to use (e.g., vgg16, resnet50)")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional CSV filename to save results")
    parser.add_argument("--num-images", type=int, default=100, help="Number of images to sample for evaluation")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size for the model")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for image sampling")
    parser.add_argument("--methods", nargs="+", default=None, help="Subset of CAM methods to evaluate")
    parser.add_argument("--target-layer", type=str, default=None, help="Target layer for CAM methods")
    parser.add_argument("--class-index", type=str, default=None, help="Optional JSON file for class index mapping")
    parser.add_argument("--device", type=str, default=None, help="Device to run the model on (cpu or cuda)")

    args = parser.parse_args()
    main(args)