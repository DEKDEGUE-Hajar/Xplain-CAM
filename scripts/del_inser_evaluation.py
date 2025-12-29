from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.models as models
from xplaincam.utils import load_image, sample_images_flat
from PIL import Image
from pathlib import Path


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------
def main(args):
    from xplaincam.metrics import InsertionDeletion
    from xplaincam.cam import (
        GradCAM, GradCAMpp, XGradCAM, GroupCAM, ScoreCAM, RISE, ISCAM, SSCAM, AblationCAM, UnionCAM, FusionCAM
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
        "AblationCAM": {"cam": AblationCAM, "args": {"target_layer": args.target_layer}},
        # "RISE": {"cam": RISE, "args": {"input_shape": (3, 224, 224)}},
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
            writer.writerow(["CAM Method", "Model", "Deletion", "Insertion"])

    # Metrics accumulators - now also storing full curves for averaging
    metrics_acc = {
        cam_name: {
            "deletion": [], 
            "insertion": [],
            "deletion_curves": [],  # Store full deletion curves for averaging
            "insertion_curves": []  # Store full insertion curves for averaging
        } 
        for cam_name in cams_dict.keys()
    }

    # Evaluation loop
    for img_idx, img_path in enumerate(image_paths):
        input_tensor = load_image(img_path, args.image_size).to(device)
        input_tensor.requires_grad_(True)

        logits = model(input_tensor)
        pred_class = logits.argmax(dim=1).item()

        for cam_name, cam_info in cams_dict.items():
            cam_cls = cam_info["cam"]
            cam_args = cam_info["args"]

            with cam_cls(model, **cam_args) as cam_extractor:
                heatmap = cam_extractor(input_tensor, class_idx=pred_class)

                # Compute Insertion and Deletion with steps
                # Insertion
                insdel = InsertionDeletion(model=model, mode="Insertion")
                ins_result = insdel.compute(image=input_tensor, heatmap=heatmap, class_idx=pred_class, return_steps=True)
                
                # Deletion
                insdel_del = InsertionDeletion(model=model, mode="Deletion")
                del_result = insdel_del.compute(image=input_tensor, heatmap=heatmap, class_idx=pred_class, return_steps=True)

                # Store AUC values
                metrics_acc[cam_name]["insertion"].append(ins_result["Insertion_auc"])
                metrics_acc[cam_name]["deletion"].append(del_result["Deletion_auc"])
                
                # Store full curves
                if "Insertion_steps" in ins_result:
                    metrics_acc[cam_name]["insertion_curves"].append(ins_result["Insertion_steps"])
                
                if "Deletion_steps" in del_result:
                    metrics_acc[cam_name]["deletion_curves"].append(del_result["Deletion_steps"])

        # Optional: print progress
        if (img_idx + 1) % 10 == 0:
            print(f"Processed {img_idx + 1}/{len(image_paths)} images")

    # Calculate averages
    print("\n" + "="*60)
    print("Evaluation completed")
    print("="*60)
    
    print(f"\nAverage results over {len(image_paths)} images:\n")
    print(f"{'CAM Method':<15} {'Model':<10} {'Deletion':<12} {'Insertion':<12}")
    print("-"*55)

    cam_results = {}
    for cam_name, vals in metrics_acc.items():
        if vals["deletion"] and vals["insertion"]:
            del_mean = sum(vals["deletion"]) / len(vals["deletion"])
            ins_mean = sum(vals["insertion"]) / len(vals["insertion"])
            
            cam_results[cam_name] = {
                "deletion_mean": del_mean,
                "insertion_mean": ins_mean,
                "deletion_curves": vals["deletion_curves"],
                "insertion_curves": vals["insertion_curves"]
            }

            del_fmt = f"{del_mean:.4f}"
            ins_fmt = f"{ins_mean:.4f}"

            print(f"{cam_name:<15} {args.model:<10} {del_fmt:<12} {ins_fmt:<12}")

            if save_csv:
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([cam_name, args.model, del_fmt, ins_fmt])
        else:
            print(f"{cam_name:<15} {args.model:<10} {'N/A':<12} {'N/A':<12}")

    if save_csv:
        print(f"\n Results saved to {csv_path}")

    # Plot average insertion and deletion curves
    if args.plot_curves:
        print("\n" + "="*60)
        print(" Generating average curves...")
        print("="*60)
        
        # Prepare results in the format expected by plot_curves
        insertion_results = {}
        deletion_results = {}
        
        for cam_name, results in cam_results.items():
            if results["insertion_curves"]:
                # Calculate average insertion curve
                insertion_curves = np.array(results["insertion_curves"])
                avg_insertion_curve = insertion_curves.mean(axis=0)
                
                insertion_results[cam_name] = {
                    "Insertion_steps": avg_insertion_curve,
                    "Insertion_auc": results["insertion_mean"]
                }
            
            if results["deletion_curves"]:
                # Calculate average deletion curve
                deletion_curves = np.array(results["deletion_curves"])
                avg_deletion_curve = deletion_curves.mean(axis=0)
                
                deletion_results[cam_name] = {
                    "Deletion_steps": avg_deletion_curve,
                    "Deletion_auc": results["deletion_mean"]
                }
        
        
        # Ensure the folder exists
        save_folder = Path(args.save_plot_path)
        save_folder.mkdir(parents=True, exist_ok=True)
        
        num_images = len(image_paths)  # number of images processed
        
        # Plot insertion curves
        if insertion_results:
            insertion_file = save_folder / f"all_insertions_{num_images}.png"
            InsertionDeletion.plot_curves(
                insertion_results,
                mode="Insertion",
                title=f"Insertion Curves Comparison (n={num_images})",
                save_path=str(insertion_file)
            )
        
        # Plot deletion curves
        if deletion_results:
            deletion_file = save_folder / f"all_deletions_{num_images}.png"
            InsertionDeletion.plot_curves(
                deletion_results,
                mode="Deletion",
                title=f"Deletion Curves Comparison (n={num_images})",
                save_path=str(deletion_file)
            )



# --------------------------------------------------
# CLI - Updated with new arguments
# --------------------------------------------------
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
    
    # New arguments for plotting
    parser.add_argument("--plot-curves", action="store_true", help="Plot average insertion/deletion curves")
    parser.add_argument("--save-plot-path", type=str, default=None, help="Path to save the generated plots")

    args = parser.parse_args()
    
    # Enable plotting by default if save path is provided
    if args.save_plot_path and not args.plot_curves:
        args.plot_curves = True
    
    main(args)