from __future__ import annotations
import argparse
import os
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, resize, normalize
from torchvision.models import vgg19

from xplaincam.cam import *
from xplaincam.metrics import AvgDropConf, InsertionDeletion
from xplaincam.utils import visualize_cam


# --------------------------------------------------
# ARGUMENTS
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("XplainCAM Demo")

    parser.add_argument("--input", type=str, required=True,
                        help="Path to input image")

    parser.add_argument("--cams", type=str, nargs="+", required=True,
                        help="One or more CAM names (e.g. GradCAM LayerCAM)")

    parser.add_argument("--cls_idx", type=int, default=None,
                        help="Target class index (default: predicted class)")

    parser.add_argument("--output", type=str, default="outputs",
                        help="Directory to save results")

    parser.add_argument("--drop_increase_conf", action="store_true",
                        help="Compute AvgDrop / Increase in confidence")

    parser.add_argument("--ins_del", action="store_true",
                        help="Compute insertion & deletion curves")

    return parser.parse_args()


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    # --------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------
    img = Image.open(args.input).convert("RGB")
    img = resize(img, (224, 224))
    img_tensor = to_tensor(img)

    input_tensor = normalize(
        img_tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ).unsqueeze(0).to(device)

    img_show = img_tensor.unsqueeze(0)

    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------
    model = vgg19(pretrained=True).eval().to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()

    class_idx = args.cls_idx if args.cls_idx is not None else pred_class
    print(f"Target class: {class_idx}")

    # --------------------------------------------------
    # CAMS
    # --------------------------------------------------
    cams = {}
    avgdrop_results = {}
    insdel_results = {}

    target_layer = None  # Let CAM handle it internally

    for cam_name in args.cams:
        print(f"\nRunning {cam_name}...")

        if cam_name not in globals():
            raise ValueError(f"Unknown CAM: {cam_name}")

        cam_class = globals()[cam_name]

        with cam_class(model, target_layer=target_layer) as cam_extractor:
            cam = cam_extractor(
                input_tensor,
                class_idx=class_idx
            )

            cam = cam.detach().cpu()
            cams[cam_name] = cam

            # ------------------------------------------
            # AvgDrop / Increase
            # ------------------------------------------
            if args.drop_increase_conf:
                metric = AvgDropConf(
                    model=model,
                    heatmap=cam,
                    input_tensor=input_tensor,
                    class_idx=class_idx
                )
                res = metric.summary()
                avgdrop_results[cam_name] = res

                print(f"{cam_name} | "
                      f"AvgDrop: {res['avg_drop']:.4f} | "
                      f"AvgIncrease: {res['avg_increase']:.4f}")

            # ------------------------------------------
            # Insertion / Deletion
            # ------------------------------------------
            if args.ins_del:
                ins = InsertionDeletion(model, mode="Insertion")
                dele = InsertionDeletion(model, mode="Deletion")

                ins_res = ins.compute(
                    image=input_tensor,
                    heatmap=cam,
                    return_steps=True
                )

                del_res = dele.compute(
                    image=input_tensor,
                    heatmap=cam,
                    return_steps=True
                )

                insdel_results.setdefault(cam_name, {})
                insdel_results[cam_name]["Insertion"] = ins_res
                insdel_results[cam_name]["Deletion"] = del_res

                InsertionDeletion.plot_curves(
                    results=ins_res,
                    mode="Insertion",
                    image=img_show,
                    cam=cam,
                    title=cam_name,
                    save_path=os.path.join(
                        args.output, f"{cam_name}_insertion.png"
                    )
                )

                InsertionDeletion.plot_curves(
                    results=del_res,
                    mode="Deletion",
                    image=img_show,
                    cam=cam,
                    title=cam_name,
                    save_path=os.path.join(
                        args.output, f"{cam_name}_deletion.png"
                    )
                )

    # --------------------------------------------------
    # VISUALIZE ALL CAMS
    # --------------------------------------------------
    if cams:
        visualize_cam(
            img=img_show,
            cams=cams,
            nrow=4,
            save_path=os.path.join(args.output, "all_cams.png")
        )

    print("\nResults saved in:", args.output)


if __name__ == "__main__":
    main()
