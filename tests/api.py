from __future__ import annotations
import os
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, resize, normalize
from torchvision.models import vgg19
from xplaincam.metrics import AvgDropConf, InsertionDeletion
from xplaincam.cam import *
from xplaincam.utils import visualize_cam



IMAGE_PATH = "data/dog_cat.png"
TARGET_LAYER=None

cams_dict = {
    "GradCAM": {"cam": GradCAM, "args": {"target_layer": TARGET_LAYER}},
    "GradCAM++": {"cam": GradCAMpp, "args": {"target_layer": TARGET_LAYER}},
    "XGradCAM": {"cam": XGradCAM, "args": {"target_layer": TARGET_LAYER}},
    "ScoreCAM": {"cam": ScoreCAM, "args": {"target_layer": TARGET_LAYER,"batch_size":2}},
    "GroupCAM": {"cam": GroupCAM, "args": {"target_layer": TARGET_LAYER}},
    "LayerCAM": {"cam": LayerCAM,"args": {"target_layer": TARGET_LAYER}},
    "AblationCAM": {"cam": AblationCAM,"args": {"target_layer": TARGET_LAYER}},
    "Integrated Gradients": {"cam": IntegratedGradients}, 
    "RISE": {"cam": RISE, "args": {"input_shape": (3, 224, 224)}}, 
    "ISCAM": {"cam": ISCAM, "args": {"target_layer": TARGET_LAYER}},
    "SSCAM": {"cam": SSCAM, "args": {"target_layer": TARGET_LAYER}},
    "UnionCAM": {"cam": UnionCAM, "args": {"target_layer": TARGET_LAYER}},
    "FusionCAM": {"cam": FusionCAM, "args": {"target_layer": TARGET_LAYER, "grad_cam":GradCAM, "region_cam":ScoreCAM}},


}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD IMAGE
# --------------------------------------------------
img = Image.open(IMAGE_PATH).convert("RGB")
img = resize(img, (224, 224))
img_tensor = to_tensor(img)  # (3,H,W)

# Model input (normalized)
input_tensor = normalize(
    img_tensor,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
).unsqueeze(0).to(DEVICE)

# Image for visualization (no normalization)
img_show = img_tensor.unsqueeze(0)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = vgg19(pretrained=True).eval().to(DEVICE)

# Forward pass to get predicted class
with torch.no_grad():
    output = model(input_tensor)
    predicted_class = output.argmax(dim=1).item()
print(f"Predicted class: {predicted_class}")

# --------------------------------------------------
# GENERATE CAMS AND COMPUTE METRICS
# --------------------------------------------------
cams = {}
avgdrop_results = {}
insdel_results = {}

for cam_name, cam_info in cams_dict.items():
    cam_cls = cam_info["cam"]
    cam_args = cam_info.get("args", {})
    predicted_class=281
    # Create the CAM instance dynamically
    with cam_cls(model, **cam_args) as cam_extractor:
        cam = cam_extractor(
            input_tensor,
            class_idx=predicted_class
        )

        # move CAM to CPU
        cam = cam.detach().cpu()
        cams[cam_name] = cam

        # AvgDrop / Increase
        metric = AvgDropConf(model, cam, input_tensor, class_idx=predicted_class)
        avgdrop_results[cam_name] = metric.summary()
        print(f"{cam_name} AvgDrop/Increase: {avgdrop_results[cam_name]}")

        # Insertion
        insdel = InsertionDeletion(model=model, mode="Insertion")
        ins_result = insdel.compute(image=input_tensor, heatmap=cam, return_steps=True)
        insdel_results.setdefault(cam_name, {})["Insertion"] = ins_result

        # Deletion
        insdel_del = InsertionDeletion(model=model, mode="Deletion")
        del_result = insdel_del.compute(image=input_tensor, heatmap=cam, return_steps=True)
        insdel_results[cam_name]["Deletion"] = del_result

        # Plot single CAM insertion + deletion
        print(f"Plotting curves for {cam_name}...")
        InsertionDeletion.plot_curves(
            results=ins_result,
            mode="Insertion",
            image=img_show,
            cam=cam,
            alpha=0.6,
            title=cam_name,
            save_path=os.path.join(SAVE_DIR, f"{cam_name}_insertion.png")
        )

        InsertionDeletion.plot_curves(
            results=del_result,
            mode="Deletion",
            image=img_show,
            cam=cam,
            title=cam_name,
            alpha=0.6,
            save_path=os.path.join(SAVE_DIR, f"{cam_name}_deletion.png")
        )


# Visualize all CAMs in one figure
visualize_cam(
    img=img_show,
    cams=cams,
    nrow=7,
    save_path=os.path.join(SAVE_DIR, "all_cams.png")
)


# Insertion curves
all_ins_results = {name: res["Insertion"] for name, res in insdel_results.items()}
InsertionDeletion.plot_curves(
    results=all_ins_results,
    mode="Insertion",
    title="Insertion Curves Comparison",
    save_path=os.path.join(SAVE_DIR, "all_cams_insertion.png")
)

# Deletion curves
all_del_results = {name: res["Deletion"] for name, res in insdel_results.items()}
InsertionDeletion.plot_curves(
    results=all_del_results,
    mode="Deletion",
    title="Deletion Curves Comparison",
    save_path=os.path.join(SAVE_DIR, "all_cams_deletion.png")
)

print("\nAll CAM processing completed. Plots saved to outputs/")