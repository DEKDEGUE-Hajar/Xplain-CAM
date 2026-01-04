<h1 align="center">
  Xplain-CAM: Explaining Convolutional Neural Networks Using Class Activation Maps.
</h1>

XplainCAM is a python library that allows you to visualize which regions of an image contribute most to a model's predictions and evaluate CAM quality with metrics like Average Drop/increase in confidence, Insertion, and Deletion.


![All CAMs](https://raw.githubusercontent.com/DEKDEGUE-Hajar/Xplain-CAM/main/output/dog/all_cams.png)



## Installation

The library can be installed directly via pip:
```bash
pip install xplaincam
```


## Quick Start: Single CAM Workflow

### Retrieving the class activation map

After initializing the CAM extractor, you can run inference as usual. By default, the class activation map is computed from the network’s last convolutional layer; to analyze a different layer, provide it explicitly via the `target_layer` argument in the constructor.

```python
import torch
from torchvision.models import vgg19
from torchvision.transforms.functional import to_tensor, resize, normalize
from PIL import Image
from xplaincam.cam import GradCAM
from xplaincam.utils import visualize_cam

# --- Configuration ---
IMAGE_PATH = "path/to/your/image.png"
model = vgg19(pretrained=True).eval()

# --- Image preparation ---
img = Image.open(IMAGE_PATH).convert("RGB")
img_tensor = resize(to_tensor(img), (224, 224))  # (3,H,W)

input_tensor = normalize(
    img_tensor,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
).unsqueeze(0)

# For visualization (unnormalized)
img_show = img_tensor.unsqueeze(0)

# --- Compute CAM ---
with GradCAM(model) as cam_extractor:
    output = model(input_tensor)
    cam = cam_extractor(input_tensor, class_idx=output.argmax(dim=1).item())

# --- Visualize ---
visualize_cam(img=img_show, cams={"GradCAM": cam})

```


<p align="center">
<img src="https://raw.githubusercontent.com/DEKDEGUE-Hajar/Xplain-CAM/main/output/gradcam_dog.png" alt="GradCAM dog" width="500" height="300"/>
</p>

You can compute the CAM for other objects present in the image by changing the `class_idx` to visualize how the model attends to alternative classes.


```python
# ImageNet class IDs (used in this example)
DOG_CLASS = 243   # “bull mastiff”
CAT_CLASS = 281   # “tabby cat”

with GradCAM(model, target_layer=None) as cam_extractor:
    output = model(input_tensor)
    cam = cam_extractor(input_tensor, class_idx = CAT_CLASS)

```
<p align="center">
<img src="https://raw.githubusercontent.com/DEKDEGUE-Hajar/Xplain-CAM/main/output/gradcam_cat.png" alt="GradCAM cat" width="500" height="300"/>
</p>

### Evaluate with Average Drop/Increase
Average Drop and Increase measure how much the model confidence decreases or increases when only the most relevant regions (according to the CAM) are retained.

```python
from xplaincam.metrics import AvgDropConf

metric = AvgDropConf(model, cam, input_tensor, class_idx=output.argmax(dim=1).item())
summary = metric.summary()
print(f"AvgDrop: {summary['avg_drop']:.4f}, AvgIncrease: {summary['avg_increase']:.4f}")

```

### Evaluate with Insertion/Deletion

Insertion and Deletion are fidelity metrics:

- Insertion: Measures how quickly model confidence rises as important pixels (from CAM) are gradually revealed on a blank background.

- Deletion: Measures how quickly model confidence drops as important pixels are progressively removed from the image.

Both metrics produce a curve of confidence vs. pixels added/removed, and the AUC (Area Under Curve) summarizes the overall performance.

```python

from xplaincam.metrics import InsertionDeletion

InsertionDeletion = InsertionDeletion(model=model)

# Insertion
ins_result = InsertionDeletion.compute(image=input_tensor, heatmap=cam, mode="Insertion", return_steps=True)

# Deletion
del_result = InsertionDeletion.compute(image=input_tensor, heatmap=cam, mode="Deletion", return_steps=True)

print(f"Insertion AUC: {ins_result['auc']:.4f}")
print(f"Deletion AUC: {del_result['auc']:.4f}")


```
### Plot Insertion and Deletion Curves
Visualize how confidence changes as pixels are added or removed.

```python

import os
from xplaincam.metrics import InsertionDeletion

os.makedirs("outputs", exist_ok=True)

InsertionDeletion.plot_curves(
    results=ins_result,
    mode="Insertion",
    image=img_show,
    cam=cam,
    alpha=0.6,
    save_path=os.path.join("output", "gradcam_insertion.png")
)

```

<p align="center">
<img src="https://raw.githubusercontent.com/DEKDEGUE-Hajar/Xplain-CAM/main/output/gradcam_dog_insertion.png" alt="GradCAM Dog Insertion" width="500" height="300"/>
</p>

```python
InsertionDeletion.plot_curves(
    results=del_result,
    mode="Deletion",
    image=img_show,
    cam=cam,
    alpha=0.6,
    save_path=os.path.join("output", "gradcam_deletion.png")
)

```
<p align="center">
<img src="https://raw.githubusercontent.com/DEKDEGUE-Hajar/Xplain-CAM/main/output/gradcam_dog_deletion.png" alt="GradCAM Dog Deletion" width="500" height="300"/>
</p>

The plots show the confidence curve for the model as pixels are gradually inserted or deleted, providing a visual indication of CAM quality.

## Comparative Analysis (Multiple CAMs)

The package supports easy comparison of multiple CAM methods on the same image.

### Visualize Multiple CAMs
Use the visualize_cam utility to display all generated heatmaps side-by-side.

```python
from XaiVisionCAM.utils import visualize_cam

# A dictionary mapping CAM names to their generated Tensor
cams = {
    "GradCAM": cam_gradcam,
    "GradCAMpp": cam_gradcampp,
    # ... and so on
}

visualize_cam(
    img=img_tensor.unsqueeze(0),
    cams=cams,
    nrow=3,
    save_path=os.path.join("output", "all_cams.png")
)

```

![All CAMs](https://raw.githubusercontent.com/DEKDEGUE-Hajar/Xplain-CAM/main/output/dog/all_cams.png)
![All CAMs](https://raw.githubusercontent.com/DEKDEGUE-Hajar/Xplain-CAM/main/output/cat/all_cams.png)



### Plot Comparative Metric Curves
To compare the performance of multiple CAMs using Insertion/Deletion, pass a dictionary of results to plot_curves

```python

# Insertion curves comparison
all_ins_results = {name: res["Insertion"] for name, res in insdel_results.items()}
InsertionDeletion.plot_curves(
    results=all_ins_results,
    mode="Insertion",
    title="Insertion Curves Comparison",
    save_path=os.path.join("output", "all_cams_insertion.png")
)

# Deletion curves comparison
all_del_results = {name: res["Deletion"] for name, res in insdel_results.items()}
InsertionDeletion.plot_curves(
    results=all_del_results,
    mode="Deletion",
    title="Deletion Curves Comparison",
    save_path=os.path.join("output", "all_cams_deletion.png")
)

```

<p align="center">
  <img src="https://raw.githubusercontent.com/DEKDEGUE-Hajar/Xplain-CAM/main/output/dog/insertion/all_cams_insertion.png" alt="All CAMs Insertion" width="300" style="display:inline-block; margin-right:20px;" />
  <img src="https://raw.githubusercontent.com/DEKDEGUE-Hajar/Xplain-CAM/main/output/dog/deletion/all_cams_deletion.png" alt="All CAMs Deletion" width="300" style="display:inline-block;" />
</p>

## Running the Scripts
The following scripts are provided to run batch evaluations over multiple images. 
Each script can be executed from the command line with configurable arguments.  
**All script arguments can be checked using `python scripts/cam_example.py --help`.**

> **Note:** This benchmark was performed on 100 randomly selected images from the ImageNet ILSV2012 validation set, and the same subset was used for all scripts.

---

### 1. Average Drop and Increase Evaluation Script

```bash
python drop_increase_evaluation.py \
    --data_path ILSV2012/val \
    --model vgg19 \
    --methods GradCAM GradCAMpp XGradCAM ScoreCAM GroupCAM AblationCAM ISCAM UnionCAM FusionCAM \
    --num-images 100 \
    --output-csv results/avg_drop_increase.csv
```
| CAM Method | Model | Average Drop (↓) | Increase in Confidence (↑) |
|------------|-------|-----------------|----------------------------|
| [Grad-CAM](https://arxiv.org/abs/1610.02391) | vgg19 | 0.2922 | 0.2000 |
| [Grad-CAM++](https://arxiv.org/abs/1710.11063) | vgg19 | 0.2897 | 0.1800 |
| [XGrad-CAM](https://arxiv.org/abs/2004.10528) | vgg19 | 0.3709 | 0.1400 |
| [Score-CAM](https://arxiv.org/pdf/1910.01279.pdf) | vgg19 | 0.2826 | 0.1800 |
| [Group-CAM](https://arxiv.org/pdf/2103.13859) | vgg19 | 0.2812 | 0.1600 |
| [Ablation-CAM](https://openaccess.thecvf.com/content_WACV_2020/papers/Desai_Ablation-CAM_Visual_Explanations_for_Deep_Convolutional_Network_via_Gradient-free_Localization_WACV_2020_paper.pdf) | vgg19 | 0.3319 | 0.1700 |
| [IS-CAM](https://arxiv.org/abs/2010.03023) | vgg19 | 0.3587 | 0.1700 |
| [Union-CAM](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1490198/full) | vgg19 | 0.2601 | 0.2100 |
| [Fusion-CAM](https://arxiv.org) | vgg19 | 0.1915 | 0.2800 |

### 2. Deletion and Insetion Evaluation Script

```bash
python del_inser_evaluation.py \
    --data_path ILSV2012/val \
    --model vgg19 \
    --methods GradCAM GradCAMpp XGradCAM ScoreCAM GroupCAM AblationCAM ISCAM UnionCAM FusionCAM \
    --num-images 100 \
    --output-csv results/del_inser_results.csv
    --save-plot-path results


```

| CAM Method | Model | Deletion (↓) | Insertion (↑) |
|------------|-------|----------|-----------|
| [Grad-CAM](https://arxiv.org/abs/1610.02391) | vgg19 | 0.0790 | 0.5167 |
| [Grad-CAM++](https://arxiv.org/abs/1710.11063) | vgg19 | 0.0861 | 0.5230 |
| [XGrad-CAM](https://arxiv.org/abs/2004.10528) | vgg19 | 0.0791 | 0.5107 |
| [Score-CAM](https://arxiv.org/pdf/1910.01279.pdf) | vgg19 | 0.0826 | 0.5375 |
| [Group-CAM](https://arxiv.org/pdf/2103.13859) | vgg19 | 0.0798 | 0.5325 |
| [Ablation-CAM](https://openaccess.thecvf.com/content_WACV_2020/papers/Desai_Ablation-CAM_Visual_Explanations_for_Deep_Convolutional_Network_via_Gradient-free_Localization_WACV_2020_paper.pdf) | vgg19 | 0.0775 | 0.5407 |
| [IS-CAM](https://arxiv.org/abs/2010.03023) | vgg19 | 0.0866 | 0.5284 |
| [Union-CAM](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1490198/full) | vgg19 | 0.0808 | 0.5362 |
| [Fusion-CAM](https://arxiv.org) | vgg19 | 0.0763 | 0.5444 |


<p align="center">
  <img src="https://raw.githubusercontent.com/DEKDEGUE-Hajar/Xplain-CAM/main/scripts/results/all_insertions_100.png" alt="All CAMs Insertion" width="300" style="display:inline-block; margin-right:20px;" />
  <img src="https://raw.githubusercontent.com/DEKDEGUE-Hajar/Xplain-CAM/main/scripts/results/all_deletions_100.png" alt="All CAMs Deletion" width="300" style="display:inline-block;" />
</p>



## FusionCAM: Combining Gradient- and Region-Based Class activtion Maps
*Toward more complete and robust class activation maps*

Class activation map (CAM) techniques can be broadly divided into gradient-based methods (e.g., GradCAM, GradCAM++, XGradCAM) and region-based methods (e.g., ScoreCAM, GroupCAM, AblationCAM, ISCAM, SSCAM). These two families capture complementary aspects of model explanations.

**By default, FusionCAM combines GradCAM (gradient-based) and ScoreCAM (region-based).**
This default configuration provides a strong balance between **class-discriminative sensitivity** and **spatial coherence**, making it a robust baseline for most visual explanation tasks.

FusionCAM is not limited to this default setup and can fuse any combination of gradient-based and region-based CAM techniques.

### Custom FusionCAM Examples

```python
cams = {}
fusion_cams = {
    "FusionCAM(Grad_SS)": (GradCAM, SSCAM),
    "FusionCAM(Gradpp_IS)": (GradCAMpp, ISCAM),
    "FusionCAM(XGrad_Abl)": (XGradCAM, AblationCAM),
}

for name, (gcam, rcam) in fusion_cams.items():
    with FusionCAM(model, target_layer=None, grad_cam=gcam, region_cam=rcam) as cam:
        output = model(input_tensor)
        cams[name] = cam(input_tensor, class_idx = CAT_CLASS)

visualize_cam(img=img_show, cams=cams)
```
<p align="center">
<img src="https://raw.githubusercontent.com/DEKDEGUE-Hajar/Xplain-CAM/main/output/fusioncams/all_gradcams.png" alt="GradCAMs" width="500" height="300"/>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/DEKDEGUE-Hajar/Xplain-CAM/main/output/fusioncams/all_regioncams.png" alt="RegionCAMs" width="500" height="300"/>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/DEKDEGUE-Hajar/Xplain-CAM/main/output/fusioncams/all_fusioncams.png" alt="Custom FusionCAM examples" width="500" height="300"/>
</p>

## CAM Zoo

This project is developed and maintained by the repo owner, but the implementation was based on the following research papers:


- [Grad-CAM](https://arxiv.org/abs/1610.02391): Computes gradients of the target class w.r.t. feature maps, averages them across spatial dimensions, and weights the maps to get the final class activation map.  
- [Grad-CAM++](https://arxiv.org/abs/1710.11063): Extends Grad-CAM by using weighted combination of first- and second-order gradients for better localization of multiple occurrences of the target object.
- [Integrated Gradients](https://arxiv.org/abs/1703.01365): Attributes model predictions to input features by integrating gradients along a straight path from a baseline to the input.
- [Layer-CAM](http://mftp.mmcheng.net/Papers/21TIP_LayerCAM.pdf): Generates CAMs by computing pixel-wise gradient contributions at intermediate layers for more precise localization.
- [XGrad-CAM](https://arxiv.org/pdf/2008.02312): Refines Grad-CAM by normalizing gradient weights and enhancing sensitivity and conservation of activations.
- [Score-CAM](https://arxiv.org/pdf/1910.01279.pdf): Weights feature maps by measuring the increase in class score when each map is masked, then linearly combines the maps based on these scores.
- [Ablation-CAM](https://openaccess.thecvf.com/content_WACV_2020/papers/Desai_Ablation-CAM_Visual_Explanations_for_Deep_Convolutional_Network_via_Gradient-free_Localization_WACV_2020_paper.pdf): Evaluates the decrease in class score when each feature map is removed (ablated). and combines contributions to highlight class-relevant regions.
- [Group-CAM](https://arxiv.org/pdf/2103.13859): Divides feature maps into groups, evaluates each group’s effect on the class score, and combines them to generate the final CAM.
- [SS-CAM](https://arxiv.org/abs/2006.14255): extends Score-CAM by applying smoothing or random perturbations to each activation map, the final map is obtained by averaging the scores across these perturbed maps.
- [IS-CAM](https://arxiv.org/abs/2010.03023): Gradually scales each activation map from zero to full averaging class-score effects to improve attribution.
- [Union-CAM](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1490198/full): Combines gradient and region-based maps, using denoising, linear combination and choosing the one with the higher class score.
- [Fusion-CAM](https://arxiv.org): Fuses gradient and region-based CAMs using denoising, linear combination, and similarity-based fusion to produce the final map.


## Citation

