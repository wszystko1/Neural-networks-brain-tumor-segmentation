from helper import get_filepath, normalized_modality
import nibabel as nib
import torch
import torch.nn.functional as F 
from pathlib import Path
from dotenv import dotenv_values, find_dotenv
import matplotlib.pyplot as plt

config = dotenv_values(find_dotenv(usecwd=True))
DATASET_PATH = Path(config["DATASET_PATH"])
DATASET_PATH.mkdir(exist_ok=True)

NUM_BRAINS = int(config["NUM_BRAINS"])
MODALITIES = config["MODALITIES"].split(",")

for idx,val in enumerate(MODALITIES):
    MODALITIES[idx] = val.strip()

TARGET_SIZE = (128, 128)

# Bounding box from dataset scan — +1 on upper bounds for exclusive slicing
# CROP_X = (41, 196)   # 196 - 41 = 155px
# CROP_Y = (29, 223)   # 223 - 29 = 194px
CROP_X = (21, 216)
CROP_Y = (29, 224)

PREVIEW_SLICE = 77   # midpoint of 155 slices


def preprocess_and_cache(brain_index, mod, save_original=False):
    path = get_filepath(brain_index, mod)
    img = nib.load(path)
    brain = torch.tensor(img.get_fdata(caching='unchanged'), dtype=torch.float32)
    brain = brain.permute(2, 0, 1)  # (155, 240, 240)

    # Capture original slice before crop (only when needed for preview)
    orig_slice = brain[PREVIEW_SLICE].clone() if save_original else None

    # Crop to bounding box
    brain = brain[:, CROP_X[0]:CROP_X[1], CROP_Y[0]:CROP_Y[1]]  # (155, 155, 194)

    if mod != "seg":
        brain = normalized_modality(brain)
        brain = F.interpolate(
            brain.unsqueeze(1),
            size=TARGET_SIZE,
            mode='bilinear',
            align_corners=False
        ).squeeze(1)               # (155, 128, 128)
        brain = brain.unsqueeze(1) # (155,   1, 128, 128)
    else:
        brain[brain == 4] = 3
        brain = F.interpolate(
            brain.unsqueeze(1).float(),
            size=TARGET_SIZE,
            mode='nearest'
        ).squeeze(1).to(torch.int64)  # (155, 128, 128)

    brain_path = DATASET_PATH / f"brain_{brain_index:03d}_{mod}.pt"
    torch.save(brain, brain_path)
    return brain, orig_slice


def save_comparison_png(brain_index, mod, brain_cropped, orig_slice):
    orig_np = orig_slice.numpy()                             # (240, 240)
    cmap = 'gray' if mod != "seg" else 'tab10'

    if mod != "seg":
        cropped_np = brain_cropped[PREVIEW_SLICE, 0].numpy()  # (128, 128)
    else:
        cropped_np = brain_cropped[PREVIEW_SLICE].numpy()     # (128, 128)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Brain {brain_index:03d} | {mod} | slice {PREVIEW_SLICE}", fontsize=13)

    # Left original with red bounding box overlay
    axes[0].imshow(orig_np, cmap=cmap, origin='upper')
    axes[0].set_title(f"Original ({orig_np.shape[0]}x{orig_np.shape[1]})")
    rect = plt.Rectangle(
        (CROP_Y[0], CROP_X[0]),          # (col, row)
        CROP_Y[1] - CROP_Y[0],           # width  = 194
        CROP_X[1] - CROP_X[0],           # height = 155
        linewidth=2, edgecolor='red', facecolor='none', label='Crop region'
    )
    axes[0].add_patch(rect)
    axes[0].legend(loc='lower right', fontsize=9)

    # Right — cropped + resized result
    axes[1].imshow(cropped_np, cmap=cmap, origin='upper')
    axes[1].set_title(f"Cropped → Resized ({TARGET_SIZE[0]}x{TARGET_SIZE[1]})")

    plt.tight_layout()
    png_path = DATASET_PATH / f"preview_brain_{brain_index:03d}_{mod}.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved preview → {png_path.name}")


if __name__ == "__main__":
    previews_saved = 0
    for idx in range(1, NUM_BRAINS + 1):
        for modality in MODALITIES:
            need_preview = previews_saved < 4
            brain, orig_slice = preprocess_and_cache(idx, modality, save_original=need_preview)
            if need_preview:
                save_comparison_png(idx, modality, brain, orig_slice)
                previews_saved += 1