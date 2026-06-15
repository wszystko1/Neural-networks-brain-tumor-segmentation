from helper import get_cache
from models import UNet, UNetNorm, UNetResNet
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path
from dotenv import dotenv_values, find_dotenv
import os

config = {
    **dotenv_values(find_dotenv(usecwd=True)),
    **os.environ
}

SAVE_PATH = Path(config["SAVE_PATH"])

SEG_CMAP = ListedColormap(["black", "red", "green", "yellow"])
SEG_LABELS = ["Background", "Necrotic", "Edema", "Enhancing"]

MODELS = {
    "UNet": UNet,
    "UNetNorm": UNetNorm,
    "UNetResNet": UNetResNet,
}

def infer(device: torch.device, run_number: int, brain_numbers: list, model_version: str):
    model = MODELS[model_version]().to(device)
    checkpoint = torch.load(f"{SAVE_PATH}/unet_checkpoint_{run_number}.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    NUM_SHOW   = min(6, len(brain_numbers))
    MIDDLE     = 77  # slice 77/155 most brain content

    results = []  # list of (flair_slice, pred_slice, seg_slice)

    with torch.no_grad():
        for brain_idx in brain_numbers:
            flair = get_cache(brain_idx, "flair")  # (155, 1, 128, 128)
            t1ce  = get_cache(brain_idx, "t1ce")   # (155, 1, 128, 128)
            seg   = get_cache(brain_idx, "seg")    # (155, 128, 128)

            flair_s = flair[MIDDLE]   # (1, 128, 128)
            t1ce_s  = t1ce[MIDDLE]    # (1, 128, 128)
            seg_s   = seg[MIDDLE]     # (128, 128)

            # Build input and run inference
            x = torch.cat([flair_s, t1ce_s], dim=0).unsqueeze(0).to(device)  # (1, 2, 128, 128)
            logits = model(x)                          # (1, 4, 128, 128)
            pred = logits.argmax(dim=1).squeeze(0).cpu()  # (128, 128)

            results.append((
                flair_s.squeeze(0).cpu().numpy(),  # show flair channel
                pred.numpy(),
                seg_s.cpu().numpy()
            ))


    fig, axes = plt.subplots(NUM_SHOW, 3, figsize=(10, 3 * NUM_SHOW))
    fig.suptitle(f"Inference results — middle slice ({MIDDLE}/155)", 
                fontsize=14, fontweight='bold', y=1.01)

    col_titles = ["FLAIR Input", "Prediction", "Ground Truth"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight='bold')

    for row, (flair_np, pred_np, seg_np) in enumerate(results[:NUM_SHOW]):
        brain_num = brain_numbers[row]

        # FLAIR — grayscale
        axes[row, 0].imshow(flair_np, cmap="gray")
        axes[row, 0].set_ylabel(f"Brain {brain_num}", fontsize=9)

        # Prediction — segmentation colormap, values 0–3
        axes[row, 1].imshow(pred_np, cmap=SEG_CMAP, vmin=0, vmax=3)

        # Ground truth — segmentation colormap, values 0–3
        axes[row, 2].imshow(seg_np, cmap=SEG_CMAP, vmin=0, vmax=3)

        for col in range(3):
            axes[row, col].axis("off")

    legend_patches = [
        mpatches.Patch(color=SEG_CMAP(i / 3), label=SEG_LABELS[i])
        for i in range(4)
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=4,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True
    )

    fig_path = f"{SAVE_PATH}/inference_results_{run_number}.png",
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Saved to results/inference_results_{run_number}.png")