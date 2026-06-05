from helper import get_filepath, normalized_modality
import nibabel as nib
import torch 
import torch.functional as F
from pathlib import Path
from dotenv import dotenv_values, find_dotenv

config = dotenv_values(find_dotenv(usecwd=True))
DATASET_PATH = Path(config["DATASET_PATH"])
DATASET_PATH.mkdir(exist_ok=True)

NUM_BRAINS = int(config["NUM_BRAINS"])
MODALITIES = config["MODALITIES"].split(",")
TARGET_SIZE = (128, 128)

def preprocess_and_cache(brain_index, mod):
    path = get_filepath(brain_index, mod)
    img = nib.load(path)
    brain = torch.tensor(img.get_fdata(caching='unchanged'), dtype=torch.float32)
    brain = brain.permute(2, 0, 1)  # (155, H, W)

    if mod != "seg":
        brain = normalized_modality(brain)
        # Resize to 128×128 instead of padding to 256
        brain = F.interpolate(
            brain.unsqueeze(1),
            size=TARGET_SIZE,
            mode='bilinear',
            align_corners=False
        ).squeeze(1)             # (155, 128, 128)
        brain = brain.unsqueeze(1)  # (155, 1, 128, 128)
    else:
        brain[brain == 4] = 3
        brain = F.interpolate(
            brain.unsqueeze(1).float(),
            size=TARGET_SIZE,
            mode='nearest'
        ).squeeze(1).to(torch.int64)  # (155, 128, 128)

    brain_path = DATASET_PATH / f"brain_{brain_index:03d}_{mod}.pt"
    torch.save(brain, brain_path)
    return brain

if __name__ == "__main__":
    for idx in range(NUM_BRAINS):
        for modality in MODALITIES:
            preprocess_and_cache(idx,modality)