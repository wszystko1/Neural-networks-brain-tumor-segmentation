from torch.utils.data import Dataset
import torch

from helper import get_cache
from config import NUM_BRAINS

class CustomDataset(Dataset):
    def __init__(self, path):
        super().__init__()

        self.data = []

        print("Loading all brains into RAM...")
        for index in range(NUM_BRAINS):
            brain_index = index + 1

            flair = get_cache(brain_index, "flair")
            t1ce = get_cache(brain_index, "t1ce")
            seg = get_cache(brain_index, "seg")

            self.data.append((flair, t1ce, seg))
            print(f"Loaded brain {brain_index}/{NUM_BRAINS}", end="\r")

        print("\nAll brains loaded.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        flair, t1ce, seg = self.data[index]

        has_tumor = seg.amax(dim=(1, 2)) > 0
        rand_bg = torch.rand(seg.shape[0]) < 0.2
        keep = has_tumor | rand_bg

        brain = torch.cat([flair[keep], t1ce[keep]], dim=1)
        return brain, seg[keep]


def collate_brains(batch):
    xs = torch.cat([item[0] for item in batch], dim=0)
    ys = torch.cat([item[1] for item in batch], dim=0)
    return xs, ys