import nibabel as nib
import numpy as np
from model import UNet
from helper import get_cache
from inference import infer
from torch.utils.data import random_split, DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn 
import torch
import wandb
from monai.losses import DiceCELoss
from pathlib import Path
from dotenv import dotenv_values, find_dotenv
import os

config = {
    **dotenv_values(find_dotenv(usecwd=True)),
    **os.environ
}

DATASET_PATH = Path(config["DATASET_PATH"])
SAVE_PATH = Path(config["SAVE_PATH"])

NUM_BRAINS = int(config["NUM_BRAINS"])
BATCH_SIZE = int(config["BATCH_SIZE"])

LR = float(config["LR"])
NUM_EPOCHS = int(config["NUM_EPOCHS"])

wandb.login(config["WANDB_API_KEY"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=== Training configuration ===")
print(f"Device       : {device}")
print(f"Dataset path : {DATASET_PATH}")
print(f"Save path    : {SAVE_PATH}")
print(f"Brains       : {NUM_BRAINS}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"LR           : {LR}")
print(f"Epochs       : {NUM_EPOCHS}")
print("==============================")

class CustomDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data = []
        print("Loading all brains into RAM...")
        for index in range(NUM_BRAINS):
            brain_index = index + 1
            flair = get_cache(brain_index, "flair")
            t1ce  = get_cache(brain_index, "t1ce")
            seg   = get_cache(brain_index, "seg")
            self.data.append((flair, t1ce, seg))
            print(f"Loaded brain {brain_index}/{NUM_BRAINS}", end="\r")
        print("\nAll brains loaded.")

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        flair, t1ce, seg = self.data[index]

        has_tumor = seg.amax(dim=(1, 2)) > 0
        rand_bg   = torch.rand(seg.shape[0]) < 0.2
        keep      = has_tumor | rand_bg

        brain = torch.cat([flair[keep], t1ce[keep]], dim=1)
        return brain, seg[keep]

print("Creating dataset...")
dataset = CustomDataset(DATASET_PATH)
print(f"Dataset loaded: {len(dataset)} brains")

train_size = int(len(dataset) * 0.9)
val_size = len(dataset) - train_size

train_sub, val_sub = random_split(dataset, [train_size, val_size])
print(f"Train split: {len(train_sub)} brains")
print(f"Val split  : {len(val_sub)} brains")

train_loader = DataLoader(train_sub, batch_size=None, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_sub, batch_size=None, shuffle=False, pin_memory=True)

CLASS_NAMES = ["background", "necrotic", "edema", "enhancing"]

def dice_per_class(logits, y, num_classes=4, eps=1e-6):
    preds = logits.argmax(dim=1)
    scores = {}

    for c in range(num_classes):
        pred_c = (preds == c).float()
        true_c = (y == c).float()

        intersection = (pred_c * true_c).sum()
        denom = pred_c.sum() + true_c.sum()

        # if a class is absent in both pred and target, score is 1 (perfect)
        if denom < eps:
            scores[CLASS_NAMES[c]] = 1.0
        else:
            scores[CLASS_NAMES[c]] = (2.0 * intersection / denom).item()

    return scores

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    cumulative_dice = {c: 0.0 for c in CLASS_NAMES}
    num_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)#.unsqueeze(1)) # needed for loss function from monai

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

        batch_dice = dice_per_class(logits, y)
        for c in CLASS_NAMES:
            cumulative_dice[c] += batch_dice[c]
        num_batches += 1

    avg_dice = {c: cumulative_dice[c] / num_batches for c in CLASS_NAMES}
    return total_loss / total_samples, avg_dice

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    cumulative_dice = {c: 0.0 for c in CLASS_NAMES}
    num_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yb_input = y#.unsqueeze(1) #needed for the loss function from monai.

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, yb_input)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

        batch_dice = dice_per_class(logits.detach(), y)
        for c in CLASS_NAMES:
            cumulative_dice[c] += batch_dice[c]
        num_batches += 1
        
    avg_dice = {c: cumulative_dice[c] / num_batches for c in CLASS_NAMES}
    return total_loss / total_samples, avg_dice

print("Creating model...")
model = UNet()
model = model.to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

optimizer = optim.Adam(model.parameters(), lr = LR)

# criterion = DiceFocalLoss(
#     to_onehot_y=True,
#     softmax=True,
#     squared_pred=True,
#     gamma=2.0,       # focal strength, the higher it is the more it focuses on hard examples
#     lambda_dice=0.5, # balance between Dice and Focal terms
#     lambda_focal=0.5,
# )

class_weights = torch.tensor([0.1, 2.0, 2.0, 2.0], device=device)
criterion = nn.CrossEntropyLoss(
    weight=class_weights
)
# criterion = DiceCELoss(
#     to_onehot_y=True,
#     softmax=True,
#     squared_pred=True,
#     weight=class_weights,
# )

run = wandb.init(
    project="Brain-tumor-segmentation",
    config={
        "learning_rate": LR,
        "architecture": "UNET",
        "dataset": "BraTS2020",
        "loss" : "weighted nn.CrossEntropy",
        "loss ratio" : "None", # CHANGE THIS TO MATCH THE DICE/ENTOPY RATIO IN THE LOSS FUNCTION WHEN ITS ADDED 
        "weights" : class_weights,
        "epochs": NUM_EPOCHS,
        "no. brains": NUM_BRAINS,
        "comment": "Loading all the brains at once. Testing showed a big problem in backward computation"
    },
)
print(f"W&B run: {run.name}")

print("Starting training...")
for epoch in range(NUM_EPOCHS):
    tr_loss, tr_dice = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_dice = evaluate(model, val_loader, criterion, device)

    log_dict = {
        "epoch": epoch + 1,
        "train_loss": tr_loss,
        "val_loss": val_loss,
    }
    
    for c in CLASS_NAMES:
        log_dict[f"train_dice/{c}"] = tr_dice[c]
        log_dict[f"val_dice/{c}"]   = val_dice[c]

    run.log(log_dict)

    print(
        f"Epoch {epoch+1} | "
        f"train_loss={tr_loss:.4f} | val_loss={val_loss:.4f}\n"
        f"  Train Dice: { {k: f'{v:.3f}' for k,v in tr_dice.items()} }\n"
        f"  Val   Dice: { {k: f'{v:.3f}' for k,v in val_dice.items()} }"
    )

print("Training finished.")
run_number = int(run.name.split("-")[-1])
run.finish()

checkpoint = {
    "epoch": NUM_EPOCHS,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}

checkpoint_path = f"{SAVE_PATH}/unet_checkpoint_{run_number}.pth"
print(f"Saving checkpoint to: {checkpoint_path}")

torch.save(checkpoint, checkpoint_path)

# infer(device, 13,[val_sub.indices[i] + 1 for i in range(6)])