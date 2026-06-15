import torch
import torch.nn as nn
import wandb
from torch.utils.data import random_split, DataLoader
from helper import compute_class_stats
from config import *
from dataset import *
from trainer import *
from model_creation import *
from training_utils import *

def run_epoch(epoch, model, train_loader, val_loader, optimizer,
              scheduler, criterion, device, run, best_val_loss, best_path):
    tr_loss, tr_dice = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_dice = evaluate(model, val_loader, criterion, device)

    log_dict = {
        "epoch": epoch + 1,
        "train_loss": tr_loss,
        "val_loss": val_loss,
        "lr": scheduler.get_last_lr()[0],
    }
    for c in CLASS_NAMES:
        log_dict[f"train_dice/{c}"] = tr_dice[c]
        log_dict[f"val_dice/{c}"] = val_dice[c]
    run.log(log_dict)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }, best_path)

    scheduler.step()
    return best_val_loss

def main():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)

    wandb.login(key=config["WANDB_API_KEY"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CustomDataset(DATASET_PATH)
    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size
    train_sub, val_sub = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True,
                              generator=generator, pin_memory=True, collate_fn=collate_brains)
    val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False,
                            generator=generator, pin_memory=True, collate_fn=collate_brains)

    model = create_model(MODEL_VERSION).to(device)
    freq, w_sqrt = compute_class_stats(train_sub)
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:12s} freq={freq[i].item():.6f} w_sqrt={w_sqrt[i].item():.6f}")

    default_weights = torch.tensor([0.1, 2.0, 2.0, 2.0], device=device)
    class_weights = w_sqrt.to(device) if WEIGHTS_VERSION == "TUNED" else default_weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    run = wandb.init(
        project="Brain-tumor-segmentation",
        config={
            "learning_rate": LR,
            "architecture": MODEL_VERSION,
            "dataset": "BraTS2020",
            "loss": "weighted nn.CrossEntropy",
            "weights": class_weights.tolist(),
            "epochs": NUM_EPOCHS,
            "no. brains": NUM_BRAINS,
            **({"stage1_epochs": STAGE1_EPOCHS, "stage2_epochs": STAGE2_EPOCHS,
                "stage3_epochs": STAGE3_EPOCHS} if MODEL_VERSION == "UNetResNet" else {}),
        },
    )

    run_number = int(run.name.split("-")[-1])
    best_path = f"{SAVE_PATH}/unet_best_{run_number}.pth"

    if MODEL_VERSION == "UNetResNet":
        freeze_encoder(model)
        optimizer, scheduler = make_optimizer_and_scheduler(
            [{"params": get_decoder_params(model), "lr": LR}],
            STAGE1_EPOCHS,
        )
        best_val_loss = float("inf")
        for epoch in range(STAGE1_EPOCHS):
            best_val_loss = run_epoch(epoch, model, train_loader, val_loader,
                                      optimizer, scheduler, criterion,
                                      device, run, best_val_loss, best_path)

        unfreeze_layers(model, "layer3", "layer4")
        model.load_state_dict(torch.load(best_path)["model_state_dict"])
        optimizer, scheduler = make_optimizer_and_scheduler(
            [
                {"params": [*model.layer3.parameters(),
                            *model.layer4.parameters()], "lr": LR * 0.1},
                {"params": get_decoder_params(model),  "lr": LR},
            ],
            STAGE2_EPOCHS,
        )
        best_val_loss = float("inf")
        for epoch in range(STAGE2_EPOCHS):
            best_val_loss = run_epoch(epoch, model, train_loader, val_loader,
                                      optimizer, scheduler, criterion,
                                      device, run, best_val_loss, best_path)

        unfreeze_layers(model, "layer0_conv", "layer1", "layer2")
        model.load_state_dict(torch.load(best_path)["model_state_dict"])
        
        optimizer, scheduler = make_optimizer_and_scheduler(
            [
                {"params": [*model.layer0_conv.parameters(),
                            *model.layer1.parameters(),
                            *model.layer2.parameters()], "lr": LR * 0.01},
                {"params": [*model.layer3.parameters(),
                            *model.layer4.parameters()],  "lr": LR * 0.1},
                {"params": get_decoder_params(model),     "lr": LR},
            ],
            STAGE3_EPOCHS,
        )
        best_val_loss = float("inf")
        for epoch in range(STAGE3_EPOCHS):
            best_val_loss = run_epoch(epoch, model, train_loader, val_loader,
                                      optimizer, scheduler, criterion,
                                      device, run, best_val_loss, best_path)

    else:
        optimizer, scheduler = make_optimizer_and_scheduler(
            [{"params": model.parameters(), "lr": LR}],
            NUM_EPOCHS,
        )
        best_val_loss = float("inf")
        for epoch in range(NUM_EPOCHS):
            best_val_loss = run_epoch(epoch, model, train_loader, val_loader,
                                      optimizer, scheduler, criterion,
                                      device, run, best_val_loss, best_path)

    run.finish()

    final_path = f"{SAVE_PATH}/unet_final_{run_number}.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, final_path)

if __name__ == "__main__":
    main()
