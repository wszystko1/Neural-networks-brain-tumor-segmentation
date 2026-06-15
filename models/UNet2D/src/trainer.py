import torch
from metrics import dice_per_class
from config import CLASS_NAMES

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_samples = 0
    cumulative_dice = {c: 0.0 for c in CLASS_NAMES}
    num_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

        batch_dice = dice_per_class(logits.detach(), y)

        for c in CLASS_NAMES:
            cumulative_dice[c] += batch_dice[c]

        num_batches += 1

    avg_dice = {
        c: cumulative_dice[c] / num_batches
        for c in CLASS_NAMES
    }

    return total_loss / total_samples, avg_dice


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
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

        batch_dice = dice_per_class(logits, y)

        for c in CLASS_NAMES:
            cumulative_dice[c] += batch_dice[c]

        num_batches += 1

    avg_dice = {
        c: cumulative_dice[c] / num_batches
        for c in CLASS_NAMES
    }

    return total_loss / total_samples, avg_dice