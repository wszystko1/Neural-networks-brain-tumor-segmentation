from config import CLASS_NAMES

def dice_per_class(logits, y, num_classes=4, eps=1e-6):
    preds = logits.argmax(dim=1)
    scores = {}
    for c in range(num_classes):
        pred_c = (preds == c).float()
        true_c = (y == c).float()

        intersection = (pred_c * true_c).sum()
        denom = pred_c.sum() + true_c.sum()

        if denom < eps:
            scores[CLASS_NAMES[c]] = 1.0
        else:
            scores[CLASS_NAMES[c]] = (
                2.0 * intersection / denom
            ).item()
    return scores