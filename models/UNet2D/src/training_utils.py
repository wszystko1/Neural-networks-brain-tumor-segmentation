import torch
import torch.optim as optim

ENCODER_LAYERS = (
    "layer0",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
)

def get_decoder_params(model):
    return [
        p
        for name, p in model.named_parameters()
        if not name.startswith(ENCODER_LAYERS)
    ]

def make_optimizer_and_scheduler(param_groups, num_epochs):
    optimizer = optim.Adam(
        param_groups,
        weight_decay=1e-5,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=1e-6,
        )
    )

    return optimizer, scheduler

def freeze_encoder(model):
    for layer_name in ENCODER_LAYERS:
        for param in getattr(model, layer_name).parameters():
            param.requires_grad = False

def unfreeze_layers(model, *layer_names):
    for layer_name in layer_names:
        for param in getattr(model, layer_name).parameters():
            param.requires_grad = True