import torch
import torch.optim as optim

ENCODER_LAYERS = ("layer0_conv", "layer0_pool", "layer1", "layer2", "layer3", "layer4")

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
    for name in ENCODER_LAYERS:
        for param in getattr(model, name).parameters():
            param.requires_grad = False

def unfreeze_layers(model, *layer_names):
    for name in layer_names:
        for param in getattr(model, name).parameters():
            param.requires_grad = True

def get_decoder_params(model):
    return [p for name, p in model.named_parameters()
        if not any(name.startswith(enc) for enc in ENCODER_LAYERS)]