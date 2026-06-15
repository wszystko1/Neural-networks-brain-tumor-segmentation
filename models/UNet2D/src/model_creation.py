from models import UNet, UNetNorm, UNetResNet

def create_model(model_version):
    models = {
        "UNet": UNet,
        "UNetNorm": UNetNorm,
        "UNetResNet": UNetResNet,
    }

    if model_version not in models:
        raise ValueError(
            f"Unknown MODEL_VERSION: {model_version}"
        )
    return models[model_version]()