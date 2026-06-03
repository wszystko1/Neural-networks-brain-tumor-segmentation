import numpy as np
from dotenv import dotenv_values,find_dotenv
from pathlib import Path
import torch
import torch.nn.functional as F
import time

config = dotenv_values(find_dotenv(usecwd=True))
TR_DATA_PATH = Path(config.get("TR_DATA_PATH"))

def get_filepath(brain_index: int, mod: str = "t1ce"):
    '''
    Provides the path to the files containing a brain scan no. **brain_index** and modality **mod**.

    :param int brain_index: A number between 1 and 369 representing the id of the brain scan.

    :param ["t1", "t1ce", "t2", "flair", "seg"] mod: Modality of the scan of the chosen brain. Defaults to "t1".

    '''

    formated_index = format_index(brain_index)
    if formated_index == "355" and mod == "seg":
        path = (
            TR_DATA_PATH / "BraTS20_Training_355" / "W39_1998.09.19_Segm.nii"
        )
    else:
        path = (
            TR_DATA_PATH /
            (f"BraTS20_Training_" + formated_index) /
            (f"BraTS20_Training_" + formated_index + f"_{mod}.nii")
        )

    return path

def normalized_modality(brain: np.ndarray):
    '''
    Standarizes all the pixels from one modality to P ~ N(0,1).

    :param np.ndarray brain: An array containing pixels from one modality from a single brain scan.
    '''
    brain_mask = brain > 0
    if brain_mask.sum() == 0:
        print('Edge Case: empty brain')
        return brain
    
    mi = brain[brain_mask].mean()
    std = brain[brain_mask].std() + 1e-8

    brain[brain_mask] = (brain[brain_mask] - mi) / std
    return brain

def pad_to_256(brain: torch.Tensor):
    """
    Pads (155, 240, 240) brain volume to (155, 256, 256)
    """
    D, H, W = brain.shape
    assert D == 155 and H == 240 and W == 240

    pad_h = 256 - H
    pad_w = 256 - W

    pad_ver = pad_h // 2
    pad_hor = pad_w // 2

    pad = (pad_ver, pad_ver, pad_hor, pad_hor)

    brain_pad = F.pad(brain,  pad)
    return brain_pad

def format_index(brain_index: int):
    '''
    Formats the number to lenght 3, filling it with 0 from the left side.
    Example: '9' -> '009'

    :param int brain_index: Number to format.
    '''
    return f"{brain_index:03}" 


def diagnose_timing(model, loader, optimizer, criterion, device):
    model.train()
    
    for x, y in loader:
        print(f"Tensor shape from loader: {x.shape}")
        
        t0 = time.time()
        x, y = x.to(device), y.to(device)
        torch.cuda.synchronize()  # ← important: forces GPU to finish before timing
        t1 = time.time()

        logits = model(x)
        torch.cuda.synchronize()
        t2 = time.time()

        loss = criterion(logits, y)# .unsqueeze(1)
        torch.cuda.synchronize()
        t3 = time.time()

        loss.backward()
        torch.cuda.synchronize()
        t4 = time.time()

        optimizer.step()
        torch.cuda.synchronize()
        t5 = time.time()

        print(f"  .to(device)  : {t1-t0:.3f}s")
        print(f"  forward      : {t2-t1:.3f}s")
        print(f"  loss         : {t3-t2:.3f}s")
        print(f"  backward     : {t4-t3:.3f}s")
        print(f"  optim.step   : {t5-t4:.3f}s")
        print(f"  TOTAL        : {t5-t0:.3f}s")

        print(f"CUDA available : {torch.cuda.is_available()}")
        print(f"Device         : {device}")
        print(f"Model device   : {next(model.parameters()).device}")
        print(f"Data device    : {x.device}")
        print(f"GPU memory allocated by PyTorch: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
          # only time the first brain



def get_cache(brain_index, mod):
    '''
    load a .pt file to memory from a .env defined path.
    '''
    CACHE_DIR = Path(config.get("CACHE_PATH"))
    cache_path = CACHE_DIR / f"brain_{brain_index:03d}_{mod}.pt"

    if cache_path.exists():
        return torch.load(cache_path)
    