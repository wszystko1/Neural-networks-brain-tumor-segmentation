import numpy as np
from dotenv import dotenv_values,find_dotenv
from pathlib import Path
import torch
import torch.nn.functional as F

config = dotenv_values(find_dotenv(usecwd=True))
TR_DATA_PATH = Path(config.get("TR_DATA_PATH"))

def get_filepath(num: int, mod: str):
    '''
    Provides the path to the files containing a brain scan no. **num** and modality **mod**.

    :param int num: A number between 1 and 369 representing the id of the brain scan.
    :param ["t1", "t1ce", "t2", "flair", "seg"] mod: Modality of the scan of the chosen brain. 

    '''
    formated_num = format_num(num)
    if formated_num != 355:
        path = (
            TR_DATA_PATH /
            (f"BraTS20_Training_" + formated_num) /
            (f"BraTS20_Training_" + formated_num + f"_{mod}.nii")
        )
    else:
        path = (
            TR_DATA_PATH / "BraTS20_Training_355" / "W39_1998.09.19_Segm.nii"
        )
    return path

def normalized_modality(brain: np.ndarray): #normalization scales to [0,1] (or similiar), standarization changes the data to N(0,1)
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

def pad_to_256(img: torch.Tensor):
    '''
    Adds padding to the image. Transforms any image to shape (256,256).

    :param torch.Tensor img: Image vector.
    '''
    if img.dim() == 2:
        img = img.unsqueeze(0)

    _, H, W = img.shape
    pad_h = 256 - H
    pad_w = 256 - W

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom))
    return img

def format_num(num):
    '''
    Formats the number to lenght 3, filling it with 0 from the left side.
    Example: 9 -> 009

    :param int num: Number to format.
    '''
    return f"{num:03}" 
