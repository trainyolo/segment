import torch

def denormalize_img_tensor(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.as_tensor(mean, dtype=img.dtype, device=img.device)
    std = torch.as_tensor(std, dtype=img.dtype, device=img.device)

    return img.mul(std[:,None,None]).add(mean[:,None,None])