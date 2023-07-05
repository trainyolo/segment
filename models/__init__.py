import torch
from segmentation_models_pytorch import create_model


def get_model(name, **kwargs):
    model = create_model(name, **kwargs)
    return model