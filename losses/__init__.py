from torch import nn
from .losses import BCEDiceLoss, CEDiceLoss


def get_loss(name, **kwargs):
    if name == 'bce':
        return nn.BCEWithLogitsLoss(**kwargs)
    elif name == 'crossentropy':
        return nn.CrossEntropyLoss(**kwargs)
    elif name == 'bcedice':
        return BCEDiceLoss(**kwargs)
    elif name == 'cedice':
        return CEDiceLoss(**kwargs)
    else:
        raise RuntimeError(f'Loss {name} is not available!')