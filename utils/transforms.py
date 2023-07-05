import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(config):
    transform_list = []
    
    if config.resize:
        transform_list.append(A.SmallestMaxSize(max_size=config.img_size))
    if config.random_crop:
        crop_size = config.crop_size
        min_size = crop_size - crop_size * config.crop_scale
        max_size = crop_size + crop_size * config.crop_scale
        transform_list.append(A.RandomSizedCrop([int(min_size),int(max_size)],config.crop_size,config.crop_size))
    if config.horizontal_flip:
        transform_list.append(A.HorizontalFlip(p=0.5))
    if config.vertical_flip:
        transform_list.append(A.VerticalFlip(p=0.5))
    if config.brightness_contrast:
        transform_list.append(A.RandomBrightnessContrast(p=0.5))
    
    transform_list.extend([
        A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32,pad_width_divisor=32),
        A.Normalize(),
        ToTensorV2(),
    ])

    return A.Compose(transform_list)


def get_val_transform(config):
    transform_list = []

    if config.resize:
        transform_list.append(A.SmallestMaxSize(max_size=config.img_size))

    transform_list.extend([
        A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32,pad_width_divisor=32),
        A.Normalize(),
        ToTensorV2(),
    ])
    
    return A.Compose(transform_list)