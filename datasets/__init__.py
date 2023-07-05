from .SegmentationDataset import SegmentationDataset

def get_dataset(name, **kwargs):
    if name == "SegmentationDataset":
        return SegmentationDataset(**kwargs)
    else:
        raise RuntimeError(f'Dataset {name} is not available!')