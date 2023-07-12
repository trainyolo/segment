# Semantic Segmentation Library

## Install

Clone github repo and install the necessary requirements

```bash
git clone https://github.com/trainyolo/segment
cd segment
pip install -r requirements.txt
```

## Train

```bash
python train.py --dataset_path /path/to/dataset
```

## Predict
```bash
python predict.py --source input.jpg   --weights /path/to/best_miou_model.pt
                           path/
                           path/*.jpg
```
