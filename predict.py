import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import argparse
import os
import glob
from tqdm import tqdm
from utils.colorizer import get_colors
from datetime import datetime


def predict(config):
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get model
    model = torch.jit.load(config.weights, map_location=device).to(device)

    # create destination dir
    destination_path = os.path.join(config.destination, 'predict', datetime.now().strftime("%m-%d-%Y-%H:%M:%S"))
    config.destination = destination_path
    os.makedirs(config.destination, exist_ok=True)

    # get colormap
    colormap = get_colors()

    # get images
    image_list = []
    if os.path.isdir(config.source):
        image_list = glob.glob(os.path.join(config.source, "*"))
    elif "*" in config.source:
        image_list = glob.glob(config.source)
    elif os.path.exists(config.source):
        image_list = [config.source]

    image_list = [
        im
        for im in image_list
        if os.path.splitext(im)[1].lower() in [".png", ".jpg", ".jpeg"]
    ]

    for im_path in tqdm(image_list):
        original = cv2.imread(im_path)

        input = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        # resize image
        h_before_resize, w_before_resize, _ = input.shape
        input = A.SmallestMaxSize(max_size=config.img_size)(image=input)["image"]

        # pad image
        h_before_pad, w_before_pad, _ = input.shape
        input = A.PadIfNeeded(
            min_height=None,
            min_width=None,
            pad_height_divisor=32,
            pad_width_divisor=32,
            position=A.PadIfNeeded.PositionType.TOP_LEFT,
        )(image=input)["image"]

        # normalize
        input = A.Normalize()(image=input)["image"]

        # to tensor
        input = ToTensorV2()(image=input)["image"]

        # get output
        with torch.no_grad():
            output = model(input.to(device).unsqueeze(0))
            if output.size(1) == 1:
                output = (output > 0).long()
            else:
                output = output.argmax(1)
        output = output.squeeze().cpu().numpy()

        # remove padding
        output = output[:h_before_pad, :w_before_pad]

        # resize back
        output = A.resize(
            output, h_before_resize, w_before_resize, interpolation=cv2.INTER_NEAREST
        )

        # color
        output_colored = colormap[output]

        # save
        if config.format == "colored":
            name = os.path.splitext(os.path.basename(im_path))[0] + "_segm.png"
            cv2.imwrite(
                os.path.join(config.destination, name), np.uint8(output_colored)
            )
        elif config.format == "overlay":
            name = os.path.splitext(os.path.basename(im_path))[0] + "_segm.jpg"
            overlay = original * 0.5 + output_colored * 0.5
            overlay[output == 0] = original[output == 0]
            print(os.path.join(config.destination, name))
            cv2.imwrite(os.path.join(config.destination, name), np.uint8(overlay))


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--source",
        type=str,
        default="./input.jpg",
        help="input path, eg input.jpg or /path/to/images",
    )
    parse.add_argument(
        "--weights", type=str, default="./best_miou_model.pt", help="weights file"
    )
    parse.add_argument("--img_size", type=int, default=512, help="image size")
    parse.add_argument(
        "--destination", type=str, default="./output", help="output path"
    )
    parse.add_argument(
        "--format",
        type=str,
        default="overlay",
        help="output format, choices: colored, overlay",
    )
    return parse.parse_args()


if __name__ == "__main__":
    config = parse_args()
    predict(config)
