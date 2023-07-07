from engine import Engine
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--save_path', type=str, default='./output')
    
    parser.add_argument('--display', type=bool, default=True)
    parser.add_argument('--display_it', type=int, default=50)

    parser.add_argument('--pretrained_model_path', type=str)

    parser.add_argument('--dataset_path', type=str, default='./dataset', help='dataset.yaml path')
    parser.add_argument('--dataset_size', type=int, default=100)
    
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--train_workers', type=int, default=4)
    
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--val_workers', type=int, default=4)

    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--model_encoder', type=str, default='resnet18')

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--max_epochs', type=int)

    parser.add_argument('--resize', type=bool, default=True)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--random_crop', type=bool, default=True)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--crop_scale', type=float, default=0.1)
    parser.add_argument('--horizontal_flip', type=bool, default=True)
    parser.add_argument('--vertical_flip', type=bool, default=True)
    parser.add_argument('--brightness_contrast', type=bool, default=True)

    return parser.parse_args()


if __name__ == "__main__":

    config = parse_args()

    # train model
    engine = Engine(config)
    engine.train()