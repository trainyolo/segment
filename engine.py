import os
import shutil
import torch
from datasets import get_dataset
from models import get_model
from losses import get_loss
from utils.meters import AverageMeter
from utils.transforms import get_train_transform, get_val_transform
from utils.iou_evaluator import IOUEvaluator
from utils.visualizer import SegmentationVisualizer
from tqdm import tqdm
import yaml
from utils.logging import Logger


class Engine:
    def __init__(self, config, device=None):
        self.config = config
        self.device = (
            device
            if device
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        # get categories from dataset
        with open(os.path.join(config.dataset_path, "dataset.yaml"), "r") as f:
            dataset_info = yaml.load(f, Loader=yaml.FullLoader)
            config.categories = dataset_info["names"]
            config.num_categories = len(dataset_info["names"])

        # create output dir
        if config.save:
            os.makedirs(config.save_path, exist_ok=True)

        # train/val dataloaders
        self.train_dataset_it, self.val_dataset_it = self.get_dataloaders(
            config, self.device
        )

        # model
        self.model = self.get_model(config, self.device)

        # loss
        self.loss_fn = self.get_loss(config)

        # optimizer/scheduler
        self.optimizer, self.scheduler = self.get_optimizer_and_scheduler(
            config, self.model
        )

        # visualizer
        self.visualizer = SegmentationVisualizer()

        # loggers
        self.logger = Logger(["train_loss", "val_loss", "miou"], "loss")
        self.iou_logger = Logger(config.categories.values(), "iou")

    @staticmethod
    def get_dataloaders(config, device):
        train_transform = get_train_transform(config)
        train_dataset = get_dataset(
            "SegmentationDataset",
            location=config.dataset_path,
            split="TRAIN",
            fake_size=config.dataset_size,
        )
        train_dataset.transform = train_transform
        train_dataset_it = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=config.train_workers,
            pin_memory=True if device.type == "cuda" else False,
        )

        # val dataloader
        val_transform = get_val_transform(config)
        val_dataset = get_dataset(
            "SegmentationDataset", location=config.dataset_path, split="VAL"
        )
        val_dataset.transform = val_transform
        val_dataset_it = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.val_workers,
            pin_memory=True if device.type == "cuda" else False,
        )

        return train_dataset_it, val_dataset_it

    @staticmethod
    def get_model(config, device):
        model = get_model(
            config.model,
            encoder_name=config.model_encoder,
            classes=config.num_categories,
        ).to(device)

        # load checkpoint
        if config.pretrained_model_path is not None and os.path.exists(
            config.pretrained_model_path
        ):
            print(f"Loading model from {config.pretrained_model_path}")
            state = torch.load(config.pretrained_model_path)
            model.load_state_dict(state["model_state_dict"], strict=True)

        return model

    @staticmethod
    def get_loss(config):
        if config.num_categories > 1:
            loss_fn = get_loss("cedice", ignore_index=255)
        else:
            loss_fn = get_loss("bcedice", ignore_index=255)

        return loss_fn

    @staticmethod
    def get_optimizer_and_scheduler(config, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=config.patience, verbose=True
        )

        return optimizer, scheduler

    def display(self, pred, sample):
        # display a single image
        image = sample["image"][0]
        pred = (
            (pred[0] > 0).cpu()
            if self.config.num_categories == 1
            else torch.argmax(pred[0], dim=0).cpu()
        )
        gt = sample["label"][0]

        self.visualizer.display(image, pred, gt)

    def forward(self, sample):
        images = sample["image"].to(self.device)
        labels = sample["label"].to(self.device)

        if self.config.num_categories == 1:
            labels = (labels > 0).unsqueeze(1).float()
        else:
            labels = labels.long()

        pred = self.model(images)
        loss = self.loss_fn(pred, labels)

        return pred, loss

    def train_step(self):
        config = self.config

        # define meters
        loss_meter = AverageMeter()

        self.model.train()
        for i, sample in enumerate(tqdm(self.train_dataset_it)):
            pred, loss = self.forward(sample)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item())

            if config.display and i % config.display_it == 0:
                with torch.no_grad():
                    self.display(pred, sample)

        return loss_meter.avg

    def val_step(self):
        config = self.config

        # define meters
        loss_meter = AverageMeter()
        iou_meter = IOUEvaluator(
            config.num_categories + 1
            if config.num_categories == 1
            else config.num_categories
        )

        self.model.eval()
        with torch.no_grad():
            for i, sample in enumerate(tqdm(self.val_dataset_it)):
                pred, loss = self.forward(sample)
                loss_meter.update(loss.item())

                if config.display and i % config.display_it == 0:
                    self.display(pred, sample)

                # compute iou
                labels = sample["label"].to(self.device)
                labels = labels.unsqueeze(1).long()

                if config.num_categories == 1:  # binary
                    labels = (labels > 0).long()
                    iou_meter.addBatch((pred > 0).long(), labels)
                else:
                    iou_meter.addBatch(pred.argmax(dim=1, keepdim=True), labels)

        # get iou metric
        miou, iou = iou_meter.getIoU()
        if config.num_categories == 1:
            metrics = {"miou": iou[1], "class_iou": iou[1:]}
        else:
            metrics = {"miou": miou, "class_iou": iou}

        return loss_meter.avg, metrics

    def save_checkpoint(
        self,
        epoch,
        is_best_val=False,
        best_val_loss=0,
        is_best_miou=False,
        metrics={},
    ):
        config = self.config

        state = {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }

        print("=> saving checkpoint")
        file_name = os.path.join(config.save_path, "checkpoint.pth")
        torch.save(state, file_name)

        if is_best_val:
            print("=> saving best_val checkpoint")
            shutil.copyfile(
                file_name, os.path.join(config.save_path, "best_val_model.pth")
            )

        if is_best_miou:
            print("=> saving best_miou checkpoint")
            shutil.copyfile(
                file_name, os.path.join(config.save_path, "best_miou_model.pth")
            )
            with open(
                os.path.join(self.config.save_path, "best_miou_model.csv"), "w"
            ) as f:
                f.write(",".join(["miou", *self.config.categories.values()]) + "\n")
                f.write(
                    ",".join(
                        [
                            f"{item:.05f}"
                            for item in [metrics["miou"], *metrics["class_iou"]]
                        ]
                    )
                )

    def trace_model(self):
        config = self.config

        # load best ap model
        model = self.get_model(config, torch.device("cpu"))
        state = torch.load(
            os.path.join(config.save_path, "best_miou_model.pth"), map_location="cpu"
        )
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.eval()

        # trace model
        print("Tracing best model")
        traced_model = torch.jit.trace(
            model, torch.randn(1, 3, config.img_size, config.img_size)
        )
        traced_model.save(os.path.join(config.save_path, "best_miou_model.pt"))

    def train(self):
        best_val_loss = float("inf")
        best_miou = 0

        # for epoch in range(config.solver.num_epochs):
        epoch = 0
        while True:
            print(f"Starting epoch {epoch}")

            train_loss = self.train_step()
            val_loss, metrics = self.val_step()

            print(f"==> train loss: {train_loss}")
            print(f"==> val loss: {val_loss}")
            print(f"metrics: {metrics}")

            self.logger.add("train_loss", train_loss)
            self.logger.add("val_loss", val_loss)
            self.logger.add("miou", metrics["miou"])
            self.logger.save(save_dir=self.config.save_path)

            [
                self.iou_logger.add(k, v)
                for k, v in zip(self.config.categories.values(), metrics["class_iou"])
            ]
            self.iou_logger.save(save_dir=self.config.save_path)

            is_best_val = val_loss < best_val_loss
            best_val_loss = min(val_loss, best_val_loss)

            is_best_miou = metrics["miou"] > best_miou
            best_miou = max(metrics["miou"], best_miou)

            self.save_checkpoint(
                epoch,
                is_best_val=is_best_val,
                best_val_loss=best_val_loss,
                is_best_miou=is_best_miou,
                metrics=metrics,
            )

            self.scheduler.step(val_loss)
            epoch = epoch + 1

            if (
                self.optimizer.param_groups[0]["lr"] < self.config.lr / 100
            ) or epoch == self.config.max_epochs:
                break

        self.trace_model()
