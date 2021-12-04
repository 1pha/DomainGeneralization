import logging
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, roc_auc_score
import wandb

import torch
import torch.nn as nn
import timm

from pytorch_adapt.containers import Models, Optimizers
from pytorch_adapt.datasets import (
    DataloaderCreator,
    SourceDataset,
    CombinedSourceAndTargetDataset,
)
from pytorch_adapt.hooks import DANNHook
from pytorch_adapt.models import Discriminator, Classifier
from pytorch_adapt.utils.common_functions import batch_to_device
from pytorch_adapt.validators import IMValidator

from .dataset import TargetDataset
from util import utils

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Trainer:
    def __init__(self, src, tgt, wandb=True):

        set_seed(42)
        self.dataloaders = self.load_dataset(src, tgt)
        self.model_setup()
        self.wandb = wandb

    def load_dataset(self, src, tgt):

        """
        Will return a dictionary of -
            src_train
            src_val
            target_train
            target_val
            train: Concat of src_train and target_train
        """

        logger.info("Load PyTorch-Adapt Dataset.")

        src_train_dataset = SourceDataset(utils.get_train_loader(src).dataset)
        src_test_dataset = SourceDataset(utils.get_test_loader(src).dataset)
        tgt_train_dataset = TargetDataset(utils.get_train_loader(tgt).dataset)
        tgt_test_dataset = TargetDataset(utils.get_test_loader(tgt).dataset)

        datasets = {
            "src_train": src_train_dataset,
            "src_val": src_test_dataset,
            "target_train": tgt_train_dataset,
            "target_val": tgt_test_dataset,
            "train": CombinedSourceAndTargetDataset(
                source_dataset=src_train_dataset, target_dataset=tgt_train_dataset
            ),
        }
        dc = DataloaderCreator(batch_size=32, num_workers=2)
        dataloaders = dc(**datasets)
        logger.info(
            f"Successfully loaded PyTorch-Adapt Dataset of source={src} and target={tgt}"
        )
        return dataloaders

    def model_setup(self):

        logger.info("Setup model.")

        self.device = torch.device("cuda")

        G = timm.create_model("mobilenetv3_rw", pretrained=True).to(self.device)
        G.classifier = nn.Identity()

        C = Classifier(in_size=1280, num_classes=10).to(self.device)
        D = Discriminator(in_size=1280, h=256).to(self.device)

        self.models = Models({"G": G, "C": C, "D": D})

        optimizers = Optimizers((torch.optim.Adam, {"lr": 1e-4}))
        optimizers.create_with(self.models)
        self.optimizers = list(optimizers.values())

        self.hook = DANNHook(self.optimizers)
        self.validator = IMValidator()
        logger.info("Successfully setup model.")

    def train(self):

        self.models.train()
        logits, labels = [], []
        for data in tqdm(self.dataloaders["train"]):
            data = batch_to_device(data, self.device)
            loss, result = self.hook({}, {**self.models, **data})
            logits.append(nn.Softmax(dim=1)(result["src_imgs_features_logits"]))
            labels.append(data["src_labels"])

            del data
            torch.cuda.empty_cache()

        logits = torch.cat(logits, dim=0).detach().cpu().numpy()
        labels = torch.cat(labels, dim=0).detach().cpu().numpy()

        metric = self.get_metric(labels, logits)
        metric["train_loss"] = loss

        return metric

    def valid(self):

        self.models.eval()
        logits, labels = [], []
        with torch.no_grad():
            for data in tqdm(self.dataloaders["target_train"]):
                data = batch_to_device(data, self.device)
                logits.append(
                    nn.Softmax(dim=1)(
                        self.models["C"](self.models["G"](data["target_imgs"]))
                    )
                )
                labels.append(data["target_labels"])

                del data
                torch.cuda.empty_cache()

            logits = torch.cat(logits, dim=0).detach().cpu().numpy()
            labels = torch.cat(labels, dim=0).detach().cpu().numpy()

        metric = self.get_metric(labels, logits, split="target")

        return metric

    def run(self, epoch):

        self.loss = []
        self.valid_logits = []

        for e in range(epoch):

            train_result = self.train()
            valid_result = self.valid()

            if self.wandb:
                wandb.log(train_result, commit=False)
                wandb.log(valid_result)

    def get_metric(self, y_true=None, y_pred=None, split="train"):

        if y_true is None and y_pred is None:
            return {f"{split}_acc": 0, f"{split}_auroc": 0}
        else:
            return {
                f"{split}_acc": accuracy_score(y_true, y_pred.argmax(axis=1)),
                f"{split}_auroc": roc_auc_score(y_true, y_pred, multi_class="ovr"),
            }

    def retrieve_logit(self, name):

        self.models.eval()
        embeddings = []
        for data in tqdm(self.dataloaders[name]):
            data = batch_to_device(data, self.device)
            embeddings.append(self.models["G"](data["target_imgs"]))

        embeddings = torch.cat(embeddings, dim=0)
        return embeddings.detach().cpu().numpy()

    def visualize(self, split="train"):

        src_embed = self.retrieve_logit(f"src_{split}")
        tgt_embed = self.retrieve_logit(f"target_{split}")

        tsne = TSNE()
        src_embed = tsne.fit_transform(src_embed)
        tgt_embed = tsne.fit_transform(tgt_embed)

        plt.scatter(src_embed[:, 0], src_embed[:, 1], label="Source")
        plt.scatter(tgt_embed[:, 0], tgt_embed[:, 1], label="Target")
        plt.show()


if __name__ == "__main__":

    import wandb

    # wandb.init(project="Domain-Generalization")

    domains = ["Art", "Clipart", "Product", "Real World"]
    trainer = Trainer(domains[0], domains[1], False)

    trainer.run(10)
