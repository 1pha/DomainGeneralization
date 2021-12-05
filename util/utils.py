import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import params
from sklearn.manifold import TSNE


import matplotlib.pyplot as plt

plt.switch_backend("agg")

import numpy as np
import os, time
from Dataset import OfficeHome


def get_train_loader(domain):
    """
    Get train dataloader of source domain or target domain
    :return: dataloader
    """

    # transform = transforms.Compose([
    #     transforms.RandomCrop((64)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean= params.dataset_mean, std= params.dataset_std)
    # ])

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std),
        ]
    )

    data = OfficeHome(split="train", transform=transform, domain=domain)

    dataloader = DataLoader(dataset=data, batch_size=params.batch_size, shuffle=True)

    return dataloader


def get_test_loader(domain):
    """
    Get test dataloader of source domain or target domain
    :return: dataloader
    """
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std),
        ]
    )

    data = OfficeHome(split="valid", transform=transform, domain=domain)

    dataloader = DataLoader(dataset=data, batch_size=params.batch_size, shuffle=False)

    return dataloader


def optimizer_scheduler(optimizer, p):
    """
    Adjust the learning rate of optimizer
    :param optimizer: optimizer for updating parameters
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = 1e-2 / (1.0 + 10 * p) ** 0.75

    return optimizer


def displayImages(dataloader, length=8, imgName=None):
    """
    Randomly sample some images and display
    :param dataloader: maybe trainloader or testloader
    :param length: number of images to be displayed
    :param imgName: the name of saving image
    :return:
    """
    if params.fig_mode is None:
        return

    # randomly sample some images.
    dataiter = iter(dataloader)
    images, labels = dataiter.next()

    # process images so they can be displayed.
    images = images[:length]

    images = torchvision.utils.make_grid(images).numpy()
    images = images / 2 + 0.5
    images = np.transpose(images, (1, 2, 0))

    if params.fig_mode == "display":

        plt.imshow(images)
        plt.show()

    if params.fig_mode == "save":
        # Check if folder exist, otherwise need to create it.
        folder = os.path.abspath(params.save_dir)

        if not os.path.exists(folder):
            os.makedirs(folder)

        if imgName is None:
            imgName = "displayImages" + str(int(time.time()))

        # Check extension in case.
        if not (
            imgName.endswith(".jpg")
            or imgName.endswith(".png")
            or imgName.endswith(".jpeg")
        ):
            imgName = os.path.join(folder, imgName + ".jpg")

        plt.imsave(imgName, images)
        plt.close()

    # print labels
    print(" ".join("%5s" % labels[j].item() for j in range(length)))


def plot_embedding(X, y, d, title=None, imgName=None):
    """
    Plot an embedding X with the class label y colored by the domain d.

    :param X: embedding
    :param y: label
    :param d: domain
    :param title: title on the figure
    :param imgName: the name of saving image

    :return:
    """
    if params.fig_mode is None:
        return

    # normalization
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)

    for i in range(X.shape[0]):
        # plot colored number
        plt.text(
            X[i, 0],
            X[i, 1],
            str(y[i]),
            color=plt.cm.bwr(d[i] / 1.0),
            fontdict={"weight": "bold", "size": 9},
        )

    plt.xticks([]), plt.yticks([])

    # If title is not given, we assign training_mode to the title.
    if title is not None:
        plt.title(title)
    else:
        plt.title(params.training_mode)

    if params.fig_mode == "display":
        # Directly display if no folder provided.
        plt.show()

    if params.fig_mode == "save":
        # Check if folder exist, otherwise need to create it.
        folder = os.path.abspath(params.save_dir)

        if not os.path.exists(folder):
            os.makedirs(folder)

        if imgName is None:
            imgName = "plot_embedding" + str(int(time.time()))

        # Check extension in case.
        if not (
            imgName.endswith(".jpg")
            or imgName.endswith(".png")
            or imgName.endswith(".jpeg")
        ):
            imgName = os.path.join(folder, imgName + ".jpg")

        print("Saving " + imgName + " ...")
        plt.savefig(imgName)
        plt.close()
