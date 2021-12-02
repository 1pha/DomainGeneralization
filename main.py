"""
Main script for models
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import torch.nn as nn
import torch.optim as optim

import numpy as np

from models import models
from train import test, train, params
from util import utils
from sklearn.manifold import TSNE

import argparse, sys, os

import torch
from torch.autograd import Variable

import time

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def main(args):

    # Set global parameters.
    params.fig_mode = args.fig_mode
    params.epochs = args.max_epoch
    params.training_mode = args.training_mode
    params.source_domain = args.source_domain
    params.target_domain = args.target_domain
    params.embed_plot_epoch = args.embed_plot_epoch
    params.lr = args.lr


    if args.save_dir is not None:
        params.save_dir = args.save_dir
    else:
        print('Figures will be saved in ./experiment folder.')

    # prepare the source data and target data

    src_train_dataloader = utils.get_train_loader(params.source_domain)
    src_test_dataloader = utils.get_test_loader(params.source_domain)
    tgt_train_dataloader = utils.get_train_loader(params.target_domain)
    tgt_test_dataloader = utils.get_test_loader(params.target_domain)

    # init models
    feature_extractor = models.Extractor()
    class_classifier = models.Class_classifier()
    domain_classifier = models.Domain_classifier()

    if params.use_gpu:
        feature_extractor.cuda()
        class_classifier.cuda()
        domain_classifier.cuda()

    # init criterions
    class_criterion = nn.NLLLoss()
    domain_criterion = nn.NLLLoss()

    # init optimizer
    optimizer = optim.SGD([{'params': feature_extractor.parameters()},
                            {'params': class_classifier.parameters()},
                            {'params': domain_classifier.parameters()}], lr= params.lr, momentum=0.9)

    for epoch in range(params.epochs):
        print('Epoch: {}'.format(epoch))
        train.train(args.training_mode, feature_extractor, class_classifier, domain_classifier, class_criterion, domain_criterion,
                    src_train_dataloader, tgt_train_dataloader, optimizer, epoch)
        test.test(feature_extractor, class_classifier, domain_classifier, src_test_dataloader, tgt_test_dataloader)



def parse_arguments(argv):
    """Command line parse."""
    domains = ['Art', 'Clipart', 'Product', 'Real World']
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_domain', type= str, default= 'Art', help= 'Choose source domain.')

    parser.add_argument('--target_domain', type= str, default= 'Real World', help = 'Choose target domain.')

    parser.add_argument('--fig_mode', type=str, default=None, help='Plot experiment figures.')

    parser.add_argument('--save_dir', type=str, default=None, help='Path to save plotted images.')

    parser.add_argument('--training_mode', type=str, default='dann', help='Choose a mode to train the model.')

    parser.add_argument('--max_epoch', type=int, default=100, help='The max number of epochs.')

    parser.add_argument('--embed_plot_epoch', type= int, default=100, help= 'Epoch number of plotting embeddings.')

    parser.add_argument('--lr', type= float, default= 1e-2, help= 'Learning rate.')

    return parser.parse_args()



if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
