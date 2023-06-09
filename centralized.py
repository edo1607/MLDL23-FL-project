import copy
import torch
import wandb
import os
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from torch import optim, nn
from collections import defaultdict
from collections import OrderedDict
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR, LinearLR, ExponentialLR, CosineAnnealingLR, ConstantLR
from copy import deepcopy
import torch.cuda.amp as amp

from utils.checkpointSaver import CheckpointSaver
from utils.loss_utils import HardNegativeMining, MeanReduction


class Centralized:

    def __init__(self, args, model, training_dataset, metric):
        self.args = args
        if training_dataset:
            self.training_dataset = training_dataset
            self.train_loader = DataLoader(training_dataset, batch_size=self.args.bs, shuffle=True, drop_last=True)
        self.model = model
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()
        self.metric = metric


    @staticmethod
    def update_metric(metric, outputs, labels):
        _, prediction = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        prediction = prediction.cpu().numpy()
        metric.update(labels, prediction)

    def get_outputs(self, images):
        if self.args.model == 'deeplabv3_mobilenetv2':
            return self.model(images)['out']
        if self.args.model == 'resnet18':
            return self.model(images)
        raise NotImplementedError

    def run_epoch(self, cur_epoch, optimizer):
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        :return: the average loss of the epoch
        """
        cumulative_loss = 0
        for cur_step, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            outputs = self.get_outputs(images)
            # the criterion used does not reduce the loss result
            # that means loss_full is a tensor representing the loss for each pixel
            loss_full = self.criterion(outputs, labels)
            # self.reduction is used to make loss_full a single number representing averall loss
            loss = self.reduction(loss_full, labels)

            cumulative_loss += loss

            self.update_metric(self.metric, outputs, labels)
            
            
            # update the model parameters using backpropagation

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return cumulative_loss/len(self.train_loader)

    def train(self, eval_datasets=None, eval_metric=None, save=True):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :param eval_datasets: if given, evaluates the loss at each epoch. It can also be a list of datasets representing the clients
        :param eval_metric: metric used for the evaluation dataset at each epoch
        :param save: choose whether or not to save the best model with respect to the evaluation mean IoU
        """

        # set model in training mode
        self.model.train()
        # use SGD as optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.m, weight_decay=self.args.wd)
        scheduler = self.get_scheduler(optimizer, self.args.schedule)

        validate = False
        if eval_datasets is not None and eval_metric is not None:
            validate = True
            if save:
                # initialize checkpoints saver
                checkpoint_saver = CheckpointSaver(dirpath='./saved_models', args=self.args, decreasing=False, top_n=1)

        for epoch in tqdm(range(self.args.num_epochs), total=self.args.num_epochs):
            self.metric.reset()
            loss = self.run_epoch(epoch, optimizer)
            self.metric.get_results()
            self.args.wandb.log({'train': self.metric.results | {'loss': loss}}, commit=not validate, step=epoch+1)

            if validate:
                eval_metric.reset()
                if isinstance(eval_datasets, list):
                    total_samples = 0
                    total_loss = 0
                    for ds in eval_datasets:
                        # note that eval_metric update itself for each dataset, without resetting
                        loss_eval = self.test(ds, eval_metric)
                        n_samples = len(ds)

                        total_samples += n_samples
                        total_loss += loss_eval * n_samples
                    loss_eval = total_loss / total_samples
                else:
                    loss_eval = self.test(eval_datasets, eval_metric)
                self.args.wandb.log({'eval': eval_metric.results | {'loss': loss_eval}}, commit=True, step=epoch+1)
                if save:
                    checkpoint_saver(self.model, eval_metric.results["Mean IoU"], epoch+1)
                # set the model again in train mode
                self.model.train()

            scheduler.step()      

    def test(self, test_dataset, test_metric):
        """
        This method tests the model on the local dataset of the client.
        :param test_dataset: the dataset to be tested
        :param test_metric: StreamMetric object
        :return: the average loss per input during the epoch
        """
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        # set model in evaluation mode
        self.model.eval()
        cumulative_loss = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                outputs = self.get_outputs(images)
                loss_full = self.criterion(outputs, labels)
                loss = self.reduction(loss_full, labels)
                cumulative_loss += loss
                self.update_metric(test_metric, outputs, labels)
        test_metric.get_results()
        return cumulative_loss/len(test_loader)

    def get_scheduler(self, optimizer, schedule):
        if schedule == 'cosine':
            # eta_min so that the last one is not zero
            scheduler = CosineAnnealingLR(optimizer, T_max=self.args.num_epochs, eta_min=0.0005)
        elif schedule == 'step':
            scheduler = StepLR(optimizer, step_size=2, gamma=0.1)
        elif schedule == 'linear':
            scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0001, total_iters=self.args.num_epochs)
        elif schedule == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.895)
        elif schedule == 'none':
            scheduler = ConstantLR(optimizer, factor=1, total_iters=self.args.num_epochs)
        
        return scheduler
