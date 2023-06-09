import copy
import torch

from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, LinearLR, ExponentialLR, CosineAnnealingLR, ConstantLR

from utils.loss_utils import HardNegativeMining, MeanReduction, KnowledgeDistillationLoss


class Client:

    def __init__(self, args, dataset, model, test_client=False):
        self.args = args
        self.dataset = dataset
        self.name = self.dataset.client_name
        self.model = model
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True, drop_last=True) \
            if not test_client else None
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()
        if self.args.step == '4':
            self.kd_loss = KnowledgeDistillationLoss(reduction='mean')
            self.lam_kd = self.args.lam_kd

    def __str__(self):
        return self.name

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

    def run_epoch(self, cur_epoch, optimizer, self_trainer=None):
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        :param self_trainer: SelfTrainer class instance to use pseudo labels instead of the actual ones
        """
        for cur_step, (images, labels) in enumerate(self.train_loader):
            optimizer.zero_grad()
            loss = 0
            
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            outputs = self.get_outputs(images)
            if self_trainer is not None:
                # substitute actual labels with the pseudo ones
                labels, pred, mask = self_trainer.get_pseudolab_pred_mask(images)
                loss += self.lam_kd * self.kd_loss(outputs, pred, mask=mask)
            # the criterion used does not reduce the loss result
            # that means loss_full is a tensor representing the loss for each pixel
            loss_full = self.criterion(outputs, labels)
            # self.reduction is used to make loss_full a single number representing averall loss
            loss += self.reduction(loss_full, labels)

            # update the model parameters using backpropagation
            loss.backward()
            optimizer.step()

    def train(self, self_trainer=None):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :param self_trainer: SelfTrainer class instance to use pseudo labels instead of the actual ones
        :return: length of the local dataset, copy of the model parameters
        """
        # set model in training mode
        self.model.train()
        # use SGD as optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.m, weight_decay=self.args.wd)
        scheduler = self.get_scheduler(optimizer, self.args.schedule)
        for epoch in range(self.args.num_epochs):
            self.run_epoch(epoch, optimizer, self_trainer)
            scheduler.step()

        return len(self.dataset), copy.deepcopy(self.model.state_dict())

    def test(self, metric):
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        # set model in evaluation mode
        self.model.eval()
        cumulative_loss = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                outputs = self.get_outputs(images)
                loss_full = self.criterion(outputs, labels)
                loss = self.reduction(loss_full, labels)
                cumulative_loss += loss
                self.update_metric(metric, outputs, labels)
        metric.get_results()
        return len(self.dataset), cumulative_loss

    def set_parameters(self, parameters):
        self.model.load_state_dict(parameters)

    def get_scheduler(self, optimizer, schedule):
        if schedule == 'cosine':
            # eta_min so that the last one is not zero
            scheduler = CosineAnnealingLR(optimizer, T_max=self.args.num_epochs, eta_min=0.0005)
        elif schedule == 'step':
            scheduler = StepLR(optimizer, step_size=2, gamma=0.1)
        elif schedule == 'linear':
            scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0001, total_iters=self.args.num_epochs)
        elif schedule == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.8)
        elif schedule == 'none':
            scheduler = ConstantLR(optimizer, factor=1, total_iters=self.args.num_epochs)
        
        return scheduler
