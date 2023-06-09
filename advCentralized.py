import random
import torch.nn.functional as F
import torch
import utils.style_transfer as st

from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from torch.nn import Softmax2d, LogSoftmax
from torch.nn import NLLLoss, CrossEntropyLoss

from utils.checkpointSaver import CheckpointSaver
from fdaCentralized import FdaCentralized



class AdvCentralized(FdaCentralized):

    def __init__(self, args, generator, discriminator, training_dataset, metric, clients=None, b=None, L=None):
        super().__init__(args, generator, training_dataset, metric, clients=clients, b=b, L=L)
        self.discriminator = discriminator

        self.optimizerG = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.m, weight_decay=args.wd)
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=args.lr, betas=(0.9, 0.99))

        self.loss_function = NLLLoss(ignore_index=255)
        self.cross_entropy_loss = CrossEntropyLoss(ignore_index=255, reduction='none')

    def run_epoch(self, cur_epoch, eval_datasets=None, eval_metric=None):
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        :return: the average loss of the epoch
        """
        lossD = 0
        total_samples = 0
        i = 0
        for batch_id, (images, labels) in enumerate(self.train_loader):

            images, labels = images.to(self.args.device), labels.to(self.args.device)

            total_samples += self.args.bs

            ohlabels = labels.detach().clone()
            # one-hot encoded label
            ohlabels[ohlabels == 255] = 15
            # print(label.unique())
            ohlabels = torch.nn.functional.one_hot(ohlabels).permute((0, 3, 1, 2))

            with torch.no_grad():
                outputs = self.get_outputs(images)
                cpmap = Softmax2d()(outputs)

            N, _, H, W = cpmap.size()

            # Generate the Real and Fake Labels
            targetf = torch.zeros((N,H,W), dtype=torch.long, requires_grad=False).to(self.args.device)
            targetr = torch.ones((N,H,W), dtype=torch.long, requires_grad=False).to(self.args.device)

            ##########################
            # DISCRIMINATOR TRAINING #
            ##########################
            self.optimizerD.zero_grad()

            # Train on Real
            confr = LogSoftmax(dim=1)(self.discriminator(ohlabels.float()))
            # compute D loss on real
            LDr = NLLLoss(ignore_index=15)(confr,targetr)
            # compute gradient
            LDr.backward()

            # Train on Fake
            conff = LogSoftmax(dim=1)(self.discriminator(cpmap.data))
            # compute D loss on G output (fake)
            LDf = self.loss_function(conff,targetf)
            # compute gradient
            LDf.backward()
            

            self.optimizerD.step()

            lossD += LDr + LDf

            ######################
            # GENERATOR TRAINING #
            ######################
            self.optimizerG.zero_grad()

            outputs = self.get_outputs(images)
            cpmap = Softmax2d()(outputs)
            cpmaplsmax = LogSoftmax(dim=1)(outputs)
            conff = LogSoftmax(dim=1)(self.discriminator(cpmap))


            LGce = self.loss_function(cpmaplsmax,labels)
            LGadv = self.loss_function(conff,targetr)
            LGseg = LGce + self.args.lam_adv *LGadv

            LGseg.backward()
            
            self.optimizerG.step()

            self.update_metric(self.metric, outputs, labels)

        return lossD / len(self.train_loader)

    def train(self, eval_datasets=None, eval_metric=None, save=True):

        schedulerG = self.get_scheduler(self.optimizerG, self.args.schedule)
        schedulerD = self.get_scheduler(self.optimizerD, self.args.schedule)

        # set G in training mode
        self.model.train()
        self.discriminator.train()

        validate = False
        if eval_datasets is not None and eval_metric is not None:
            validate = True
            if save:
                # initialize checkpoints saver
                checkpoint_saver = CheckpointSaver(dirpath='./saved_models', args=self.args, decreasing=False, top_n=1)
                checkpoint_saver_D = CheckpointSaver(dirpath='./saved_models', args=self.args, decreasing=True, top_n=1)

        for epoch in tqdm(range(self.args.num_epochs), total=self.args.num_epochs):

            self.metric.reset()
            lossD = self.run_epoch(epoch, eval_datasets, eval_metric)

            self.metric.get_results()
            self.args.wandb.log({'train': self.metric.results}, commit=not validate, step=epoch+1)

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
                    checkpoint_saver_D(self.discriminator, lossD, epoch+1, is_model = False)
                # set the model again in train mode
                self.model.train()

            schedulerD.step()
            schedulerG.step()












        


