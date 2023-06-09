import copy
import torch

from client import Client
from torch import optim
from torch.nn import Softmax2d, LogSoftmax
from torch.nn import NLLLoss
from tqdm import tqdm
from utils.loss_utils import KnowledgeDistillationLoss



class AdvClient(Client):

    def __init__(self, args, dataset, model, discriminator, test_client=False):
        super().__init__(args, dataset, model, test_client)
        self.discriminator = discriminator
        self.optimizerG = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.m, weight_decay=args.wd)
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=args.lr, betas=(0.9, 0.99))
        self.loss_function = NLLLoss(ignore_index=255)
        if self.args.step == '5c':
            self.kd_loss = KnowledgeDistillationLoss(reduction='mean', alpha=self.args.alpha_kd)


    def run_epoch(self, cur_epoch, self_trainer=None):
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        :param self_trainer: SelfTrainer class instance to use pseudo labels instead of the actual ones
        """

        for batch_id, (images, labels) in enumerate(self.train_loader):

            images, labels = images.to(self.args.device), labels.to(self.args.device)
            
            if self_trainer is not None:
                # no discriminator training
                # thus note that the self.discriminator is the same of the centralized one
                ######################
                # GENERATOR TRAINING #
                ######################
                self.optimizerG.zero_grad()

                outputs = self.get_outputs(images)
                cpmap = Softmax2d()(outputs)
                cpmaplsmax = LogSoftmax(dim=1)(outputs)

                # substitute actual labels with the pseudo ones
                labels, pred, mask = self_trainer.get_pseudolab_pred_mask(images)

                # semi supervised loss
                LGsemi = self.loss_function(cpmaplsmax,labels)
                
                # knowledge distillation loss
                LGkd = self.kd_loss(outputs, pred, mask=mask)

                conf = LogSoftmax(dim=1)(self.discriminator(cpmap))

                N, _, H, W = cpmap.size()
                targetr = torch.ones((N,H,W), dtype=torch.long).to(self.args.device)
                # adversarial loss
                LGadv = self.loss_function(conf, targetr)

                self.args.wandb.log({'LGsemi': LGsemi}, commit=False, step=cur_epoch+1)
                self.args.wandb.log({'LGkd': LGkd}, commit=False, step=cur_epoch+1)
                self.args.wandb.log({'LGadv': LGadv}, commit=True, step=cur_epoch+1)

                print(f"""
                  {LGsemi =}
                  {LGkd   =}
                  {LGadv  =}
                  ----------
                """)


                LGseg = LGsemi + self.args.lam_adv * LGadv + self.args.lam_kd * LGkd

                LGseg.backward()
                
                self.optimizerG.step()

            else: # train with ground truth labels

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
                LDr = self.loss_function(confr,targetr)
                # compute gradient
                LDr.backward()

                # Train on Fake
                conff = LogSoftmax(dim=1)(self.discriminator(cpmap.data))
                # compute D loss on G output (fake)
                LDf = self.loss_function(conff,targetf)
                # compute gradient
                LDf.backward()

                self.optimizerD.step()

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


    def train(self, self_trainer=None):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :param self_trainer: SelfTrainer class instance to use pseudo labels instead of the actual ones
        :return: length of the local dataset, copy of the model parameters
        """
        schedulerG = self.get_scheduler(self.optimizerG, self.args.schedule)
        schedulerD = self.get_scheduler(self.optimizerD, self.args.schedule)

        # set models in training mode
        self.model.train()
        self.discriminator.train()

        for epoch in range(self.args.num_epochs):

            self.run_epoch(epoch, self_trainer)

            schedulerD.step()
            schedulerG.step()

        return len(self.dataset), copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.discriminator.state_dict())

    def set_parameters_discriminator(self, parameters):
        self.discriminator.load_state_dict(parameters)
