import copy
import numpy as np

from server import Server
from utils.checkpointSaver import CheckpointSaver
from utils.advTrainer import AdvTrainer
from tqdm import tqdm



class AdvServer(Server):

    def __init__(self, args, model, discriminator, train_clients, metric):
        super().__init__(args, model, train_clients, metric)
        self.discriminator = discriminator
        if args.step == '5c':
            self.self_trainer = AdvTrainer(discriminator, conf_th=args.conf_th, fraction=args.fraction)
            self.self_trainer.set_teacher(self.model)


    # updates is a list of dictionaries containing parameters obtained locally
    def train_round(self, clients):
        """
        This method trains the model with the dataset of the clients. It handles the training at single round level
        :param clients: list of all the clients to train
        :return: model updates gathered from the clients, to be aggregated
        """
        updates_model = []
        updates_discriminator = []
        total_samples = 0
        weights = []
        for client in clients:
            n_samples, update_model, update_discriminator = client.train(self.self_trainer)
            total_samples += n_samples
            updates_model.append(update_model)
            updates_discriminator.append(update_discriminator)
            weights.append(n_samples)
        weights = [w/total_samples for w in weights]
        return updates_model, updates_discriminator, weights

    def train(self):
        """
        This method orchestrates the training the evals and tests at rounds level
        """

        # initialize checkpoints saver
        checkpoint_saver = CheckpointSaver(dirpath='./saved_models', args=self.args, decreasing=False, top_n=3)
        
        # loss = self.eval_train()
        # self.args.wandb.log({'train': self.metric.results | {'loss': loss}})
        for r in tqdm(range(self.args.num_rounds), total=self.args.num_rounds):
            if self.self_trainer is not None and self.args.T != 0:
                if r % self.args.T == 0:
                    self.self_trainer.set_teacher(self.model)
            # select randomly clients to consider
            selected_clients = self.select_clients()
            # use these clients to train local networks
            updates_model, updates_discriminator, weights = self.train_round(selected_clients)
            # aggregate local parameters learned
            new_parameters_model = self.aggregate(updates_model, weights)
            new_parameters_discriminator = self.aggregate(updates_discriminator, weights)


            # update the models learned for the clients
            for client in self.train_clients:
                client.set_parameters(new_parameters_model)
                client.set_parameters_discriminator(new_parameters_discriminator)
            
            self.metric.reset()
            loss = self.eval_train(r, checkpoint_saver)
            self.args.wandb.log({'train': self.metric.results | {'loss': loss}}, step=r+1)

        # load parameters learned into the server
        self.model.load_state_dict(new_parameters_model)
        self.discriminator.load_state_dict(new_parameters_discriminator)

    def eval_train(self, r=None, checkpoint_saver= None):
        """
        This method handles the evaluation on the train clients
        :return: the average loss
        """
        total_samples = 0
        total_loss = 0
        for client in self.train_clients:
            n_samples, cumulative_loss = client.test(self.metric)
            total_samples += n_samples
            total_loss += cumulative_loss
        loss = cumulative_loss / total_samples
        self.metric.get_results()
        if checkpoint_saver is not None:
            checkpoint_saver(self.model, self.metric.results["Mean IoU"], r+1)
            

        return loss

