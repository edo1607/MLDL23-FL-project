import copy
import numpy as np
import torch

from collections import defaultdict
from tqdm import tqdm

from utils.selfTrainer import SelfTrainer
from utils.checkpointSaver import CheckpointSaver



class Server:

    def __init__(self, args, model, train_clients, metric):
        self.args = args
        self.train_clients = train_clients
        self.model = model
        self.metric = metric
        if self.args.step == '4':
            self.self_trainer = SelfTrainer(conf_th=args.conf_th, fraction=args.fraction)
            self.self_trainer.set_teacher(self.model)
        else:
            self.self_trainer = None

    def select_clients(self):
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        return np.random.choice(self.train_clients, num_clients, replace=False) 

    # updates is a list of dictionaries containing parameters obtained locally
    def train_round(self, clients):
        """
        This method trains the model with the dataset of the clients. It handles the training at single round level
        :param clients: list of all the clients to train
        :return: model updates gathered from the clients, to be aggregated
        """
        updates = []
        total_samples = 0
        weights = []
        for client in clients:
            n_samples, update = client.train(self.self_trainer)
            total_samples += n_samples
            updates.append(update)
            weights.append(n_samples)
        weights = [w/total_samples for w in weights]
        return updates, weights

    def aggregate(self, updates, weights):
        """
        This method handles the FedAvg aggregation
        :param updates: updates received from the clients
        :return: aggregated parameters
        """
        params = defaultdict(lambda: 0)
        for i, update in enumerate(updates):
            for key in update:
                params[key] += update[key]*weights[i]
        return params

    def train(self):
        """
        This method orchestrates the training the evals and tests at rounds level
        """
        # loss = self.eval_train()
        # self.args.wandb.log({'train': self.metric.results | {'loss': loss}})

        # initialize checkpoints saver
        checkpoint_saver = CheckpointSaver(dirpath='./saved_models', args=self.args, decreasing=False, top_n=1)

        for r in tqdm(range(self.args.num_rounds), total=self.args.num_rounds):
            if self.self_trainer is not None and self.args.T != 0:
                if r % self.args.T == 0:
                    self.self_trainer.set_teacher(self.model)
            # select randomly clients to consider
            selected_clients = self.select_clients()
            # use these clients to train local networks
            updates, weights = self.train_round(selected_clients)
            # aggregate local parameters learned
            new_parameters = self.aggregate(updates, weights)
            # update the model learned for the clients
            for client in self.train_clients:
                client.set_parameters(new_parameters)
            
            self.metric.reset()
            loss = self.eval_train(r, checkpoint_saver)
            self.args.wandb.log({'train': self.metric.results | {'loss': loss}}, step=r+1)

        # load parameters learned into the server model
        self.model.load_state_dict(new_parameters)

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
  

    def test(self, test_client, test_metric):
        """
        This method handles the test on the test clients
        :param test_client: the client to be tested
        :param test_metric: StreamMetric object
        :return: the average loss
        """
        n_samples, cumulative_loss = test_client.test(test_metric)
        test_metric.get_results()
        
        return cumulative_loss / n_samples





