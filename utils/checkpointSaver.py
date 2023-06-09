import numpy as np 
import os
import logging
import torch


class CheckpointSaver:
    def __init__(self, dirpath, args, decreasing=True, top_n=5):
        """
        dirpath: Directory path where to store all model weights 
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value
        """
        if not os.path.exists(dirpath): os.makedirs(dirpath)
        self.dirpath = dirpath
        self.top_n = top_n 
        self.decreasing = decreasing
        self.top_model_paths = []
        self.best_metric_val = np.Inf if decreasing else -np.Inf
        self.args = args

    # serve per salvare il modello se è meglio
    def __call__(self, model, metric_val, epoch, is_model = True):
        if is_model:
            model_path = os.path.join(self.dirpath, f'{self.args.step}_model_{self.args.wandb.name}_epoch{epoch}_{metric_val:.3f}.pt')
        else:
            model_path = os.path.join(self.dirpath, f'{self.args.step}_discriminator_{self.args.wandb.name}_epoch{epoch}_{metric_val:.3f}.pt')
        # check if model is better than the previous ones saved (save is Bool)
        save = metric_val<self.best_metric_val if self.decreasing else metric_val>self.best_metric_val
        if save:
            self.best_metric_val = metric_val
            torch.save(model.state_dict(), model_path)
            self.top_model_paths.append({'path': model_path, 'score': metric_val})
            self.top_model_paths = sorted(self.top_model_paths, key=lambda o: o['score'], reverse=not self.decreasing)
        # checks if the total number of models saved is more than top_n,
        # in that case, the checkpoint saver cleans the extra models that have a lower performance. 
        if len(self.top_model_paths)>self.top_n: 
            self.cleanup()
        
    def cleanup(self):
        to_remove = self.top_model_paths[self.top_n:]
        for o in to_remove:
            os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[:self.top_n]



