import torch
import numpy as np
import copy
import torch.nn.functional as F

from torch.nn import Softmax, Softmax2d



class AdvTrainer():
    def __init__(self, discriminator, conf_th=0.9, fraction=0.66, ignore_index=255):
        self.discriminator = discriminator
        self.conf_th = conf_th          
        self.fraction = fraction
        self.teacher = None
        self.ignore_index = ignore_index

    def set_teacher(self, model):
        self.teacher = model
        self.teacher.eval()

    def get_image_mask(self, prob, pseudo_lab):
        """
        For each semantic class, we accept the predictions with confidence that is 
        within the top self.fraction % (in the pseudo label) or above self.conf_th (in the prediction probabilities).
        :param prob: probability of real label given by the teacher model
        :param pseudo_lab: pseudo labels given by the teacher model
        :return: the image mask of the pixels to be considered for the loss
        """
        mask_prob = prob > self.conf_th if 0. < self.conf_th < 1. else torch.zeros(prob.size(), dtype=torch.bool).to(prob.device)
        
        mask_topk = torch.zeros(prob.size(), dtype=torch.bool).to(prob.device)
        if 0. < self.fraction < 1.:
            for c in pseudo_lab.unique():
                mask_c = pseudo_lab == c
                prob_c = prob.clone()
                prob_c[~mask_c] = 0
                _, idx_c = torch.topk(prob_c.flatten(), k=int(mask_c.sum() * self.fraction))
                mask_topk_c = torch.zeros_like(prob_c.flatten(), dtype=torch.bool)
                mask_topk_c[idx_c] = 1
                mask_c &= mask_topk_c.unflatten(dim=0, sizes=prob_c.size())
                mask_topk |= mask_c
        return mask_prob | mask_topk

    def get_batch_mask(self, prob, pseudo_lab):
        """
        :param prob: the batch of probabilities confidence for each pixel of the image
        :param pseudo_lab: the batch of pseudo labels predicted by the teacher
        :return: the batch of the images' mask
        """
        b, _, _ = prob.size()
        mask = torch.stack([self.get_image_mask(pb, pl) for pb, pl in zip(prob, pseudo_lab)], dim=0)
        return mask

    def get_pseudolab_pred_mask(self, imgs):
        """
        :param imgs: batch of images used by the self.teacher
        :return: batch of pseudo labels
        """
        if self.teacher is not None:
            with torch.no_grad():
                try:
                    pred = self.teacher(imgs)['out']
                except:
                    pred = self.teacher(imgs)
        pseudo_lab = torch.argmax(pred, dim=1)

        conf = self.discriminator(Softmax2d()(pred))
        prob = Softmax2d()(conf)[:,1,:,:]
        mask = self.get_batch_mask(prob, pseudo_lab)
        pseudo_lab[~mask] = self.ignore_index
        return pseudo_lab, pred, mask