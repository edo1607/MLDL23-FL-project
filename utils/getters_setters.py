import os
import json
import torch
import random
import numpy as np
import copy

from torch import nn
from torchvision.models import resnet18

import datasets.ss_transforms as sstr
import datasets.np_transforms as nptr
from datasets.idda import IDDADataset
from datasets.gta import GTADataset

from client import Client
from advClient import AdvClient

from models.deeplabv3 import deeplabv3_mobilenetv2
from utils.stream_metrics import StreamSegMetrics, StreamClsMetrics



def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(random_seed)


def get_dataset_num_classes(dataset):
    if dataset == 'idda' or dataset == 'gta':
        return 16
    raise NotImplementedError


def model_init(args):
    if args.model == 'deeplabv3_mobilenetv2':
        return deeplabv3_mobilenetv2(num_classes=get_dataset_num_classes(args.dataset), device=args.device)
        raise NotImplementedError


def get_transforms(args):

    transforms = []
    if args.dataset == 'gta':
        transforms += [sstr.Resize(size=(1052, 1914))]
    transforms += [
      sstr.RandomHorizontalFlip(),
      sstr.ColorJitter(brightness=args.cjvalue, contrast=args.cjvalue, saturation=args.cjvalue, hue=0),
      sstr.RandomResizedCrop((512, 928), scale=(0.5, 2.0)),
      sstr.ToTensor(),
      sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    train_transforms = sstr.Compose(transforms)
    test_transforms = sstr.Compose([
        sstr.ToTensor(),
        sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, test_transforms


def set_metrics(args):
    num_classes = get_dataset_num_classes(args.dataset)
    if args.model == 'deeplabv3_mobilenetv2':
        metrics = {
            'train': StreamSegMetrics(num_classes, 'train'),
            'eval': StreamSegMetrics(num_classes, 'eval'),
            'test_same_dom': StreamSegMetrics(num_classes, 'test_same_dom'),
            'test_diff_dom': StreamSegMetrics(num_classes, 'test_diff_dom')
        }
    else:
        raise NotImplementedError
    return metrics


def gen_clients(args, train_datasets, test_datasets, model, discriminator=None):
    clients = [[], []]
    for i, datasets in enumerate([train_datasets, test_datasets]):
        # ds is a IDDADataset instance related to a specific client
        # the for loop is used to create one Client instance per specific client
        for ds in datasets:
            if args.step in ['5b', '5c']:
                clients[i].append(AdvClient(args, ds, model, discriminator, test_client=i == 1))
            else:
                clients[i].append(Client(args, ds, model, test_client=i == 1))
    # returns respectively training clients and test clients
    return clients[0], clients[1]


def loadCheckpoint(path, model, device=torch.device('cuda')):
    checkpoint = torch.load(os.path.join(os.getcwd(), 'saved_models', f'{path}.pt'), map_location=device)
    model.load_state_dict(checkpoint)


def get_datasets(args):

    train_datasets = []
    eval_datasets = []
    train_transforms, test_transforms = get_transforms(args)

    if args.step == '1':
        root = 'data/idda'
        with open(os.path.join(root, 'train.txt'), 'r') as f:
            train_data = f.read().splitlines()
            train_datasets = IDDADataset(root=root, list_samples=train_data, transform=train_transforms)

    elif args.step in ['2', '4', '5b', '5c']:
        root = 'data/idda'
        with open(os.path.join(root, 'train.json'), 'r') as f:
            # load data in .json format
            all_data = json.load(f)
            for client_id in all_data.keys():
                # each element of train_datasets is a IDDADataset object related to a given client
                # from IDDADataset obj one can recover all images with their label assigned to the client 
                train_datasets.append(IDDADataset(root=root, list_samples=all_data[client_id], transform=train_transforms, client_name=client_id))

    elif args.step in ['3a', '3b', '5a']:
        root = 'data/GTA5'
        with open(os.path.join(root, 'train.txt'), 'r') as f:
            train_data = f.read().splitlines()
            train_datasets = GTADataset(root=root, list_samples=train_data, transform=train_transforms)
        root = 'data/idda'
        with open(os.path.join(root, 'train.json'), 'r') as f:
            # load data in .json format
            all_data = json.load(f)
            for client_id in all_data.keys():
                # each element of train_datasets is a IDDADataset object related to a given client
                # from IDDADataset obj one can recover all images with their label assigned to the client 
                eval_datasets.append(IDDADataset(root=root, list_samples=all_data[client_id], transform=test_transforms, client_name=client_id))

    root = 'data/idda'
    with open(os.path.join(root, 'test_same_dom.txt'), 'r') as f:
        test_same_dom_data = f.read().splitlines()
        test_same_dom_dataset = IDDADataset(root=root, list_samples=test_same_dom_data, transform=test_transforms,
                                            client_name='test_same_dom')
    with open(os.path.join(root, 'test_diff_dom.txt'), 'r') as f:
        test_diff_dom_data = f.read().splitlines()
        test_diff_dom_dataset = IDDADataset(root=root, list_samples=test_diff_dom_data, transform=test_transforms,
                                            client_name='test_diff_dom')
    test_datasets = [test_same_dom_dataset, test_diff_dom_dataset]

    return train_datasets, test_datasets, eval_datasets

