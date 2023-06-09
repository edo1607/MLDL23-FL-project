import torch
import utils.getters_setters as gs

from server import Server
from client import Client
from centralized import Centralized
from fdaCentralized import FdaCentralized
from advCentralized import AdvCentralized
from advServer import AdvServer
from utils.checkpointSaver import CheckpointSaver
from models.discriminator import Discriminator
from utils.args import get_parser

import wandb
wandb.login()



def main():
    parser = get_parser()
    args = parser.parse_args()
    gs.set_seed(args.seed)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    # in order to include transformations in hyperparameters to save
    train_transforms, test_transforms = gs.get_transforms(args)

    # if the such args are not set, they are choosen as the best one
    if args.hnm is None:
        args.hnm = False if args.step == '4' else True
    if args.lr is None:
        args.lr = 0.025 if args.step in ['1', '2'] else 0.01

    config={
      "step": args.step,
      "seed": args.seed,
      "dataset": args.dataset,
      "model": args.model,
      "num_rounds": args.num_rounds,
      "num_epochs": args.num_epochs,
      "clients_per_round": args.clients_per_round,
      "hnm": args.hnm,
      "learning_rate": args.lr,
      "schedule": args.schedule,
      "batch_size": args.bs,
      "weight_decay": args.wd,
      "momentum": args.m,
      "train_transform": train_transforms,
      "test_transform": test_transforms,
      "FdaSize": args.L,
      "fdaSize_pixelss": args.b,
      "T": args.T,
      "lambda_adv": args.lam_adv,
      "lambda_kd": args.lam_kd,
      "path_model": args.path_model,
      "path_discriminator": args.path_discriminator,
      "L": args.L
    }

    # to have a config without personalized names
    # vars(args) | {"train_transform": train_transforms, "test_transform": test_transforms}


    # initialize wandb and save it in args for ease of convenience
    run = wandb.init(project="project-2b", config=config)
    args.wandb = run

    print('Initializing model...')
    model = gs.model_init(args)
    model.to(args.device)
    print('Done.')

    if args.path_model is not None:
        print('Loading checkpoint...')
        gs.loadCheckpoint(path=args.path_model, model=model, device=args.device)
        print('Done.')

    if args.step in ['5a', '5b', '5c']:
        print('Initializing discriminator...')
        discriminator = Discriminator(in_channels=gs.get_dataset_num_classes(args.dataset))
        discriminator.to(args.device)
        print('Done.')
        if args.path_discriminator is not None:
            print('Loading checkpoint...')
            gs.loadCheckpoint(path=args.path_discriminator, model=discriminator, device=args.device)
            print('Done.')
    

    print('Generate datasets...')
    train_datasets, test_datasets, eval_datasets = gs.get_datasets(args)
    print('Done.')

    metrics = gs.set_metrics(args)

    print('training and testing...')
    if args.step == '1':
        clf = Centralized(args, model, train_datasets, metrics['train'])
        clf.train()
        loss_tsd = clf.test(test_datasets[0], metrics['test_same_dom'])
        loss_tdd = clf.test(test_datasets[1], metrics['test_diff_dom'])

    elif args.step == '2':
        train_clients, test_clients = gs.gen_clients(args, train_datasets, test_datasets, model)
        clf = Server(args, model, train_clients, metrics['train'])
        clf.train()        
        loss_tsd = clf.test(test_clients[0], metrics['test_same_dom'])
        loss_tdd = clf.test(test_clients[1], metrics['test_diff_dom'])

    elif args.step == '3a':
        clf = Centralized(args, model, train_datasets, metrics['train'])
        clf.train(eval_datasets, metrics['eval'])
        loss_tsd = clf.test(test_datasets[0], metrics['test_same_dom'])
        loss_tdd = clf.test(test_datasets[1], metrics['test_diff_dom'])

    elif args.step == '3b':
        # generate clients to extract, if needed, the styles from eval_clients
        eval_clients, test_clients = gs.gen_clients(args, eval_datasets, test_datasets, model)
        clf = FdaCentralized(args, model, train_datasets, metrics['train'], b=args.b, L=args.L, clients=eval_clients)
        clf.train(eval_datasets, metrics['eval'])
        loss_tsd = clf.test(test_datasets[0], metrics['test_same_dom'])
        loss_tdd = clf.test(test_datasets[1], metrics['test_diff_dom'])

    elif args.step == '4':
        train_clients, test_clients = gs.gen_clients(args, train_datasets, test_datasets, model)
        clf = Server(args, model, train_clients, metrics['train'])
        clf.train()        
        loss_tsd = clf.test(test_clients[0], metrics['test_same_dom'])
        loss_tdd = clf.test(test_clients[1], metrics['test_diff_dom'])
        
    elif args.step == '5a':
        clf = AdvCentralized(args, model, discriminator, train_datasets, metrics['train'], b=args.b)
        clf.train(eval_datasets, metrics['eval'])
        loss_tsd = clf.test(test_datasets[0], metrics['test_same_dom'])
        loss_tdd = clf.test(test_datasets[1], metrics['test_diff_dom'])
    
    elif args.step == '5b':
        train_clients, test_clients = gs.gen_clients(args, train_datasets, test_datasets, model, discriminator)
        clf = AdvServer(args, model, discriminator, train_clients, metrics['train'])
        clf.train()        
        loss_tsd = clf.test(test_clients[0], metrics['test_same_dom'])
        loss_tdd = clf.test(test_clients[1], metrics['test_diff_dom'])

    elif args.step == '5c':
        train_clients, test_clients = gs.gen_clients(args, train_datasets, test_datasets, model, discriminator)
        clf = AdvServer(args, model, discriminator, train_clients, metrics['train'])
        clf.train()
        loss_tsd = clf.test(test_clients[0], metrics['test_same_dom'])
        loss_tdd = clf.test(test_clients[1], metrics['test_diff_dom'])

    elif args.step == 'test':
        # centralized is used just to exploit its function test
        clf = Centralized(args, model, train_datasets, metrics['train'])
        loss_tsd = clf.test(test_datasets[0], metrics['test_same_dom'])
        loss_tdd = clf.test(test_datasets[1], metrics['test_diff_dom'])

    print('Done.')

    if args.step in ['1', '2', '4', '5b']: # the ones not saved during training
        checkpoint_saver = CheckpointSaver(dirpath='./saved_models', args=args, decreasing=False, top_n=1)
        checkpoint_saver(model, metrics['train'].results["Mean IoU"], args.num_epochs)

    # log test metrics
    wandb.log({
              'tsd': metrics['test_same_dom'].results | {'loss': loss_tsd},
              'tdd': metrics['test_diff_dom'].results | {'loss': loss_tdd}
              })
    wandb.finish()



if __name__ == '__main__':
    main()
