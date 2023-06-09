import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--dataset', type=str, default='idda', choices=['idda', 'femnist', 'gta'], help='dataset name')
    parser.add_argument('--model', type=str, default='deeplabv3_mobilenetv2', choices=['deeplabv3_mobilenetv2', 'resnet18', 'cnn'], help='model name')
    parser.add_argument('--num_rounds', type=int, help='number of rounds')
    parser.add_argument('--num_epochs', type=int, help='number of local epochs')
    parser.add_argument('--clients_per_round', type=int, help='number of clients trained per round')
    parser.add_argument('--hnm', type=bool, help='Use hard negative mining reduction or not')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--bs', type=int, default=8, help='batch size')
    parser.add_argument('--wd', type=float, default=1e-8, help='weight decay')
    parser.add_argument('--m', type=float, default=0.9, help='momentum')
    parser.add_argument('--schedule', type=str, default='exp', choices=['none', 'cosine', 'linear', 'exp', 'step'], help='schedule the learning rate decay')
    parser.add_argument('--L', type=float, help='fda window size')
    parser.add_argument('--b', type=int, help='fda window size')
    parser.add_argument('--step', type=str, choices=['1', '2', '3a', '3b', '4', '5a', '5b', '5c', 'test'], help='Step of the project')
    parser.add_argument('--path_model', type=str, help='path of the model to be loaded')
    parser.add_argument('--path_discriminator', type=str, help='path of the discriminator to be loaded')
    parser.add_argument('--T', type=int, help='server model update every T rounds. If it is 0, it is never updated')
    parser.add_argument('--cjvalue', type=float, default=0.3, help='colorJitter value of brightness, contrast and saturation')
    parser.add_argument('--conf_th', type=float, default=0.9, help='confidence threshold for teacher model')
    parser.add_argument('--fraction', type=float, default=0.66, help='fraction of pixel to take for each class')
    parser.add_argument('--lam_adv', type=float, default=1e-6, help='adversarial loss lambda coefficient')
    parser.add_argument('--lam_kd', type=float, default=0, help='knowledge distillation loss lambda coefficient')
    parser.add_argument('--alpha_kd', type=float, default=0.5, help='knowledge distillation alpha coefficient')

    return parser
