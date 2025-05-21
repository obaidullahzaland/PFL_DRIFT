import argparse
import os
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
from Datasets import NumpyImageDataset
from torchvision import transforms

from Datasets import DatasetObject
from Models import client_model
from algo.fedavg import train_FedAvg


def main(args):
    data_obj = DatasetObject(
        dataset     = args.dataset_name,
        n_client    = args.n_client,
        seed        = args.data_seed,
        result_path = args.result_path,
        data_dir    = args.data_dir,
        personalize = args.personalize
    )
    print(f"Dataset {args.dataset_name!r} has {data_obj.n_cls} classes.")
    for i, ys in enumerate(data_obj.clnt_y):
        unique = np.unique(ys)
        print(f"  domain {i}: "
            f"{len(unique)} classes → {unique.tolist()}")

    if data_obj.dataset in ['DIGIT','CIFAR10','CIFAR10-C','SVHN']:
        img_size = (32,32)
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        test_tf  = train_tf
    else:
        img_size = (224, 224)
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])
        test_tf = train_tf

    client_loaders = {}
    client_test_loaders = {}
    for i in range(data_obj.n_client):
        ds_tr = NumpyImageDataset(
            data_obj.clnt_x[i],
            data_obj.clnt_y[i],
            transforms=train_tf
        )
        client_loaders[f'client_{i}'] = DataLoader(
            ds_tr,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        ds_te = NumpyImageDataset(
            data_obj.clnt_test_x[i],
            data_obj.clnt_test_y[i],
            transforms=test_tf
        )
        client_test_loaders[f'client_{i}'] = DataLoader(
            ds_te,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    data_obj.client_loaders      = client_loaders
    data_obj.client_test_loaders = client_test_loaders

    model_func = lambda: client_model(args.model_name, pretrained=args.pretrained, num_classes=args.num_classes)


    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training.')
    else:
        cudnn.benchmark = True

    init_model = model_func()
    if not os.path.exists('%sModel/%s/%s_init_model.pt' % (args.result_path, data_obj.name, args.model_name)):
        os.makedirs('%sModel/%s/' % (args.result_path, data_obj.name), exist_ok=True)
        torch.save(init_model.state_dict(), '%sModel/%s/%s_init_model.pt' % (args.result_path, data_obj.name, args.model_name))
    else:
        # Load model
        init_model.load_state_dict(torch.load('%sModel/%s/%s_init_model.pt' % (args.result_path, data_obj.name, args.model_name)))

    print(args.method)
    if args.method == 'fedavg':
        train_FedAvg(args, data_obj, model_func, init_model)
    else:
        raise ValueError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--result_path', default='../resultsPFLICONIP/')
    parser.add_argument('--data_dir', default='../data')
    parser.add_argument('--method', default='fedavg', help='fedavg')
    parser.add_argument('--model_name', default='LeNet', help='LeNet, ResNet18')
    parser.add_argument('--dataset_name', default='DIGIT', help='DIGIT')
    parser.add_argument('--data_seed', default=23, type=int)
    parser.add_argument('--seed', default=23, type=int)
    parser.add_argument('--com_amount', default=200, type=int)
    parser.add_argument('--save_period', default=50, type=int)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--lr_decay_per_round', default=0.998, type=float)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--epoch', default=1, type=int)
    parser.add_argument('--n_client', default=100, type=int)
    parser.add_argument('--sch_step', default=1, type=int)
    parser.add_argument('--sch_gamma', default=1, type=int)
    parser.add_argument('--act_prob', default=0.4, type=float)
    parser.add_argument('--tau', default=5.0, type=float)
    parser.add_argument('--rand_percent', type=int,   default=80,
                    help='%% of local data sampled for ALA')
    parser.add_argument('--layer_idx',   type=int,   default=2,
                    help='number of top layers to adaptively aggregate')
    parser.add_argument('--eta',         type=float, default=1.0,
                    help='learning rate for ALA weight updates')
    parser.add_argument('--personalize', type=bool, default=False)
    parser.add_argument('--aggregation', default='fedavg',
        choices=['fedavg','fedprox','fedavgm','fedadam','scaffold'])
    parser.add_argument('--fedprox_mu',   type=float, default=0.01)
    parser.add_argument('--server_momentum', type=float, default=0.9)
    parser.add_argument('--server_lr',      type=float, default=1.0)
    parser.add_argument('--server_beta1',   type=float, default=0.9)
    parser.add_argument('--server_beta2',   type=float, default=0.999)
    parser.add_argument('--server_eps',     type=float, default=1e-8)
    parser.add_argument('--mixed', type=bool, default=False)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--num_classes', type=int, default=10)

    args = parser.parse_args()

    if args.result_path[-1] != '/':
        args.result_path += '/'

    if args.dataset_name == 'DIGIT':
        args.n_client = 5
    elif args.dataset_name == 'CIFAR10-C':
        args.n_client = 19
    elif args.dataset_name == 'domainnet':
        args.n_client = 6
        print('Reset client:', args.n_client)

    main(args)
