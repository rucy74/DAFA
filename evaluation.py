"""
Evaluate robustness against specific attack.
Loosely based on code from https://github.com/yaodongyu/TRADES
"""

import os
import numpy as np
import re
import argparse
import logging
import torch

from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from datasets import CustomDataset
from torch.utils.data import DataLoader
from utils import get_model


def eval(args, model, device, eval_set, loader):
    loss = 0
    total = 0
    correct = 0
    adv_correct = 0
    adv_correct_clean = 0
    adv_total = 0

    model.eval()

    with torch.no_grad():
        for _, (data, target) in enumerate(loader):
            
            data, target = data.to(device), target.to(device)
            data, target = data[target != -1], target[target != -1]
            output = model(data)

            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
                
            is_correct_clean, is_correct_rob, _ = pgd(model, data, target, 
                                                            epsilon=args.test_epsilon, 
                                                            num_steps=args.test_pgd_num_steps, 
                                                            step_size=args.test_pgd_step_size, 
                                                            )

            incorrect_clean = (1 - is_correct_clean).sum()
            incorrect_rob = (1 - np.prod(is_correct_rob, axis=1)).sum()
            adv_correct_clean += (len(data) - int(incorrect_clean))
            adv_correct += (len(data) - int(incorrect_rob))
            adv_total += len(data)
            
            total += len(data)
        
    loss /= total
    accuracy = correct / total
    if adv_total > 0:
        robust_clean_accuracy = adv_correct_clean / adv_total
        robust_accuracy = adv_correct / adv_total
    else:
        robust_accuracy = robust_clean_accuracy = 0.

    eval_data = dict(loss=loss, accuracy=accuracy,
                     robust_accuracy=robust_accuracy,
                     robust_clean_accuracy=robust_clean_accuracy)
    eval_data = {eval_set + '_' + k: v for k, v in eval_data.items()}


    logging.info(
        '{}: Clean loss: {:.4f}, '
        'Clean accuracy: {}/{} ({:.2f}%), '
        '{} clean accuracy: {}/{} ({:.2f}%), '
        'Robust accuracy {}/{} ({:.2f}%)'.format(
            eval_set.upper(), loss,
            correct, total, 100.0 * accuracy,
            'PGD',
            adv_correct_clean, adv_total, 100.0 * robust_clean_accuracy,
            adv_correct, adv_total, 100.0 * robust_accuracy))

    return eval_data

def pgd(model, X, y, epsilon=0.031, num_steps=20, step_size=0.003):
    out = model(X)
    is_correct_natural = (out.max(1)[1] == y).float().cpu().numpy()

    perturbation = torch.rand_like(X, requires_grad=True)
    perturbation.data = perturbation.data * 2 * epsilon - epsilon

    is_correct_adv = []
    opt = optim.SGD([perturbation], lr=1e-3)  # This is just to clear the grad
    for i in range(num_steps):
        opt.zero_grad()

        with torch.enable_grad():
            logits = model(X + perturbation)
            loss = nn.CrossEntropyLoss()(logits, y)

        loss.backward()
        perturbation.data = (perturbation + step_size * perturbation.grad.detach().sign()).clamp(-epsilon, epsilon)
        perturbation.data = torch.min(torch.max(perturbation.detach(), -X), 1 - X)  # clip X+delta to [0,1]

        X_pgd = Variable(torch.clamp(X.data + perturbation.data, 0, 1.0), requires_grad=False)
        is_correct_adv.append(np.reshape((model(X_pgd).max(1)[1] == y).float().cpu().numpy(),[-1, 1]))
    
    is_correct_adv = np.concatenate(is_correct_adv, axis=1)
    return is_correct_natural, is_correct_adv, X_pgd.detach()



def eval_classwise(model, test_loader, args):
    
    clean_acc_classwise = np.zeros(args.n_class)
    pgd_acc_classwise = np.zeros(args.n_class)
    
    cur_nat_correct = 0.0
    cur_pgd_correct = 0.0
    cur_nat_correct_classwise = np.zeros(args.n_class)
    cur_pgd_correct_classwise = np.zeros(args.n_class)
    total = 0.0

    for _, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        
        _, _, x_adv = pgd(model, data, target, 
                        epsilon=args.epsilon, 
                        num_steps=args.pgd_num_steps, 
                        step_size=args.pgd_step_size, 
                        )
        logits = model(x_adv)
        logits_nat = model(data)
        
        total += len(data)
        cur_pgd_correct += (logits.max(1)[1] == target).sum().float().cpu().numpy()
        cur_nat_correct += (logits_nat.max(1)[1] == target).sum().float().cpu().numpy()
        for j in range(args.n_class):
            cur_indices = (target == j)
            if cur_indices.sum() == 0: continue
            cur_nat_correct_classwise[j] += (logits_nat[cur_indices].max(1)[1] == j).sum().float().cpu().numpy()
            cur_pgd_correct_classwise[j] += (logits[cur_indices].max(1)[1] == j).sum().float().cpu().numpy()
        
        
    clean_acc = cur_nat_correct / total
    pgd_acc = cur_pgd_correct / total
    
    clean_acc_classwise = cur_nat_correct_classwise / (total / args.n_class)
    pgd_acc_classwise = cur_pgd_correct_classwise / (total / args.n_class)    
    
    overall_results = np.array({'clean_acc':clean_acc, 'pgd_acc':pgd_acc,
                                'clean_acc_classwise':clean_acc_classwise, 'pgd_acc_classwise':pgd_acc_classwise,
                                })

    np.save(os.path.join("/".join(args.model_dir.split("/")[:-1]), 'eval_epochwise.npy'), overall_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR Attack Evaluation')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=('cifar10', 'cifar100', 'stl10'),
                        help='The dataset')
    parser.add_argument('--model_dir',
                        help='Model for attack evaluation')
    parser.add_argument('--data_dir', default='./data', type=str,
                        help='Directory where datasets are located')
    parser.add_argument('--model', '-m', default='resnet', type=str, choices=('resnet', 'pre-resnet','wrn-28-10'),
                        help='Name of the model')    
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='Input batch size for testing (default: 200)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training')
    parser.add_argument('--epsilon', default=0.031, type=float,
                        help='Attack perturbation magnitude')    
    parser.add_argument('--pgd_num_steps', default=20, type=int,
                        help='Number of PGD steps')
    parser.add_argument('--pgd_step_size', default=0.003, type=float,
                        help='PGD step size')    
    parser.add_argument('--random_seed', default=0, type=int,
                        help='Random seed for permutation of test instances')    

    args = parser.parse_args()
    torch.manual_seed(args.random_seed)
    if args.dataset == 'cifar100':
        args.n_class = 100
    else:
        args.n_class = 10    
   
    logger = logging.getLogger()

    logging.info('Attack evaluation')
    logging.info('Args: %s' % args)

    # settings
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dl_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # set up data loader
    testset = CustomDataset(dataset=args.dataset,
                        root=args.data_dir, train=False,
                        download=True, get_indices=False,
                        )
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, **dl_kwargs)

    # load model
    checkpoint = torch.load(args.model_dir)    
    model = get_model(args)

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
    
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except:
        try:
            model.load_state_dict(checkpoint)
        except:
            model = get_model(args)
            model.load_state_dict(checkpoint)
            if use_cuda:
                model = torch.nn.DataParallel(model).cuda()

    model.eval()

    eval_classwise(model, test_loader, args)


