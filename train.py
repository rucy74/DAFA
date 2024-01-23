"""
Main DAFA script. Based loosely on code from
https://github.com/yaircarmon/semisup-adv


"""


import os
import sys
import time
import logging
import numpy as np

import torch
import torch.optim as optim

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from utils import get_model, calculate_class_weights
from evaluation import eval
from losses import trades_loss, madry_loss
from arguments import get_args

from datasets import CustomDataset



# ----------------------------- CONFIGURATION ----------------------------------
args = get_args()
args.nesterov = not args.nesterov

# ------------------------------ OUTPUT SETUP ----------------------------------
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.model_dir, 'training.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

logging.info('Args: %s', args)

if not args.overwrite:
    final_checkpoint_path = os.path.join(
        model_dir, 'checkpoint-epoch{}.pt'.format(args.epochs))
    if os.path.exists(final_checkpoint_path):
        logging.info('Appropriate checkpoint found - quitting!')
        sys.exit(0)

# ------------------------------- CUDA SETUP -----------------------------------
# should provide some improved performance
cudnn.benchmark = True

use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device('cuda' if use_cuda else 'cpu')


# ----------------- DATASET SETUP -----------------------

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

trainset = CustomDataset(dataset=args.dataset,                                 
                        root=args.data_dir, train=True,
                        download=True, get_indices=True,
                        )
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)

testset = CustomDataset(dataset=args.dataset,
                        root=args.data_dir, train=False,
                        download=True
                        )
test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
  

# ----------------------- TRAIN AND EVAL FUNCTIONS -----------------------------
def train(args, model, device, optimizer, train_loader, epoch, class_weights):

    model.train()

    # save probability for calculating class similarity
    if args.rob_fairness_algorithm == 'dafa' and epoch == args.dafa_warmup:
        memory_dict = {'probs':np.zeros((len(trainset), args.n_class)), 
                       'labels':np.zeros(len(trainset))}
    else:
        memory_dict = None


    for batch_idx, dataset in enumerate(train_loader):
        
        (data, target), batch_indices = dataset
        data, target = data.to(device), target.to(device)

        model.train()
        optimizer.zero_grad()

        # calculate robust loss
        if args.loss == 'trades':
            loss, loss_dict = trades_loss(model=model, x_natural=data, y=target, optimizer=optimizer, args=args, 
                                          class_weights=class_weights, batch_indices=batch_indices, memory_dict=memory_dict)
        elif args.loss == 'pgd':
            loss, loss_dict = madry_loss(model=model, x_natural=data, y=target, optimizer=optimizer, args=args, 
                                          class_weights=class_weights, batch_indices=batch_indices, memory_dict=memory_dict)
            
        model.train()

        loss.backward()
        optimizer.step()
       
        # print progress
        if batch_idx % args.log_interval == 0:
            default_log = 'Train Epoch: {} [{}/{} ({:.0f}%)]\t'.format(epoch, (batch_idx + 1) * args.batch_size, len(trainset),
                                                                       100. * (batch_idx + 1) / len(train_loader))
            loss_log = '[Loss] '
            for k, v in loss_dict.items():
                if isinstance(v, int) or isinstance(v, float):
                    loss_log += k + ' : {:.6f}\t'.format(v)
                else:
                    loss_log += k + ' : ' + v + '\t'
            logging.info(default_log + loss_log)

    return memory_dict
        


def adjust_learning_rate(optimizer, epoch, lr=None):
    """decrease the learning rate"""
    if lr is None:
        lr = args.lr
    if optimizer is None:
        return None
    schedule = args.lr_schedule
    
    if schedule == 'trades':
        if epoch >= 0.75 * args.epochs:
            lr = args.lr * 0.1
        if epoch >= 0.9 * args.epochs:
            lr = args.lr * 0.01
        if epoch >= args.epochs:
            lr = args.lr * 0.001
    elif schedule == 'madry':
        if epoch >= 0.5 * args.epochs:
            lr = args.lr * 0.1
        if epoch >= 0.75 * args.epochs:
            lr = args.lr * 0.01
        if epoch >= args.epochs:
            lr = args.lr * 0.001
    elif schedule == 'bag_of_tricks':
        if epoch >= args.epochs - 10:
            lr = args.lr * 0.1
        if epoch >= args.epochs - 5:
            lr = args.lr * 0.01    
    else:
        raise ValueError('Unkown LR schedule %s' % schedule)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
# ------------------------------------------------------------------------------

# ----------------------------- TRAINING LOOP ----------------------------------
def main():    
    if args.dataset == 'cifar100':
        args.n_class = 100
    else:
        args.n_class = 10    
    
    model = get_model(args)
    logging.info(args.model)
    
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    if args.load_epoch > 0:
        print('resuming ... ', args.load_model_dir)
        f_path = os.path.join(args.load_model_dir)
        checkpoint = torch.load(f_path)

        if 'latest' in f_path:
            model.load_state_dict(checkpoint)
        else:
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

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                              nesterov=args.nesterov)
    
    # DAFA class weights, initialized as all-one
    class_weights = torch.ones(args.n_class).cuda()

    init_time = time.time()
    
    for epoch in range(args.load_epoch + 1, args.epochs + 1):
        # adjust learning rate for SGD
        lr = adjust_learning_rate(optimizer, epoch)
        logger.info('Setting learning rate to %g' % lr)
        args.epoch = epoch - 1

        # adversarial training        
        
        memory_dict = train(args, model, device, optimizer, train_loader, epoch, class_weights)
        
        # evaluation on natural examples
        logging.info(120 * '=')

        if epoch % args.eval_freq == 0 or (args.epochs - epoch) <= args.save_start or epoch - args.load_epoch == 1 or epoch == args.epochs:            
            eval(args, model, device, 'test', test_loader)
            
            logging.info(120 * '=')

        # save checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs or ((args.epochs - epoch) < args.save_start) or args.save_all:
            torch.save(dict(state_dict=model.state_dict()), os.path.join(model_dir, 'checkpoint-epoch{}.pt'.format(epoch)))            

        torch.save(model.state_dict(), os.path.join(model_dir, 'checkpoint-latest.pt'))

        if args.rob_fairness_algorithm == 'dafa' and epoch == args.dafa_warmup:
            class_weights = calculate_class_weights(memory_dict, args.dafa_lambda)
            
            class_weights_log = 'Assigned class weights => '
            for i in range(len(class_weights)):
                class_weights_log += '{}: {}, '.format(i, class_weights[i])
            logging.info(class_weights_log)

        elapsed_time = time.time() - init_time
        print('elapsed time : %d h %d m %d s' % (elapsed_time / 3600, (elapsed_time % 3600) / 60, (elapsed_time % 60)))
    
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
