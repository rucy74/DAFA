"""
Robust training losses. Based on code from
https://github.com/yaodongyu/TRADES
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np



def trades_loss(model, x_natural, y, optimizer, args, class_weights, batch_indices=None, memory_dict=None):
    """The TRADES KL-robustness regularization term proposed by
       Zhang et al., with added support for stability training and entropy
       regularization"""
    
    if args is not None:
        step_size, perturb_steps, epsilon = args.pgd_step_size, args.pgd_num_steps, args.epsilon
        beta = args.beta

    loss_dict = {}

    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='sum')
    
    model.eval()  # moving to eval mode to freeze batchnorm stats
    batch_size = len(x_natural)

    ## dafa masking
    class_weights_mask = torch.zeros(len(y)).cuda()
    for i in range(args.n_class):
        cur_indices = np.where(y.detach().cpu().numpy() == i)[0]
        class_weights_mask[cur_indices] = class_weights[i]
    
    # generate adversarial example
    x_adv = x_natural.detach() + 0.  # the + 0. is for copying the tensor
    x_adv += 0.001 * torch.randn(x_natural.shape).cuda().detach()

    logits_nat = model(x_natural)
    
    for i in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            logits = model(x_adv)
            loss_kl = criterion_kl(F.log_softmax(logits, dim=1), F.softmax(logits_nat, dim=1))
        
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        
        x_adv = x_adv.detach() + (class_weights_mask * step_size).view(-1, 1, 1, 1)  * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - (class_weights_mask * epsilon).view(-1, 1, 1, 1)) , 
                                           x_natural + (class_weights_mask * epsilon).view(-1, 1, 1, 1))        
       
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv = Variable(x_adv, requires_grad=False)

    model.train()

    # zero gradient
    optimizer.zero_grad()

    logits = model(x_adv)
    logits_nat = model(x_natural)

    ## Cross Entropy loss for clean data
    
    loss_natural = (torch.nn.CrossEntropyLoss(reduction='none')(logits_nat, y) * class_weights_mask).mean()
    loss_dict['natural'] = loss_natural.item()

    ## robust loss

    p_natural = F.softmax(logits_nat, dim=1)
    loss_robust = criterion_kl(F.log_softmax(logits, dim=1), p_natural) / batch_size

    loss_dict['robust'] = loss_robust.item()
    loss = loss_natural + beta * loss_robust

    if memory_dict is not None:
        memory_dict['probs'][batch_indices] = F.softmax(logits, dim=1).detach().cpu().numpy()
        memory_dict['labels'][batch_indices] = y.detach().cpu().numpy()

    return loss, loss_dict


def madry_loss(model, x_natural, y, optimizer, args, class_weights, batch_indices=None, memory_dict=None):

    if args is not None:
        step_size, perturb_steps, epsilon = args.pgd_step_size, args.pgd_num_steps, args.epsilon
    
    criterion_ce = torch.nn.CrossEntropyLoss()
    model.eval()

    loss_dict = {}

    ## dafa masking
    class_weights_mask = torch.zeros(len(y)).cuda()
    for i in range(args.n_class):
        cur_indices = np.where(y.detach().cpu().numpy() == i)[0]
        class_weights_mask[cur_indices] = class_weights[i]
    
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    
    for i in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            logits = model(x_adv)
            loss_ce = criterion_ce(logits, y)
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        x_adv = x_adv.detach() + (class_weights_mask * step_size).view(-1, 1, 1, 1) * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - (class_weights_mask * epsilon).view(-1, 1, 1, 1)) , 
                                           x_natural + (class_weights_mask * epsilon).view(-1, 1, 1, 1))
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv = Variable(x_adv, requires_grad=False)

    model.train()
    optimizer.zero_grad()

    logits = model(x_adv)

    
    
    loss_robust = (torch.nn.CrossEntropyLoss(reduction='none')(logits, y) * class_weights_mask).mean()
    loss = loss_robust
    
    loss_dict['robust'] = loss_robust.item()

    loss_dict['total'] = loss.item()

    if memory_dict is not None:
        memory_dict['probs'][batch_indices] = F.softmax(logits, dim=1).detach().cpu().numpy()
        memory_dict['labels'][batch_indices] = y.detach().cpu().numpy()

    return loss, loss_dict

