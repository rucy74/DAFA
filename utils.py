from models.wideresnet import WideResNet
from models.resnet import ResNet18
from models.preact_resnet import PreActResNet18

import numpy as np
import torch.nn as nn

def get_model(args):
    
    # STL dataset is resized to (64, 64) -> 512*4

    name = args.model
    num_classes = args.n_class

    name_parts = name.split('-')
    if name_parts[0] == 'wrn':
        depth = int(name_parts[1])
        widen = int(name_parts[2])
        model = WideResNet(depth=depth, num_classes=num_classes, widen_factor=widen, args=args)
        if args.dataset == 'stl10':
            model.fc = nn.Linear(640*4, 10)
    elif name_parts[0] == 'resnet':
        model = ResNet18(num_classes=num_classes)
        if args.dataset == 'stl10':
            model.linear = nn.Linear(512*4, 10)
    elif name_parts[0] == 'pre' and name_parts[1] == 'resnet':
        model = PreActResNet18(num_classes=num_classes)
        if args.dataset == 'stl10':
            model.linear = nn.Linear(512*4, 10)
    else:
        raise ValueError('Could not parse model name %s' % name)

    return model

def calculate_class_weights(memory_dict, lamb=1.0):
    probs, labels = memory_dict['probs'], memory_dict['labels']
    num_classes = len(np.unique(labels))
    class_similarity = np.zeros((num_classes, num_classes))
    class_weights = np.ones(num_classes)

    for i in range(num_classes):
        cur_indices = np.where(labels == i)[0]
        class_similarity[i] = np.mean(probs[cur_indices], axis=0)

    for i in range(num_classes):
        for j in range(num_classes):
            if i == j: continue
            if class_similarity[i, i] < class_similarity[j, j]:                
                class_weights[i] += lamb * class_similarity[i, j] * class_similarity[j, j]                
            else:
                class_weights[i] -= lamb * class_similarity[i, j] * class_similarity[j, j]

    return class_weights
