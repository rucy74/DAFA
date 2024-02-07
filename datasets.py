"""
Datasets with unlabeled (or pseudo-labeled) data
"""

from torchvision.datasets import CIFAR10, CIFAR100, STL10
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import logging

def get_transform(dataset='cifar10', train=True):
    if dataset == 'cifar10' or dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor()
            ])

    elif dataset == 'stl10':
        transform_train = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])
               
        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor()
            ])
        
    if not train:
        return transform_test
    else:
        return transform_train


class CustomDataset(Dataset):
    def __init__(self,
                 dataset='cifar10',
                 train=True,
                 get_indices=False,
                 **kwargs):
        """A dataset with auxiliary pseudo-labeled data"""

        kwargs['transform'] = get_transform(dataset, train)
        
        if dataset == 'cifar10':
            self.dataset = CIFAR10(train=train, **kwargs)
        elif dataset == 'cifar100':
            self.dataset = CIFAR100(train=train, **kwargs)
        elif dataset == 'stl10':
            self.dataset = STL10(split='train' if train else 'test', **kwargs)
            self.dataset.targets = self.dataset.labels
        else:
            raise ValueError('Dataset %s not supported' % dataset)
        
        self.get_indices = get_indices

        logger = logging.getLogger()

        if train:
            logger.info("Train")
        else:
            logger.info("Test")
        logger.info("Number of training samples: %d", len(self.targets))
        logger.info("Label (and pseudo-label) histogram: %s",
                    tuple(
                        zip(*np.unique(self.targets, return_counts=True))))
        logger.info("Shape of training data: %s", np.shape(self.data))

    @property
    def data(self):
        return self.dataset.data

    @property
    def targets(self):
        return self.dataset.targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        self.dataset.labels = self.targets  # because torchvision is annoying
        if self.get_indices:
            return self.dataset[item], item
        else:
            return self.dataset[item]

    def __repr__(self):
        fmt_str = 'Semisupervised Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '  Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '  Root Location: {}\n'.format(self.dataset.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.dataset.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.dataset.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
