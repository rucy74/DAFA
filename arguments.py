import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description='PyTorch TRADES Adversarial Training')

    # Dataset config
    parser.add_argument('--dataset', type=str, default='cifar10',  choices=('cifar10', 'cifar100', 'stl10'),                       
                        help='The dataset to use for training)')
    parser.add_argument('--data_dir', default='./data', type=str,
                        help='Directory where datasets are located')

    # Model config
    parser.add_argument('--model', '-m', default='wrn-28-10', type=str, choices=('resnet', 'pre-resnet','wrn-28-10'),
                        help='Name of the model (see utils.get_model)')
    parser.add_argument('--model_dir', default='./models',
                        help='Directory of model for saving checkpoint')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Cancels the run if an appropriate checkpoint is found')
    parser.add_argument('--load_model_dir', default='', help='directory of model for saving checkpoint')
    parser.add_argument('--load_epoch', type=int, default=0, metavar='N', help='load epoch')    

    # Logging and checkpointing
    parser.add_argument('--log_interval', type=int, default=100, help='Number of batches between logging of training status')
    parser.add_argument('--save_freq', default=10, type=int, help='Checkpoint save frequency (in epochs)')

    # Generic training configs
    parser.add_argument('--seed', type=int, default=1, help='Random seed. ')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='Input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=200, metavar='N', help='Input batch size for testing (default: 200)')
    parser.add_argument('--epochs', type=int, default=110, metavar='N', help='Number of epochs to train. ')

    # Eval config
    parser.add_argument('--eval_freq', default=10, type=int, help='Eval frequency (in epochs)')
    parser.add_argument('--save_start', default=10, type=int, help='Eval frequency (in epochs)')
    parser.add_argument('--save_all', action='store_true', default=False, help='save all epoch checkpoints option')

    # Optimizer config
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='Learning rate')
    parser.add_argument('--lr_schedule', type=str, default='bag_of_tricks', choices=(
                        'trades', 'bag_of_tricks', 'madry'),
                        help='Learning rate schedule')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
    parser.add_argument('--nesterov', action='store_true', default=False, help='Use extragrdient steps')

    # Adversarial / stability training config
    parser.add_argument('--loss', default='trades', type=str,
                        choices=('trades', 'pgd'),
                        help='Which loss to use: TRADES or PGD')
    parser.add_argument('--epsilon', default=0.031, type=float, help='Adversarial perturbation size')
    parser.add_argument('--test_epsilon', default=0.031, type=float, help='Adversarial perturbation size')

    parser.add_argument('--pgd_num_steps', default=10, type=int, help='number of pgd steps in adversarial training')
    parser.add_argument('--pgd_step_size', default=0.007, help='pgd steps size in adversarial training', type=float)
    parser.add_argument('--test_pgd_num_steps', default=20, type=int, help='number of pgd steps in adversarial training')
    parser.add_argument('--test_pgd_step_size', default=0.003, help='pgd steps size in adversarial training', type=float)
    parser.add_argument('--beta', default=6.0, type=float, help='1/lambda in TRADES')     

    # DAFA argument
    parser.add_argument('--rob_fairness_algorithm', default='dafa', type=str, choices=('dafa', 'none'), 
                        help='use or not to use dafa')
    parser.add_argument('--dafa_warmup', default=70, type=int, help='number of epochs before applying dafa in adversarial training')
    parser.add_argument('--dafa_lambda', default=1.0, help='scale of dafa (cifar10: 1.0, cifar100: 1.5, stl10: 1.5)', type=float)


    args = parser.parse_args()


    return args

