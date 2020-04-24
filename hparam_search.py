from train import Net, Trainer
from utils.iam import IAMWords, Resize, CHARACTERS, word_to_tensor, tensor_to_word

import optuna
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from pathlib import Path


def experiment(trial: optuna.Trial, train_dl, valid_dl, *,
               device='cpu', n_classes, n_epochs, input_shape):
    net = Net(input_shape, n_classes, trial).to(device)
    trainer = Trainer(trial, n_epochs=n_epochs, device=device)

    def epoch_end_callback(epoch, _, acc):
        trial.report(acc, epoch)
        if trial.should_prune(epoch):
            raise optuna.exceptions.TrialPruned()

        if acc < 0.05:  # Prune really bad runs
            raise optuna.exceptions.TrialPruned()

    losses, accuracies = trainer.fit(net, train_dl, valid_dl,
                                     epoch_end_callback=epoch_end_callback)
    return accuracies[-1]


if __name__ == '__main__':
    DEVICE = 'cuda:0'
    ROOT = next(p
                for p in [Path('C:/datasets'), Path('/home/ubuntu/datasets')]
                if p.is_dir())
    BATCH_SIZE = 64
    N_EPOCHS = 20
    INPUT_SHAPE = (32, 128)

    tfms = Compose([Resize(INPUT_SHAPE[1], INPUT_SHAPE[0]), ToTensor()])
    train_ds = IAMWords(ROOT, split='train', transform=tfms)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_ds = IAMWords(ROOT, split='valid', transform=tfms)
    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_ds = IAMWords(ROOT, split='test', transform=tfms)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    n_classes = len(CHARACTERS) + 1

    study = optuna.create_study(study_name='hparam_search',
                                load_if_exists=True,
                                storage='sqlite:///hparam_search.db',
                                direction='maximize')
    study.optimize(lambda trial: experiment(trial, train_dl, valid_dl, device=DEVICE,
                                            n_classes=n_classes, n_epochs=20,
                                            input_shape=INPUT_SHAPE),
                   n_trials=10)
