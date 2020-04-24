from pathlib import Path

import optuna
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from train import Trainer, Net
from utils.iam import Resize, IAMWords, CHARACTERS


if __name__ == '__main__':
    DEVICE = 'cuda:0'
    ROOT = next(p
                for p in [Path('C:/datasets'), Path('/home/ubuntu/datasets')]
                if p.is_dir())
    BATCH_SIZE = 8  # 64
    INPUT_SHAPE = (32, 128)

    tfms = Compose([Resize(INPUT_SHAPE[1], INPUT_SHAPE[0]), ToTensor()])
    test_ds = IAMWords(ROOT, split='test', transform=tfms)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    n_classes = len(CHARACTERS) + 1

    study = optuna.load_study('hparam_search', storage='sqlite:///hparam_search.db')
    trial = optuna.trial.FixedTrial(study.best_params)
    net = torch.load('best_model.pkl').to(DEVICE)
    trainer = Trainer(trial, n_epochs=1, device=DEVICE)
    trainer.test(net, test_dl)
