import torch
from torch import Tensor
from torch.nn import Module, Sequential, Conv2d, MaxPool2d, LSTM, ReLU, Conv1d, LogSoftmax, BatchNorm2d
import optuna
from utils.patch import CTCLoss
from utils.layers import NoOpt
from utils.cumulative_average import CumulativeAverage
import numpy as np

from utils.iam import IAMWords, Resize, CHARACTERS, word_to_tensor, tensor_to_word
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
from pathlib import Path
from typing import Union


class Net(Module):
    def __init__(self, input_shape, n_classes: int, trial: Union[optuna.Trial, optuna.trial.FixedTrial]):
        super().__init__()
        self.n_classes = n_classes

        input_shape = (1, 1, *input_shape)

        n_layers = trial.suggest_int('cnn_layers', 1, 5)
        use_batchnorm = trial.suggest_categorical(f'cnn_batchnorm', [True, False])
        prev_filters = 1
        layers = []
        for i in range(n_layers):
            kernel_size = trial.suggest_int(f'cnn_kernel_{i}', 1, 5)
            max_pool_size = trial.suggest_int(f'cnn_maxpool_{i}', 1, 3)
            filters = trial.suggest_categorical(f'cnn_filters_{i}', [32, 64, 128, 256, 512])
            max_pool_stride = max_pool_size if i < 2 else (max_pool_size, 1)
            layers.append(Sequential(
                Conv2d(prev_filters, filters, kernel_size, padding=3),
                BatchNorm2d(filters) if use_batchnorm else NoOpt(),
                ReLU(),
                MaxPool2d(kernel_size=max_pool_size, stride=max_pool_stride),
            ))
            prev_filters = filters
        self.cnn = Sequential(*layers)
        cnn_out_shape = self.cnn(torch.zeros(input_shape)).size()

        lstm_input_size = cnn_out_shape[1] * cnn_out_shape[2]
        lstm_layers = trial.suggest_int('lstm_layers', 1, 7)
        lstm_hidden_size = trial.suggest_int('lstm_hidden_size', 64, 512)
        lstm_dropout = trial.suggest_uniform('lstm_dropout', 0., 0.5)
        self.rnn = LSTM(lstm_input_size, lstm_hidden_size, num_layers=lstm_layers,
                        batch_first=True, bidirectional=True, dropout=lstm_dropout)
        lstm_output_size = lstm_hidden_size * 2

        self.fc = Conv1d(lstm_output_size, n_classes, kernel_size=1)
        self.log_softmax = LogSoftmax(dim=-1)

    def forward(self, x: Tensor):
        x = self.cnn(x)

        x = x.transpose(1, 3)
        x = x.reshape((x.size(0), x.size(1), -1))

        x, _ = self.rnn(x)

        x = x.transpose(1, 2)
        x = self.fc(x)
        x = x.transpose(1, 2)

        return self.log_softmax(x)


class Trainer:
    def __init__(self, trial: Union[optuna.Trial, optuna.trial.FixedTrial], n_epochs: int, device: str = 'cuda'):
        self.n_epochs = n_epochs
        self.criterion = CTCLoss()
        self.lr = trial.suggest_loguniform('lr', 0.0001, 0.01)
        self.device = device

    def train_step(self, model, batch):
        x, y_true = batch

        x = x.to(self.device)

        y_true = [word_to_tensor(w) for w in y_true]
        y_true_packed = torch.cat(y_true).type(torch.int32).to(self.device)
        y_true_lengths = Tensor([w.size(0) for w in y_true]).type(torch.int32).to(self.device)

        y_pred = model(x).transpose(0, 1).to(self.device)
        y_pred_lengths = Tensor([y_pred.shape[0] for _ in range(y_pred.shape[1])]).type(torch.int32).to(self.device)

        return self.criterion(y_pred, y_true_packed, y_pred_lengths, y_true_lengths)

    def valid_step(self, model, batch):
        x, y_true = batch

        x = x.to(self.device)

        y_pred = model(x).argmax(-1)
        y_pred = [tensor_to_word(t)
                  for t in y_pred]

        accuracy = np.mean([y_p == y_t for y_p, y_t in zip(y_pred, y_true)])
        return accuracy

    def fit(self, model, train_dl, valid_dl, *, epoch_end_callback=None):
        optimizer = Adam(model.parameters(), lr=self.lr)

        losses, accuracies = [], []
        for epoch in range(self.n_epochs):
            print(f'\n===========\nEpoch {epoch}\n===========')

            model.train()
            avg_loss = CumulativeAverage()
            for batch in tqdm(train_dl):
                optimizer.zero_grad()
                loss = self.train_step(model, batch)
                loss.backward()
                optimizer.step()
                avg_loss.append(loss.item(), weight=batch[0].size(0))
            avg_loss = avg_loss.average()
            losses.append(avg_loss)
            print(f'Average loss = {avg_loss:.2f}')

            accuracy = CumulativeAverage()
            model.eval()
            for batch in tqdm(valid_dl):
                accuracy.append(self.valid_step(model, batch), weight=batch[0].size(0))
            accuracy = accuracy.average()
            accuracies.append(accuracy)
            print(f'Validation accuracy = {accuracy:.2f}')

            if epoch_end_callback is not None:
                epoch_end_callback(epoch, avg_loss, accuracy)

        return losses, accuracies

    def test(self, model, test_dl):
        accuracy = CumulativeAverage()
        model.eval()
        for batch in tqdm(test_dl):
            accuracy.append(self.valid_step(model, batch), weight=batch[0].size(0))
        print(f'Test accuracy = {accuracy.average():.2f}')


if __name__ == '__main__':
    DEVICE = 'cuda:0'
    ROOT = next(p
                for p in [Path('C:/datasets'), Path('/home/ubuntu/datasets')]
                if p.is_dir())
    BATCH_SIZE = 64
    N_EPOCHS = 100
    INPUT_SHAPE = (32, 128)

    tfms = Compose([Resize(INPUT_SHAPE[1], INPUT_SHAPE[0]), ToTensor()])
    train_ds = IAMWords(ROOT, split='train', transform=tfms)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_ds = IAMWords(ROOT, split='valid', transform=tfms)
    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_ds = IAMWords(ROOT, split='test', transform=tfms)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    n_classes = len(CHARACTERS) + 1

    study = optuna.load_study('hparam_search', storage='sqlite:///hparam_search.db')
    trial = optuna.trial.FixedTrial(study.best_params)
    net = Net(INPUT_SHAPE, n_classes=n_classes, trial=trial).to(DEVICE)
    trainer = Trainer(trial, n_epochs=N_EPOCHS, device=DEVICE)
    trainer.fit(net, train_dl, valid_dl)
    trainer.test(net, test_dl)
    torch.save(net, 'best_model.pkl')
