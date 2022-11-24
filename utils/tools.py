import numpy as np
import torch


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        args.learning_rate = args.learning_rate * 0.5
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def reset_counter(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean

    def return_mean_std(self):
        return self.mean, self.std


def loss_process(pred, true, criterion=None, flag=0, mix=False, dataset=None):
    if flag == 2:
        if dataset:
            weight_pred = pred.reshape(-1, pred.shape[-1])
            if dataset.target_feature() == pred.shape[-1]:
                weight_pred = dataset.inverse_transform(weight_pred.detach().cpu().numpy())
                weight_pred = dataset.standard_transformer(weight_pred)
                return weight_pred
            else:
                weight_pred = weight_pred.expand(weight_pred.shape[0], dataset.target_feature())
                weight_pred = dataset.inverse_transform(weight_pred.detach().cpu().numpy())
                weight_pred = dataset.standard_transformer(weight_pred)
                return weight_pred[:, :1]
        else:
            return pred

    if flag == 0:
        if mix:
            loss = criterion(pred, true)
        else:
            SE = (pred - true) ** 2
            MSE = torch.mean(SE, dim=0)
            loss = torch.mean(MSE, dim=0)
        return loss
    else:
        loss2 = criterion(pred, true)
        return loss2.detach().cpu().numpy()
