from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_WTH
from exp.exp_basic import Exp_Basic
from GBT.GBT import GBT
from SCINet.SCINet import SCINet
from FEDformer.FEDformer import FEDformer
from ETSformer.model import ETSformer
from FEDformer.Autoformer import Autoformer
from NBEATS.nbeats import NBeats
from NHiTS.nhits import NHiTS

from utils.tools import EarlyStopping, adjust_learning_rate, loss_process
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings

warnings.filterwarnings('ignore')


class Exp_Model(Exp_Basic):
    def __init__(self, args):
        super(Exp_Model, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'GBT': GBT,
            'SCINet': SCINet,
            'FEDformer': FEDformer,
            'ETSformer': ETSformer,
            'Autoformer': Autoformer,
            'NBEATS': NBeats,
            'NHiTS': NHiTS
        }
        if self.args.model == 'GBT':
            e_layers = self.args.s_layers
            d_layers = self.args.d_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                e_layers,  # self.args.e_layers,
                d_layers,  # self.args.d_layers,
                self.args.auto_d_layers,
                self.args.dropout,
                self.args.attn,
                self.args.time,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.args.feature_extractor,
                self.args.kernel,
                self.args.fd_model,
                self.args.moving_avg,
                self.args.instance,
                self.args.use_RevIN,
                self.args.format,
                self.device,
            ).float()
        elif self.args.model == 'SCINet':
            model = model_dict[self.args.model](
                self.args.pred_len,
                self.args.seq_len,
                self.args.enc_in,
                self.args.hidden_size,
                self.args.num_stacks,
                self.args.num_levels,
                self.args.concat_len,
                self.args.num_groups,
                self.args.kernel,
                self.args.dropout,
                self.args.single_step_output_One,
                self.args.positionalEcoding,
                self.args.INN,
                self.args.d_model,
                self.args.dec_in,
                self.args.time,
                self.args.attn,
                self.args.activation,
                self.args.d_layers,
                self.args.c_out,
                self.args.n_heads,
                self.args.factor,
                self.args.instance
            ).float()
        elif self.args.model == 'FEDformer' or self.args.model == 'Autoformer':
            e_layers = self.args.s_layers[0]
            d_layers = self.args.d_layers
            model = model_dict[self.args.model](
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.version,
                self.args.mode_select,
                self.args.modes,
                self.args.output_attention,
                self.args.moving_avg,
                self.args.enc_in,
                self.args.dec_in,
                self.args.d_model,
                self.args.dropout,
                self.args.time,
                self.args.factor,
                self.args.L,
                self.args.base,
                self.args.c_out,
                self.args.n_heads,
                self.args.activation,
                e_layers,  # self.args.e_layers,
                d_layers,  # self.args.d_layers,
                self.args.instance,
                self.device
            ).float()
        elif self.args.model == 'ETSformer':
            e_layers = self.args.s_layers[0]
            d_layers = self.args.d_layers
            model = model_dict[self.args.model](
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.enc_in,
                self.args.dec_in,
                self.args.d_model,
                self.args.dropout,
                self.args.time,
                self.args.K,
                self.args.sigma,
                self.args.c_out,
                self.args.n_heads,
                self.args.activation,
                e_layers,  # self.args.e_layers,
                d_layers,  # self.args.d_layers,
                self.args.output_attention,
                self.args.instance,
                self.device
            ).float()
        elif self.args.model == 'NBEATS':
            model = model_dict[self.args.model](
                self.args.seq_len,
                self.args.pred_len,
                self.args.c_out,
                self.args.d_model,
                self.args.dropout,
                self.args.time,
                self.args.attn,
                self.args.factor,
                self.args.mix,
                self.args.activation,
                self.args.d_layers,
                self.args.output_attention,
                self.args.n_heads,
                self.args.moving_avg,
                self.args.trend_blocks,
                self.args.trend_layers,
                self.args.trend_layer_size,
                self.args.degree_of_polynomial,
                self.args.seasonality_blocks,
                self.args.seasonality_layers,
                self.args.seasonality_layer_size,
                self.args.num_of_harmonics,
                self.args.instance
            ).float()
        elif self.args.model == 'NHiTS':
            model = model_dict[self.args.model](
                self.args.seq_len,
                self.args.pred_len,
                self.args.d_model,
                self.args.dropout,
                self.args.time,
                self.args.attn,
                self.args.factor,
                self.args.mix,
                self.args.activation,
                self.args.d_layers,
                self.args.output_attention,
                self.args.n_heads,
                self.args.stack_num,
                self.args.n_pool_kernel_size,
                self.args.n_layers,
                self.args.n_hidden,
                self.args.n_freq_downsample
            ).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        total_num = sum(p.numel() for p in model.parameters())
        print("Total parameters: {0}MB".format(total_num/1024/1024))
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_WTH,
            'ECL': Dataset_Custom,
            'Traffic': Dataset_Custom,
            'custom': Dataset_Custom,
            'Exchange': Dataset_Custom,
            'ILI': Dataset_Custom,
            'weather': Dataset_Custom
        }
        Data = data_dict[self.args.data]
        timeenc = 0

        if flag == 'test':
            shuffle_flag = False;
            drop_last = True;
            batch_size = args.batch_size;
        else:
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size;
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            cols=args.cols,
            criterion=args.criterion
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data=None, vali_loader=None, criterion=None, flag='first stage'):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                if self.args.instance:
                    B, _, D = batch_x.shape
                    batch_x = batch_x.permute(0, 2, 1).reshape(B * D, -1).unsqueeze(-1)
                    batch_y = batch_y.permute(0, 2, 1).reshape(B * D, -1).unsqueeze(-1)
                    batch_x_mark = batch_x_mark.repeat(D, 1, 1)
                    batch_y_mark = batch_y_mark.repeat(D, 1, 1)
                pred, true = self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, flag=flag)
                if self.args.instance:
                    pred = pred.reshape(B, D, -1).permute(0, 2, 1)
                    true = true.reshape(B, D, -1).permute(0, 2, 1)
                loss = loss_process(pred, true, criterion, flag=1)
                total_loss.append(loss)

            total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        lr = self.args.learning_rate

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        print('-' * 99)
        print('starting first stage training')

        for epoch in range(self.args.train_epochs):
            iter_count = 0

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                if self.args.instance:
                    B, _, D = batch_x.shape
                    batch_x = batch_x.permute(0, 2, 1).reshape(B * D, -1).unsqueeze(-1)
                    batch_y = batch_y.permute(0, 2, 1).reshape(B * D, -1).unsqueeze(-1)
                    batch_x_mark = batch_x_mark.repeat(D, 1, 1)
                    batch_y_mark = batch_y_mark.repeat(D, 1, 1)
                model_optim.zero_grad()
                iter_count += 1
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark, flag='first stage')
                if self.args.instance:
                    pred = pred.reshape(B, D, -1).permute(0, 2, 1)
                    true = true.reshape(B, D, -1).permute(0, 2, 1)

                loss = loss_process(pred, true, criterion, flag=0, mix=self.args.mix)

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward(torch.ones_like(loss))
                    model_optim.step()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, torch.mean(loss).item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            # true_train_loss = self.vali(train_data, train_loader, criterion)
            vali_loss = self.vali(vali_data, vali_loader, criterion, flag='first stage')
            test_loss = self.vali(test_data, test_loader, criterion, flag='first stage')

            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, true_train_loss, vali_loss, test_loss))
            print("Epoch: {0}, Steps: {1} |  Vali Loss: {2:.7f} Test Loss: {3:.7f}".format(
                epoch + 1, train_steps, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        print('-' * 99)
        print('starting second stage training')

        self.args.learning_rate = lr
        early_stopping.reset_counter()
        for epoch in range(self.args.train_epochs):
            iter_count = 0

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                if self.args.instance:
                    B, _, D = batch_x.shape
                    batch_x = batch_x.permute(0, 2, 1).reshape(B * D, -1).unsqueeze(-1)
                    batch_y = batch_y.permute(0, 2, 1).reshape(B * D, -1).unsqueeze(-1)
                    batch_x_mark = batch_x_mark.repeat(D, 1, 1)
                    batch_y_mark = batch_y_mark.repeat(D, 1, 1)
                model_optim.zero_grad()
                iter_count += 1
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark, flag='second stage')

                if self.args.instance:
                    pred = pred.reshape(B, D, -1).permute(0, 2, 1)
                    true = true.reshape(B, D, -1).permute(0, 2, 1)
                loss = loss_process(pred, true, criterion, flag=0, mix=self.args.mix)

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward(torch.ones_like(loss))
                    model_optim.step()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, torch.mean(loss).item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            # true_train_loss = self.vali(train_data, train_loader, criterion)
            vali_loss = self.vali(vali_data, vali_loader, criterion, flag='second stage')
            test_loss = self.vali(test_data, test_loader, criterion, flag='second stage')

            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, true_train_loss, vali_loss, test_loss))
            print("Epoch: {0}, Steps: {1} |  Vali Loss: {2:.7f} Test Loss: {3:.7f}".format(
                epoch + 1, train_steps, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, load=False, stage='second stage'):
        test_data, test_loader = self._get_data(flag='test')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

        preds = []
        trues = []
        mses = []
        maes = []

        criterion = self._select_criterion()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if self.args.instance:
                    B, _, D = batch_x.shape
                    batch_x = batch_x.permute(0, 2, 1).reshape(B * D, -1).unsqueeze(-1)
                    batch_y = batch_y.permute(0, 2, 1).reshape(B * D, -1).unsqueeze(-1)
                    batch_x_mark = batch_x_mark.repeat(D, 1, 1)
                    batch_y_mark = batch_y_mark.repeat(D, 1, 1)
                pred, true = self._process_one_batch(
                    test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, flag=stage)
                if self.args.instance:
                    pred = pred.reshape(B, D, -1).permute(0, 2, 1)
                    true = true.reshape(B, D, -1).permute(0, 2, 1)
                if self.args.test_inverse:
                    pred = loss_process(pred, true, criterion, flag=2, dataset=test_data)
                    pred = pred.reshape(self.args.batch_size, self.args.pred_len, self.args.c_out)
                    true = true.reshape(-1, pred.shape[-1])
                    if test_data.feature_num == true.shape[-1]:
                        true = test_data.inverse_transform(true.detach().cpu().numpy())
                        true = test_data.standard_transformer(true)
                    else:
                        true = true.expand(true.shape[0], test_data.feature_num)
                        true = test_data.inverse_transform(true.detach().cpu().numpy())
                        true = test_data.standard_transformer(true)
                        true = true[:, :1]
                    true = true.reshape(self.args.batch_size, self.args.pred_len, self.args.c_out)
                else:
                    pred = loss_process(pred, true, criterion, flag=2)
                    pred = pred.detach().cpu().numpy()
                    true = true.detach().cpu().numpy()

                if self.args.save_result:
                    preds.append(pred)
                    trues.append(true)
                else:
                    mse = np.mean((pred - true) ** 2)
                    mae = np.mean(abs(pred - true))
                    mses.append(mse)
                    maes.append(mae)

        if self.args.save_result:
            preds = np.array(preds)
            trues = np.array(trues)
            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            # msse = (preds - trues) ** 2
            # msse = np.mean(msse, axis=0)
            # np.savetxt("./gbt_2880.log", msse, fmt='%f', delimiter='\n')

            print('test shape:', preds.shape, trues.shape)
            mae, mse = metric(preds, trues)
        else:
            mses = np.array(mses)
            maes = np.array(maes)
            mse = np.mean(mses)
            mae = np.mean(maes)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('mse:{}, mae:{}'.format(mse, mae))
        path = './result.log'
        with open(path, "a") as f:
            f.write(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
            f.write('|dataset: {}_{}|pred_{}|stage_{}|mse:{}, mae:{}'.
                    format(self.args.data, self.args.features, self.args.pred_len, stage, mse, mae) + '\n')
            f.flush()
            f.close()
        if self.args.save_result:
            np.save(folder_path + 'metrics.npy', np.array([mae, mse]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
        return mse, mae

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, flag='first stage'):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float().to(self.device)
        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        if self.args.features == 'MS':
            dec_inp[:, :, 1:] = batch_y[:, -self.args.pred_len:, 1:]
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, flag=flag)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, flag=flag)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        if self.args.features == 'S' or self.args.features == 'M':
            batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
        else:
            batch_y = batch_y[:, -self.args.pred_len:, 0].unsqueeze(-1).to(self.device)

        return outputs, batch_y
