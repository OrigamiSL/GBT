import argparse
import os
import torch
import time
import numpy as np
from exp.exp_model import Exp_Model

parser = argparse.ArgumentParser(description='[GBT]')

parser.add_argument('--model', type=str, required=True, default='GBT',
                    help='model of experiment')

parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, '
                         'S:univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--seq_len', type=int, default=48, help='input sequence length of encoder')
parser.add_argument('--label_len', type=int, default=48, help='start token of decoder')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model at the second Auto-Regression Stage')
parser.add_argument('--fd_model', type=int, default=32, help='dimension of model at the first Auto-Regression Stage')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of (pyramid) encoder layers')
parser.add_argument('--auto_d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='Full', help='attention used in encoder, options:[prob, Full]')
parser.add_argument('--criterion', type=str, default='Standard', choices=['Standard', 'Maxabs'],
                    help='options:[Standard, Maxabs]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=5, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling'
                    , default=True)
parser.add_argument('--instance', action='store_true',
                    help='whether to datasets have instance variables'
                    , default=False)
parser.add_argument('--max_batch', type=int, default=512)
parser.add_argument('--feature_extractor', type=str, default='Attention',
                    choices=['ResNet', 'CSPNet', 'Attention'], help='feature extractor used in backbone')
parser.add_argument('--kernel', type=int, default=3)
parser.add_argument('--time', action='store_false',
                    help='whether to use time embedding'
                    , default=True)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument('--test_inverse', action='store_true', help='only inverse test data', default=False)
parser.add_argument('--ConvKer', type=int, default=3,
                    help='Conv Kernel in attention mechanism of logsparse Attention')
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--train', action='store_false',
                    help='whether to train'
                    , default=True)
parser.add_argument('--save_result', action='store_false',
                    help='whether to train'
                    , default=True)
parser.add_argument('--reproducible', action='store_true',
                    help='whether to train'
                    , default=False)
parser.add_argument('--use_RevIN', action='store_true',
                    help='whether to use RevIN'
                    , default=False)
parser.add_argument('--format', type=str, default='transformer', help='[transformer, autoformer]')

# SCINet
parser.add_argument('--hidden_size', default=8, type=int, help='hidden channel of module')
parser.add_argument('--num_levels', default=3, type=int, help='num of tree levels')
parser.add_argument('--num_stacks', default=1, type=int, help='num of stacks')
parser.add_argument('--num_groups', default=1, type=int, help='num of groups')
parser.add_argument('--concat_len', default=0, type=int, help='concat results of two stacks or not')
parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
parser.add_argument('--dilation', default=1, type=int, help='dilation')
parser.add_argument('--positionalEcoding', type=bool, default=True)
parser.add_argument('--single_step_output_One', type=int, default=0)

# FEDformer
parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
parser.add_argument('--version', type=str, default='Fourier',
                    help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
parser.add_argument('--mode_select', type=str, default='random',
                    help='for FEDformer, there are two mode selection method, options: [random, low]')
parser.add_argument('--L', type=int, default=3, help='ignore level')
parser.add_argument('--base', type=str, default='legendre', help='mwt base')
parser.add_argument('--moving_avg', default=[24], help='window size of moving average')

# ETSformer
parser.add_argument('--sigma', type=float, default=0.2)
parser.add_argument('--K', type=int, default=1, help='Top-K Fourier bases')

# NBEATS
parser.add_argument('--trend_blocks', type=int, default=5, help='')
parser.add_argument('--trend_layers', type=int, default=5, help='')
parser.add_argument('--trend_layer_size', type=int, default=512, help='')
parser.add_argument('--degree_of_polynomial', type=int, default=5, help='')
parser.add_argument('--seasonality_blocks', type=int, default=5, help='')
parser.add_argument('--seasonality_layers', type=int, default=5, help='')
parser.add_argument('--seasonality_layer_size', type=int, default=512, help='')
parser.add_argument('--num_of_harmonics', type=int, default=5, help='')

# N-HiTS
parser.add_argument('--n_s_hidden', type=int, default=0, help='')
parser.add_argument('--n_x', type=int, default=0, help='')
parser.add_argument('--stack_num', type=int, default=3, help='')
parser.add_argument('--n_blocks', type=str, default='1,1,1', help='')
parser.add_argument('--n_layers', type=str, default='2,2,2,2,2,2,2,2,2', help='')
parser.add_argument('--n_hidden', type=int, default=512, help='')
parser.add_argument('--n_pool_kernel_size', type=str, default='4,4,4', help='')
parser.add_argument('--n_freq_downsample', type=str, default='60,8,1', help='')
parser.add_argument('--dropout_prob_theta', type=int, default=512, help='')
parser.add_argument('--batch_normalization', action='store_true', help='', default=False)
parser.add_argument('--shared_weights', action='store_true', help='', default=False)

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

if args.reproducible:
    torch.manual_seed(4321)  # reproducible
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
else:
    torch.backends.cudnn.deterministic = False  # Can change it to False --> default: False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1': {'data': 'ETTh1.csv', 'MS': [7, 7, 1], 'M': [7, 7, 7], 'S': [1, 1, 1]},
    'ETTh2': {'data': 'ETTh2.csv', 'MS': [7, 7, 1], 'M': [7, 7, 7], 'S': [1, 1, 1]},
    'ETTm1': {'data': 'ETTm1.csv', 'MS': [7, 7, 1], 'M': [7, 7, 7], 'S': [1, 1, 1]},
    'ETTm2': {'data': 'ETTm2.csv', 'MS': [7, 7, 1], 'M': [7, 7, 7], 'S': [1, 1, 1]},
    'WTH': {'data': 'WTH.csv', 'MS': [5, 5, 1], 'M': [12, 12, 12], 'S': [1, 1, 1]},
    'ECL': {'data': 'ECL.csv', 'MS': [321, 321, 1], 'M': [321, 321, 321], 'S': [1, 1, 1]},
    'Traffic': {'data': 'Traffic.csv', 'MS': [862, 862, 1], 'M': [862, 862, 862], 'S': [1, 1, 1]},
    'Exchange': {'data': 'Exchange.csv', 'MS': [8, 8, 1], 'M': [8, 8, 8], 'S': [1, 1, 1]},
    'weather': {'data': 'weather.csv', 'MS': [21, 21, 1], 'M': [21, 21, 21], 'S': [1, 1, 1]},
    'ILI': {'data': 'ILI.csv', 'MS': [7, 7, 1], 'M': [7, 7, 7], 'S': [1, 1, 1]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

if args.instance:
    if args.enc_in * args.batch_size > args.max_batch:
        args.batch_size = 1

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ', '').split(',')]
args.n_blocks = [int(s_l) for s_l in args.n_blocks.replace(' ', '').split(',')]
args.n_layers = [int(s_l) for s_l in args.n_layers.replace(' ', '').split(',')]
args.n_pool_kernel_size = [int(s_l) for s_l in args.n_pool_kernel_size.replace(' ', '').split(',')]
args.n_freq_downsample = [int(s_l) for s_l in args.n_freq_downsample.replace(' ', '').split(',')]
args.target = args.target.replace('/r', '').replace('/t', '').replace('/n', '')
print('Args in experiment:')
print(args)

Exp = Exp_Model

f_mse_list = []
f_mae_list = []
s_mse_list = []
s_mae_list = []
for ii in range(args.itr):
    # setting record of experiments
    lr = args.learning_rate
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_sl{}_dl{}_at{}_fc{}_dt{}_mx{}_ConvKer{}' \
              '_criterion{}_{}_{}'.format(args.model, args.data, args.features,
                                          args.seq_len, args.label_len, args.pred_len,
                                          args.d_model, args.n_heads, args.s_layers, args.d_layers,
                                          args.attn,
                                          args.factor, args.distil, args.mix,
                                          args.ConvKer, args.criterion
                                          , args.des, ii)

    exp = Exp(args)  # set experiments
    if args.train:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        try:
            exp.train(setting)
        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from training early')

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    time_now = time.time()
    mse, mae = exp.test(setting, load=True, stage='first stage')
    f_mse_list.append(mse)
    f_mae_list.append(mae)
    time_first = time.time() - time_now
    print('\t inference time of the first stage: {:.4f}s'.format(time_first))

    time_now = time.time()
    mse, mae = exp.test(setting, load=True, stage='second stage')
    s_mse_list.append(mse)
    s_mae_list.append(mae)
    time_second = time.time() - time_now - time_first
    print('\t inference time of the second stage: {:.4f}s'.format(time_second))

    args.learning_rate = lr
    torch.cuda.empty_cache()

mse = np.asarray(f_mse_list)
mae = np.asarray(f_mae_list)
avg_mse = np.mean(mse)
std_mse = np.std(mse)
avg_mae = np.mean(mae)
std_mae = np.std(mae)
print('first stage|Mean|mse:{}, mae:{}|Std|mse:{}, mae:{}'.format(avg_mse, avg_mae, std_mse, std_mae))
path = './result.log'
with open(path, "a") as f:
    f.write('first stage|{}_{}|pred_len{}: '.
            format(args.data, args.features, args.pred_len) + '\n')
    f.write('first stage|Mean|mse:{}, mae:{}|Std|mse:{}, mae:{}'.
            format(avg_mse, avg_mae, std_mse, std_mae) + '\n')
    f.flush()
    f.close()

mse = np.asarray(s_mse_list)
mae = np.asarray(s_mae_list)
avg_mse = np.mean(mse)
std_mse = np.std(mse)
avg_mae = np.mean(mae)
std_mae = np.std(mae)
print('second stage|Mean|mse:{}, mae:{}|Std|mse:{}, mae:{}'.format(avg_mse, avg_mae, std_mse, std_mae))
path = './result.log'
with open(path, "a") as f:
    f.write('second stage|{}_{}|pred_len{}: '.
            format(args.data, args.features, args.pred_len) + '\n')
    f.write('second stage|Mean|mse:{}, mae:{}|Std|mse:{}, mae:{}'.
            format(avg_mse, avg_mae, std_mse, std_mae) + '\n')
    f.flush()
    f.close()
