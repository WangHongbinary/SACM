'''
Author: WangHongbinary
E-mail: hbwang22@gmail.com
Date: 2025-12-09 10:14:14
LastEditTime: 2026-01-07 15:55:48
Description: Baseline model training code (multi_modality)
'''

import argparse
import datetime
import logging
import numpy as np
import os
import random
import torch
import torch.nn.functional as F

from torch.utils.data import  DataLoader

from net import *
from dataset.dataset_SEEG import MakeSet, seeg_dataset
from dataset.wav_SEEG import load_features

from baseline_models.EEGNet import EEGNet
from baseline_models.DeepConvNet import DeepConvNet
from baseline_models.ShallowConvNet import ShallowConvNet
from baseline_models.CNN_BiGRU import CNN_BiGRU
from baseline_models.Conformer import Conformer
from baseline_models.DBConformer import DBConformer

from util import log_print, top_k, segment_level_eval, cal_acc


def random_init(seed):
    """Perform random init

    Parameters
    -----------
    seed: 
        random seed
    """
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def backbone(args):
    """Select backbone model

    Parameters
    -----------
    modelname: 
        Backbone model name
    """

    if args.backbone == 'EEGNet':
        backbone_model = EEGNet(n_classes=args.class_num,
                                Chans=args.chn,
                                Samples=args.time_sample_num,
                                kernLength=args.EEGNet_configs[0],
                                F1=args.EEGNet_configs[1],
                                D=args.EEGNet_configs[2],
                                F2=args.EEGNet_configs[3],
                                proj_dim=args.out_dim,
                                proj_time=args.out_time,
                                dropoutRate=0.25,
                                norm_rate=0.5).to(device)
    elif args.backbone == 'DeepConvNet':
        backbone_model = DeepConvNet(args,
                                     n_classes=args.class_num,
                                     input_ch=args.chn,
                                     input_time=args.time_sample_num,
                                     batch_norm=True,
                                     batch_norm_alpha=0.1)
    elif args.backbone == 'ShallowConvNet':
        backbone_model = ShallowConvNet(args,
                                        n_classes=args.class_num,
                                        input_ch=args.chn,
                                        input_time=args.time_sample_num,
                                        batch_norm=True,
                                        batch_norm_alpha=0.1)
    elif args.backbone == 'CNN-BiGRU':
        backbone_model = CNN_BiGRU(n_classes=args.class_num,
                                   input_ch=args.chn,
                                   hidden_size=args.CNN_BiGRU_hidden_size,
                                   proj_dim=args.out_dim,
                                   proj_time=args.out_time)
    elif args.backbone == 'Conformer':
        backbone_model = Conformer(emb_size=args.Conformer_configs[0], 
                                   depth=args.Conformer_configs[1], 
                                   chn=args.chn, 
                                   n_classes=args.class_num,
                                   proj_dim=args.out_dim,
                                   proj_time=args.out_time)
    elif args.backbone == 'DBConformer':
        backbone_model = DBConformer(emb_size=args.DBConformer_configs[0], 
                                     patch_size=args.DBConformer_configs[1],
                                     tem_depth=args.DBConformer_configs[2], 
                                     chn_depth=args.DBConformer_configs[3], 
                                     chn=args.chn, 
                                     time_sample_num=args.time_sample_num,
                                     n_classes=args.class_num,
                                     proj_dim=args.out_dim,
                                     proj_time=args.out_time)
    else:
        raise ValueError('Invalid model name')

    return backbone_model


def train_baseline(args):
    """Train, valid, test

    Parameters
    -----------
    args: 
        Parameters for baseline model training, validation and testing
    """

    log_print(args)

    model_path = os.path.join(args.model_path, args.subject_id, args.backbone, str(args.seed), '')
    matrix_label_path = os.path.join(args.matrix_label_path, args.subject_id, args.backbone, str(args.seed), '')
    forward_feature_path = os.path.join(args.forward_feature_path, args.subject_id, args.backbone, str(args.seed), '')

    # load audio features
    wav_features = load_features(args.device_system,
                                 args.feature_type, 
                                 args.subject_id, 
                                 args.vad_half_window,
                                 args.gpu_id)

    all_data, all_label, in_channels = seeg_dataset(args.device_system,
                                                    args.subject_id, 
                                                    args.session_num, 
                                                    args.channel_path, 
                                                    args.label_type)
    print('all_data.shape:', all_data.shape)
    print('all_wav_features.shape:', wav_features.shape)
    print('all_label.shape:', all_label.shape)
    print('in_channels:', in_channels)

    args.chn = in_channels
    args.time_sample_num = all_data.shape[2]
    
    # data from each session of each subject were divided into training, validation, and testing sets in an 8:1:1 ratio, 
    # with the first 8 blocks used for training, and the remaining 2 blocks for validation and testing, respectively
    train_index = []
    valid_index = []
    test_index = []

    for i in range(args.session_num):
        train_start = i * 480
        train_end = train_start + 384
        train_index.append(np.arange(train_start, train_end))
    
        valid_start = train_end
        valid_end = valid_start + 48
        valid_index.append(np.arange(valid_start, valid_end))
    
        test_start = valid_end
        test_end = test_start + 48
        test_index.append(np.arange(test_start, test_end))

    train_index = np.hstack(train_index)
    valid_index = np.hstack(valid_index)
    test_index = np.hstack(test_index)
    
    train_seeg = all_data[train_index]
    valid_seeg = all_data[valid_index]
    test_seeg = all_data[test_index]

    train_wav_features = wav_features[train_index]
    valid_wav_features = wav_features[valid_index]
    test_wav_features = wav_features[test_index]

    train_label = all_label[train_index]
    valid_label = all_label[valid_index]
    test_label = all_label[test_index]

    if args.random:
        perm = torch.randperm(train_seeg.size(0))
        train_wav_features = train_wav_features[perm]

    # make set
    train_set = MakeSet(train_seeg, train_wav_features, train_label)
    valid_set = MakeSet(valid_seeg, valid_wav_features, valid_label)
    test_set  = MakeSet(test_seeg, test_wav_features, test_label)
    
    # make DataLoader
    train_loader = DataLoader(train_set, args.batch_tr, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, args.batch_tr, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, args.batch_te, shuffle=False, drop_last=True)

    model = backbone(args).to(device)
    model.apply(weights_init)
    audio_proj = Audio_Proj(in_dim=768, out_dim=args.out_dim, in_time=79, out_time=args.out_time).to(device)
    audio_proj.apply(weights_init)

    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.l_r, "weight_decay": 1e-4},
                                  {"params": audio_proj.parameters(), "lr": args.l_r_proj, "weight_decay": 0.0}])

    model_loss = np.inf
    last_best_epoch = 0

    for epoch in range(args.max_epoch):
        '''################################### train ###################################'''
        model.train()

        p_5 = 0
        p_1 = 0
        loss = 0
        train_id = 0

        for batch_id, data in enumerate(train_loader):
            inputs, wav_features, label = data
            inputs = inputs.to(device)
            wav_features = wav_features.to(device)
            inputs = inputs.reshape(args.batch_tr, 1, inputs.shape[1], -1)

            outputs = model.get_features(inputs)
            wav_features_proj = audio_proj(wav_features)
            
            p_5, p_1, loss, batch_loss, pred_train_label = segment_level_eval(outputs, 
                                                                              wav_features_proj, 
                                                                              label, 
                                                                              args.batch_tr, 
                                                                              args.loss_f, 
                                                                              p_5, 
                                                                              p_1, 
                                                                              loss, 
                                                                              args.temperature, 
                                                                              device)
            train_id = train_id + 1
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        train_acc5, train_acc1, train_loss = cal_acc(train_id, p_5, p_1, loss, args.batch_tr)

        '''################################### valid ###################################'''
        model.eval()

        p_5 = 0
        p_1 = 0
        loss = 0
        valid_id = 0

        for batch_id, data in enumerate(valid_loader):
            inputs, wav_features, label = data
            inputs = inputs.to(device)
            wav_features = wav_features.to(device)
            inputs = inputs.reshape(args.batch_tr, 1, inputs.shape[1], -1)

            with torch.no_grad():
                outputs = model.get_features(inputs)
                wav_features_proj = audio_proj(wav_features)
            
            p_5, p_1, loss, _ , pred_valid_label = segment_level_eval(outputs, 
                                                                      wav_features_proj, 
                                                                      label, 
                                                                      args.batch_tr, 
                                                                      args.loss_f, 
                                                                      p_5, 
                                                                      p_1, 
                                                                      loss, 
                                                                      args.temperature, 
                                                                      device)
            valid_id = valid_id + 1
        valid_acc5, valid_acc1, valid_loss = cal_acc(valid_id, p_5, p_1, loss, args.batch_tr)

        # save model
        if valid_loss < model_loss:
            model_loss = valid_loss
            last_best_epoch = epoch

            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(model, os.path.join(model_path, 'Net.pkl'))

        # early stopping
        if epoch - last_best_epoch > args.patience:
            break

    pred_label = []
    true_label = []

    

    '''################################### test ###################################'''
    net_path = os.path.join(model_path, 'Net.pkl')
    model = torch.load(net_path).to(device)

    model.eval()
    p_5 = 0
    p_1 = 0
    loss = 0
    test_id = 0

    test_seeg_features=[]
    test_audio_features=[]

    for batch_id, data in enumerate(test_loader):
        inputs, wav_features, label = data
        inputs = inputs.to(device)
        wav_features = wav_features.to(device)
        inputs = inputs.reshape((args.batch_te, 1, inputs.shape[1], -1))

        with torch.no_grad():
            outputs = model.get_features(inputs)
            wav_features_proj = audio_proj(wav_features)
            if args.forward_feature:
                test_seeg_features.append(outputs.cpu().numpy())
                test_audio_features.append(wav_features_proj.cpu().numpy())
        
        p_5, p_1, loss, _ , pred_test_label = segment_level_eval(outputs, 
                                                                 wav_features_proj, 
                                                                 label, 
                                                                 args.batch_te, 
                                                                 args.loss_f, 
                                                                 p_5, 
                                                                 p_1, 
                                                                 loss, 
                                                                 args.temperature, 
                                                                 device)
        test_id = test_id + 1
        pred_label.append(np.array(pred_test_label.cpu()).tolist())
        true_label.append(np.array(label).tolist())

    pred_label = np.array(pred_label)
    true_label = np.array(true_label)

    if args.forward_feature:
        test_seeg_features = np.vstack(test_seeg_features)
        test_audio_features = np.vstack(test_audio_features)
        if not os.path.exists(forward_feature_path):
            os.makedirs(forward_feature_path)
        np.save(os.path.join(forward_feature_path, f'{args.backbone}_seeg_forward_features.npy'), test_seeg_features)
        np.save(os.path.join(forward_feature_path, f'{args.backbone}_audio_forward_features.npy'), test_audio_features)
        np.save(os.path.join(forward_feature_path, f'{args.backbone}_forward_labels.npy'), true_label)

    if args.save_matrix_label == True:
        if not os.path.exists(matrix_label_path):  
            os.makedirs(matrix_label_path)
        np.savez(matrix_label_path + 'matrix_label.npz', pred_label=pred_label, true_label=true_label)

    test_acc5, test_acc1, test_loss = cal_acc(test_id, p_5, p_1, loss, args.batch_te)
    log_print('subject:{} seed:{} test: acc5: {:.2f}, acc1: {:.2f}'.format(args.subject_id, args.seed, test_acc5, test_acc1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SEEG_TJH')

    parser.add_argument('--subject_id', type=str, default='S02', help='Subject participating in the training')
    parser.add_argument('--device_system', type=str, default='Natus/V_48', help='The data acquisition system, Natus/V_48 | Neuracle/V_48')
    parser.add_argument('--label_type', type=str, default='word', help='Label type, represents what is being classified, \
                                                                        word (48 classes), initial_class (24 classes), \
                                                                        tone (4 classes), final (35 classes with imbalanced categories)')
    parser.add_argument('--class_num', type=int, default=48, help='Number of categories, automatically determined based on label_type')
    parser.add_argument('--channel_path', type=str, default='read_full', help='The folder where the data is saved after channel selection')
    parser.add_argument('--feature_type', type=str, default='hubertraw_8', help='Audio feature type')
    parser.add_argument('--chn', type=int, default=8, help='Number of SEEG channels')
    parser.add_argument('--time_sample_num', type=int, default=320, help='Number of time samples for each SEEG segment')

    parser.add_argument('--session_num', type=int, default=4, help='The number of sessions each subject participated in for the training')
    parser.add_argument('--fs', type=int, default=200, help='Sampling rate of the input data')
    parser.add_argument('--vad_half_window', type=float, default=0.5, help='VAD half-window length in seconds (s)')

    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--backbone', type=str, default='EEGNet', help='backbone model, EEGNet | DeepConvNet | ShallowConvNet | CNN-BiGRU | Conformer | DBConformer')
    parser.add_argument('--EEGNet_configs', type=int, nargs='+', default=[4, 16, 4, 64], help='EEGNet kernLength F1 D F2')
    parser.add_argument('--DeepConvNet_n_chs', type=int, default=[25, 50, 100, 200], help='Channel numbers for DeepConvNet layers')
    parser.add_argument('--ShallowConvNet_configs', type=int, nargs='+', default=[4, 75, 15], help='kernel_size and avg_pool2d for ShallowConvNet')
    parser.add_argument('--CNN_BiGRU_hidden_size', type=int, default=128, help='Hidden size for CNN-BiGRU model')
    parser.add_argument('--Conformer_configs', type=int, nargs='+', default=[128, 2], help='emb_size and depth for Conformer')
    parser.add_argument('--DBConformer_configs', type=int, nargs='+', default=[256, 32, 1, 1], help='emb_size patch_size tem_depth and chn_depth for Conformer')

    parser.add_argument('--batch_tr', type=int, default=48, help='Batch size during training')
    parser.add_argument('--batch_te', type=int, default=48, help='Batch size during testing')
    parser.add_argument('--l_r', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--l_r_proj', type=float, default=1e-3, help='Learning rate of Audio Projection module')
    parser.add_argument('--topk', type=int, default=5, help='Top_k accuracy')
    parser.add_argument('--max_epoch', type=int, default=100, help='Max epoch number')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--loss_f', type=str, default='CLIP', help='Loss function, CLIP | MSE')
    parser.add_argument('--temperature', type=float, default=20, help='1 / temperature coefficient')
    parser.add_argument('--out_dim', type=int, default=128, help='Output dimension of audio feature projection')
    parser.add_argument('--out_time', type=int, default=32, help='Output time length of audio feature projection')
    parser.add_argument('--random', type=bool, default=False, help='Whether to randomly shuffle the training audio')
    parser.add_argument('--save_matrix_label', type=bool, default=False, help='Whether to save test matrix label')
    parser.add_argument('--forward_feature', type=bool, default=False, help='Whether to save test forward features')

    parser.add_argument('--model_path', type=str, default=f'{os.path.dirname(__file__)}/model_save_multi_modality/', help='Path to save Model')
    parser.add_argument('--log_path', type=str, default=f'{os.path.dirname(__file__)}/log_save_multi_modality/', help='Path to save log')
    parser.add_argument('--matrix_label_path', type=str, default=f'{os.path.dirname(__file__)}/matrix_label_save_multi_modality/', help='Classification confusion matrix label save path')
    parser.add_argument('--forward_feature_path', type=str, default=f'{os.path.dirname(__file__)}/forward_feature_save_multi_modality/', help='Test forward features save path')
    parser.add_argument('--gpu_id', type=int, default=3, help='GPU id')

    args = parser.parse_args()

    random_init(args.seed)

    # today's log path
    current_date = datetime.date.today().strftime('%Y-%m-%d')
    args.log_path = os.path.join(args.log_path, current_date, str(args.seed), '')
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # log config
    logging.basicConfig(level=logging.DEBUG,
                        filename=args.log_path + 'record.log',
                        filemode='a',
                        format='%(asctime)s- %(levelname)s:-%(message)s')

    # device
    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print('=' * 30)
    log_print(f'device:{device}:{torch.cuda.current_device()}')
    torch.set_num_threads(4)

    train_baseline(args)