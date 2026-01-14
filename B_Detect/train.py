import argparse
import datetime
import logging
import numpy as np
import os
import random
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from net import *
from dataset.dataset_SEEG import MakeSet, seeg_dataset
from util import log_print, top_k


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


def train_net(args):
    """Train, valid, test

    Parameters
    -----------
    args: 
        Parameters for model training, validation and testing
    """

    log_print(args)

    model_path = os.path.join(args.model_path, str(args.seed), '')
    all_data, all_detect_label, in_channels = seeg_dataset(args.device_system,
                                                           args.subject_id, 
                                                           args.session_num, 
                                                           args.channel_path)
    
    k_folds = 5
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=args.seed)

    acc_k = 0
    acc_1 = 0

    log_print('training start')
    for train_index, test_index in kf.split(all_data, all_detect_label):
        train_data = all_data[train_index]
        train_label = all_detect_label[train_index]

        # random
        if args.random == True:
            np.random.shuffle(train_label)

        test_data = all_data[test_index]
        test_label = all_detect_label[test_index]

        train_data, valid_data, train_label, valid_label = train_test_split(train_data, 
                                                                            train_label, 
                                                                            test_size=0.2, 
                                                                            random_state=args.seed, 
                                                                            stratify=train_label)

        train_set = MakeSet(train_data, train_label)
        valid_set = MakeSet(valid_data, valid_label)
        test_set = MakeSet(test_data, test_label)

        train_loader = DataLoader(train_set, args.batch_tr, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_set, args.batch_tr, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_set, args.batch_te, shuffle=False, drop_last=False)

        model = EEGNet_new(classes_num=args.class_num, 
                           in_channels=in_channels, 
                           time_step=args.sample_len,
                           kernLenght=4,
                           F1=16,
                           D=4,
                           F2=64,
                           dropout_size=0.5).to(device)

        model.apply(weights_init)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.l_r, weight_decay=0.0001)

        model_loss = np.inf
        last_best_epoch = 0

        for epoch in range(args.max_epoch):
            '''################################### train ###################################'''
            model.train()

            p_k = 0
            p_1 = 0
            train_acck = 0
            train_acc1 = 0
            train_loss = 0
            train_id = 0

            for batch_id, data in enumerate(train_loader):
                inputs, label = data
                inputs = inputs.to(torch.float32).to(device)
                label = label.to(device)

                inputs = inputs.reshape(args.batch_tr, 1, inputs.shape[1], -1)
                label_pred = model(inputs)

                ce_loss = F.cross_entropy(label_pred, label)
                loss = ce_loss

                p_k = top_k(label_pred, label, args.topk)
                p_1 = top_k(label_pred, label, 1)

                train_loss = train_loss + ce_loss
                train_acck = train_acck + p_k
                train_acc1 = train_acc1 + p_1

                train_id = train_id + 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_acck = 100 * train_acck / train_id
            train_acc1 = 100 * train_acc1 / train_id
            train_loss = train_loss / train_id

            '''################################### valid ###################################'''
            model.eval()

            p_k = 0
            p_1 = 0
            valid_acck = 0
            valid_acc1 = 0
            valid_loss = 0
            valid_id = 0

            for batch_id, data in enumerate(valid_loader):
                inputs, label = data
                inputs = inputs.to(torch.float32).to(device)
                label = label.to(device)

                inputs = inputs.reshape(args.batch_tr, 1, inputs.shape[1], -1)
                with torch.no_grad():
                    label_pred = model(inputs)

                ce_loss = F.cross_entropy(label_pred, label)
                loss = ce_loss

                p_k = top_k(label_pred, label, args.topk)
                p_1 = top_k(label_pred, label, 1)

                valid_loss = valid_loss + ce_loss
                valid_acck = valid_acck + p_k
                valid_acc1 = valid_acc1 + p_1

                valid_id = valid_id + 1

            valid_acck = 100 * valid_acck / valid_id
            valid_acc1 = 100 * valid_acc1 / valid_id
            valid_loss = valid_loss / valid_id

            if valid_loss < model_loss:
                model_loss = valid_loss
                last_best_epoch = epoch

                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                torch.save(model, os.path.join(model_path, 'Net.pkl'))
                
            if epoch - last_best_epoch > args.patience:
                break

        '''################################### test ###################################'''
        net_path = os.path.join(model_path, 'Net.pkl')
        model = torch.load(net_path).to(device)

        model.eval()
        p_k = 0
        p_1 = 0
        test_acck = 0
        test_acc1 = 0
        test_id = 0

        for batch_id, data in enumerate(test_loader):
            inputs, label = data
            inputs = inputs.to(torch.float32).to(device)
            label = label.to(device)

            inputs = inputs.reshape((args.batch_te, 1, inputs.shape[1], -1))
            with torch.no_grad():
                label_pred = model(inputs)

            p_k = top_k(label_pred, label, args.topk)
            p_1 = top_k(label_pred, label, 1)

            test_acck = test_acck + p_k
            test_acc1 = test_acc1 + p_1

            test_id = test_id + 1

        test_acck = 100 * test_acck / test_id
        test_acc1 = 100 * test_acc1 / test_id

        acc_k = acc_k + test_acck
        acc_1 = acc_1 + test_acc1

    acc_k = acc_k / k_folds
    acc_1 = acc_1 / k_folds

    log_print('{} folds avg: subject:{} seed:{} test: acc5: {:.2f}, acc1: {:.2f}'.format(k_folds, args.subject_id, args.seed, acc_k, acc_1))
    log_print('=' * 30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SEEG_TJH')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--device_system', type=str, default='Neuracle/V_48', help='The data acquisition system, Natus/V_48 | Neuracle/V_48')
    parser.add_argument('--label_type', type=str, default='detect', help='0|1, speech or non-speech')
    parser.add_argument('--class_num', type=int, default=2, help='Number of categories, automatically determined based on label_type')
    parser.add_argument('--channel_path', type=str, default='read_good_320', help='The folder where the data is saved after channel selection')
    parser.add_argument('--out_channels', type=int, default=120, help='Feature dimension of the network output')
    parser.add_argument('--subject_id', type=str, default='S08', help='Subject participating in the training')
    parser.add_argument('--session_num', type=int, default=4, help='The number of sessions each subject participated in for the training')
    parser.add_argument('--fs', type=int, default=200, help='Sampling rate of the input data')
    parser.add_argument('--sample_len', type=int, default=100, help='Length of the input data, 100=0.5*200')

    parser.add_argument('--batch_tr', type=int, default=32, help='Batch size during training')
    parser.add_argument('--batch_te', type=int, default=32, help='Batch size during testing')
    parser.add_argument('--l_r', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max_epoch', type=int, default=1000, help='Max epoch number')
    parser.add_argument('--patience', type=int, default=30, help='Patience for early stopping')
    parser.add_argument('--loss_f', type=str, default='CE', help='Loss function, cross_entropy')
    parser.add_argument('--topk', type=int, default=1, help='Top_k accuracy')
    parser.add_argument('--ea', type=bool, default=False, help='Whether to use EA alignment during the data loading stage')
    parser.add_argument('--random', type=bool, default=False, help='Whether to randomly shuffle the label')

    parser.add_argument('--model_path', type=str, default=f'{os.path.dirname(__file__)}/model_save/', help='Path to save Model')
    parser.add_argument('--log_path', type=str, default=f'{os.path.dirname(__file__)}/log_save/', help='Path to save log')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')
    args = parser.parse_args()

    random_init(args.seed)

    label_class = {'detect': 2}
    args.class_num = label_class[args.label_type]

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

    train_net(args)