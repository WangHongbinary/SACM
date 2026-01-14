'''
Author: WangHongbinary
E-mail: hbwang22@gmail.com
Date: 2025-12-19 10:46:53
LastEditTime: 2025-12-22 19:00:17
Description: VocalMind SEEG dataset loader
'''

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import csv

def vocalmind_dataset(root='/mnt/data1/whb/Public/VocalMind',
                      datatype='Processed_sEEG_Vocalized_Sentence'):
    """Load VocalMind dataset

    Parameters
    -----------
    root: 
        Path to load the VocalMind data
    datatype:
        Type of data to load
    """
    
    # load SEEG
    file_path = os.path.join(root, datatype)

    seeg_vm, sample_keys = load_seeg_vm(file_path)
    labels, sentence2label = build_labels(sample_keys)
    # print("SEEG VocalMind shape:", seeg_vm.shape)

    all_data = torch.tensor(seeg_vm, dtype=torch.float32)
    all_data = all_data.permute(0, 2, 1)

    if datatype == 'Processed_sEEG_Vocalized_Word':
        avg_sample = all_data[72:77].mean(dim=0, keepdim=True)  # [1, C, T]
        # print("labels[72:77]", labels[72:77])
        all_data = torch.cat([all_data[:77], avg_sample, all_data[77:]], dim=0)
        labels = np.concatenate([labels[:77], labels[76:77], labels[77:]], axis=0)
        # print("labels[71:79]", labels[71:79])

    return all_data, labels, sentence2label


def parse_key(filename):
    """
    Vocalized_BuShiNiXiangDeNaYang_1.npy -> (task, sentence, repetition)
    """
    name = os.path.splitext(filename)[0]
    task, sentence, rep = name.split('_', 2)
    return task, sentence, int(rep)


def load_seeg_vm(seeg_dir):
    files = [f for f in os.listdir(seeg_dir) if f.endswith(".npy")]
    files = sorted(files, key=parse_key)

    # print('files:', files)
    
    seeg_list = []
    sample_keys = []

    for f in files:
        data = np.load(os.path.join(seeg_dir, f))
        seeg_list.append(data)
        sample_keys.append(parse_key(f))

    seeg_matrix = np.stack(seeg_list, axis=0)
    return seeg_matrix, sample_keys


def build_labels(sample_keys):
    sentences = sorted({key[1] for key in sample_keys})
    sentence2label = {s: i for i, s in enumerate(sentences)}
    labels = np.array([sentence2label[key[1]] for key in sample_keys])

    return labels, sentence2label

class MakeSet(Dataset):
    def __init__(self, seeg, wav_features=None, label=None):
        """Create dataset

        Parameters
        -----------
        seeg: 
            SEEG data
        wav_features: 
            Audio features data
        label:
            Label data
        """

        super().__init__()
        self.seeg = seeg
        self.wav_features = wav_features
        self.label = label


    def __getitem__(self, index):
        """Return the required items in order 

        Parameters
        -----------
        index: 
            Item index
        """

        seeg_item = self.seeg[index]
        label_item = self.label[index] if self.label is not None else None

        if self.wav_features is None:
            return seeg_item, label_item
        else:
            return seeg_item, self.wav_features[index], label_item
        
    def __len__(self):
        """Return the length of the dataset
        """
        
        return len(self.seeg)

if __name__ == "__main__":
    datatype = 'Processed_sEEG_Imagined_Sentence' # Processed_sEEG_Vocalized_Sentence | Processed_sEEG_Vocalized_Word | Processed_sEEG_Mimed_Sentence | Processed_sEEG_Imagined_Sentence
    all_data, labels, sentence2label = vocalmind_dataset(datatype=datatype)

    print("all_data shape:", all_data.shape)
    print("Labels shape:", labels.shape)