'''
Author: WangHongbinary
E-mail: hbwang22@gmail.com
Date: 2025-12-19 11:10:52
LastEditTime: 2025-12-25 16:31:42
Description: 
'''

import os
import torch
import numpy as np
import soundfile as sf
import csv
import librosa
import numpy as np
import os
import torch
import torchaudio

from torchaudio.transforms import Resample
from torch.utils.data import Dataset


class WavLoader_VM(Dataset):
    def __init__(self, 
                 fs=16000, 
                 feature_type='Sentencehubertraw_8', 
                 feature_path='',
                 root='/mnt/data1/whb/Public/VocalMind'):
        
        super().__init__()
        self.fs = fs
        self.feature_type = feature_type

        # load pre-trained speech model
        if 'hubert' in self.feature_type:
            bundle = torchaudio.pipelines.HUBERT_BASE

        if 'hubert' in self.feature_type:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = bundle.get_model().to(device)
        
        if 'Sentence' in self.feature_type:
            datatype = 'Original_Audio_Sentence'
            self.all_wav = vocalmind_wav(datatype=datatype)
            self.wav_len = 5
        elif 'Word' in self.feature_type:
            datatype = 'Original_Audio_Word'
            self.all_wav = vocalmind_wav(datatype=datatype)
            self.wav_len = 3

        print('all_wav.shape:', self.all_wav.shape)
        self.all_wav = torch.tensor(self.all_wav, dtype=torch.float32)

        feature_template = []
        if 'hubert' in self.feature_type:
            for i in range(len(self.all_wav)):
                temp_wav = self.all_wav[i][..., :int(self.wav_len*16000)].to(device)
                hubert_feature = trans_hubert(temp_wav, model, feature_type)
                feature_template.append(hubert_feature)
        self.feature = np.array(feature_template)

        if not os.path.exists(feature_path):
            os.makedirs(feature_path)
        feature_file = (feature_path + 'features.npy')
        np.save(feature_file, self.feature)


def trans_hubert(temp_wav, model, feature_type):
    """Extract hubert features

    Parameters
    -----------
    temp_wav: 
        audio segment to be processed
    model: 
        hubert model
    feature_type: 
        Audio feature type
    """

    with torch.inference_mode():
        hubert_features , _ = model.extract_features(temp_wav)
        hubert_features = torch.stack(hubert_features)
        layer_num = extract_layer_number(feature_type)
        hubert_feature = hubert_features[layer_num]
    hubert_feature = hubert_feature.cpu()
    hubert_feature = hubert_feature.reshape(hubert_feature.shape[1], hubert_feature.shape[2])
    hubert_feature = hubert_feature.T

    if not 'raw' in feature_type:
        hubert_feature = Resample(49, 197)(hubert_feature)

    if not 'woz' in feature_type:
        hubert_mean = torch.mean(hubert_feature, dim=1).reshape(hubert_feature.shape[0], 1)
        hubert_std = torch.std(hubert_feature, dim=1).reshape(hubert_feature.shape[0], 1)
        hubert_feature = (hubert_feature - hubert_mean) / hubert_std

    hubert_feature = np.array(hubert_feature)
    
    if not 'raw' in feature_type:
        if 'Sentence' in feature_type:
            hubert_feature = hubert_feature[..., :1000]
        elif 'Word' in feature_type:
            hubert_feature = hubert_feature[..., :600]

    return hubert_feature


def extract_layer_number(feature_type):
    """Extract layer number in the feature_type, e.g.hubert_8: 8

    Parameters
    -----------
    feature_type: 
        Audio feature type
    """

    underscore_index = feature_type.find('_')
    if underscore_index != -1:
        number_part = feature_type[underscore_index + 1:]
        return int(number_part)
    else:
        raise ValueError("no '_' in the str")


def load_features(feature_type, gpu_id, root='/mnt/data1/whb/Public/VocalMind'):
    feature_path = root + f'/wav_features/{feature_type}/'
    
    if not os.path.exists(feature_path):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        wav = WavLoader_VM(feature_type=feature_type, feature_path=feature_path)
        
    feature_file = (feature_path + 'features.npy')

    feature = np.load(feature_file, allow_pickle=True)
    feature = torch.tensor(feature, dtype=torch.float32)

    if 'Word' in feature_type:
        avg_feature = feature[72:77].mean(dim=0, keepdim=True)  # [1, d, T]
        feature = torch.cat([feature[:77], avg_feature, feature[77:]], dim=0)
    
    return feature


def vocalmind_wav(root='/mnt/data1/whb/Public/VocalMind', datatype='Original_Audio_Sentence'):
    """Load VocalMind dataset

    Parameters
    -----------
    root: 
        Path to load the VocalMind wav
    datatype:
        Type of wav to load
    """
    
    # load WAV
    file_path = os.path.join(root, datatype)
    wav_vm, sample_keys, fs = load_wav_vm(file_path)

    print("WAV VocalMind shape:", wav_vm.shape)
    # print("sample_keys:", sample_keys)

    print("fs:", fs)

    all_wav = wav_vm.transpose((0, 2, 1))
    all_wav = np.mean(all_wav, axis=1, keepdims=True)
    print("all_data shape:", all_wav.shape)

    new_all_wav = librosa.resample(all_wav, orig_sr=fs, target_sr=16000)
    print("new_all_wav shape:", new_all_wav.shape)

    return new_all_wav


def parse_key(filename):
    """
    Vocalized_BuShiNiXiangDeNaYang_1.npy -> (task, sentence, repetition)
    """
    name = os.path.splitext(filename)[0]
    task, sentence, rep = name.split('_', 2)
    return task, sentence, int(rep)


def load_wav_vm(wav_dir):

    files = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]
    files = sorted(files, key=parse_key)

    # print('files:', files)

    wav_list = []
    sample_keys = []

    for f in files:
        wav, fs = sf.read(os.path.join(wav_dir, f))
        wav_list.append(wav)
        sample_keys.append(parse_key(f))

    wav_vm = np.stack(wav_list, axis=0)
    return wav_vm, sample_keys, fs


if __name__ == "__main__":
    gpu_id = 6

    feature_types = ['Wordhubertraw_0',
                     'Wordhubertraw_2',
                     'Wordhubertraw_4',
                     'Wordhubertraw_6',
                     'Wordhubertraw_8',
                     'Wordhubertraw_10'] # Sentencehubertraw_8 | Wordhubertraw_8

    for feature_type in feature_types:
        feature = load_features(feature_type=feature_type, gpu_id=gpu_id)
        print(f'feature_{feature_type}.shape:{feature.shape}')