import numpy as np
import os
import torch
import torchaudio

from torchaudio.transforms import Resample
from torch.utils.data import Dataset

class WavLoader(Dataset):
    def __init__(self, 
                 device_system='Neuracle/V_48',
                 subject_id='S08', 
                 session_num=4, 
                 fs=16000, 
                 feature_type='hubert_8', 
                 vad_half_window=0.3,
                 feature_path='',
                 root='/mnt/data1/whb/SACM_Data/processed'):
        """Extract audio features

        Parameters
        -----------
        device_system: 
            The data acquisition system, Natus/V_48 | Neuracle/V_48
        subject_id: 
            Subject participating in the training
        session_num: 
            The number of sessions each subject participated in for the training
        fs: 
            New sampling rate of the audio
        feature_type: 
            Audio feature type
        vad_half_window: 
            VAD half-window length in seconds (s)
        feature_path: 
            Path to save the audio features
        root: 
            Path to load the processed data
        """
        
        super().__init__()
        self.device_system = device_system
        self.fs = fs
        self.subject_id = subject_id
        self.feature_type = feature_type
        self.vad_half_window = vad_half_window

        # load pre-trained speech model
        if 'xlsr' in self.feature_type:
            bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
        elif 'hubert' in self.feature_type:
            bundle = torchaudio.pipelines.HUBERT_BASE

        if 'xlsr' in self.feature_type or 'hubert' in self.feature_type:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = bundle.get_model().to(device)
        
        for session in range(session_num):
            print('loading:----({})----session{}----'.format(self.subject_id, session))
            file_path = root + f'/{self.device_system}/{self.subject_id}/read/session{session+1}/wav/audio'
            if os.path.exists(file_path):
                file_name = file_path + '/wav_word.npy'
                load_wav = np.load(file_name)
                if session == 0:
                    self.all_wav = load_wav
                else:
                    self.all_wav = np.concatenate((self.all_wav, load_wav), axis=0)
            else:
                print("no such file")
                continue
        print('all_wav.shape:', self.all_wav.shape)

        self.all_wav = torch.tensor(self.all_wav, dtype=torch.float32)
        feature_template = []
        if 'xlsr' in self.feature_type:
            for i in range(len(self.all_wav)):
                temp_wav = self.all_wav[i][..., :25360].to(device)
                xlsr_feature = trans_xlsr(temp_wav, model, feature_type)
                feature_template.append(xlsr_feature)

        elif 'hubert' in self.feature_type:
            for i in range(len(self.all_wav)):
                temp_wav = self.all_wav[i][..., :25360].to(device)
                hubert_feature = trans_hubert(temp_wav, model, feature_type)
                feature_template.append(hubert_feature)
        self.feature = np.array(feature_template)

        if not os.path.exists(feature_path):
            os.makedirs(feature_path)
        feature_file = (feature_path + 'features.npy')
        np.save(feature_file, self.feature)
            

def trans_xlsr(temp_wav, model, feature_type):
    """Extract wav2vec2.0_xlsr features

    Parameters
    -----------
    temp_wav: 
        audio segment to be processed
    model: 
        wav2vec2.0_xlsr model
    feature_type: 
        Audio feature type
    """

    with torch.inference_mode():
        xlsr_features , _ = model.extract_features(temp_wav)
        xlsr_features = torch.stack(xlsr_features)
        layer_num = extract_layer_number(feature_type)
        xlsr_feature = xlsr_features[layer_num]
    xlsr_feature = xlsr_feature.cpu()
    xlsr_feature = xlsr_feature.reshape(xlsr_feature.shape[1], xlsr_feature.shape[2])
    xlsr_feature = xlsr_feature.T
    xlsr_feature = Resample(49, 200)(xlsr_feature)

    xlsr_mean = torch.mean(xlsr_feature, dim=1).reshape(xlsr_feature.shape[0], 1)
    xlsr_std = torch.std(xlsr_feature, dim=1).reshape(xlsr_feature.shape[0], 1)
    xlsr_feature = (xlsr_feature - xlsr_mean) / xlsr_std

    xlsr_feature = np.array(xlsr_feature)
    xlsr_feature = xlsr_feature[..., :320]

    return xlsr_feature


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
        hubert_feature = Resample(49, 200)(hubert_feature)

    if not 'woz' in feature_type:
        hubert_mean = torch.mean(hubert_feature, dim=1).reshape(hubert_feature.shape[0], 1)
        hubert_std = torch.std(hubert_feature, dim=1).reshape(hubert_feature.shape[0], 1)
        hubert_feature = (hubert_feature - hubert_mean) / hubert_std

    hubert_feature = np.array(hubert_feature)
    
    if not 'raw' in feature_type:
        hubert_feature = hubert_feature[..., :320]

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


def load_features(device_system, feature_type, subject_id, vad_half_window, gpu_id, root='/mnt/data1/whb/SACM_Data/processed'):
    """Extract features based on the feature_type

    Parameters
    -----------
    device_system: 
        The data acquisition system, Natus/V_48 | Neuracle/V_48
    feature_type: 
        Audio feature type
    subject_id: 
        Subject participating in the training
    vad_half_window: 
        VAD half-window length in seconds (s)
    gpu_id:
        GPU id
    root: 
        Path to load the processed data
    """
    
    feature_path = root + f'/{device_system}/{subject_id}/read/wav_features/{feature_type}/'
    
    if not os.path.exists(feature_path):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        wav = WavLoader(device_system=device_system,
                        subject_id=subject_id, 
                        feature_type=feature_type, 
                        vad_half_window=vad_half_window,
                        feature_path=feature_path)
        
    feature_file = (feature_path + 'features.npy')

    feature = np.load(feature_file, allow_pickle=True)
    feature = torch.tensor(feature, dtype=torch.float32)
    return feature


if __name__ == '__main__':
    torch.cuda.set_device(2)
    subject_ids = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08']
    device_systems = ['Natus/V_48', 'Natus/V_48', 'Natus/V_48', 'Natus/V_48', 'Natus/V_48', 'Natus/V_48', 'Neuracle/V_48', 'Neuracle/V_48']

    # feature_types = ['hubert_0', 
    #                  'hubert_1', 
    #                  'hubert_2', 
    #                  'hubert_3', 
    #                  'hubert_4', 
    #                  'hubert_5', 
    #                  'hubert_6', 
    #                  'hubert_7', 
    #                  'hubert_8', 
    #                  'hubert_9', 
    #                  'hubert_10', 
    #                  'hubert_11']

    feature_types = ['hubert_8']

    for s, subject_id in enumerate(subject_ids):
        for feature_type in feature_types:
            feature = load_features(device_system=device_systems[s],
                                    feature_type=feature_type, 
                                    subject_id=subject_id, 
                                    vad_half_window=0.5,
                                    gpu_id=0)
            print(f'subject_{subject_id}_feature_{feature_type}.shape:{feature.shape}')