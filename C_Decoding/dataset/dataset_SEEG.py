import os
import torch
import numpy as np
from torch.utils.data import Dataset

from scipy.linalg import fractional_matrix_power


def seeg_dataset(device_system='Neuracle/V_48',
                 subject_id='S08', 
                 session_num=4, 
                 channel_path='read_good', 
                 label_type='word', 
                 ea=False,
                 root='/mnt/data1/whb/SACM_Data/processed'):
    """Load SEEG dataset

    Parameters
    -----------
    device_system: 
        The data acquisition system, Natus/V_48 | Neuracle/V_48
    subject_id: 
        Subject participating in the training
    session_num: 
        The number of sessions each subject participated in for the training
    channel_path: 
        The folder where the data is saved after channel selection
    label_type: 
        Label type, represents what is being classified
    ea: 
        Whether to use EA alignment during the data loading stage
    root: 
        Path to load the processed data
    """
    
    # load SEEG
    for ses in range(session_num):
        file_path = root + f'/{device_system}/{subject_id}/read/session{ses+1}/{channel_path}/seeg'
        if os.path.exists(file_path):
            file_name = file_path + '/envelope_word.npy'
            print('loading:----{}----session{}----'.format(subject_id, ses))
            data = np.load(file_name)

            if ea == True:
                data = EA(data)
            if ses == 0:
                all_data = data
            else:
                all_data = np.concatenate((all_data, data), axis=0)
        else:
            print("no such file")
            continue
    all_data = torch.tensor(all_data, dtype=torch.float32)

    # load label
    for ses in range(session_num):
        label_path = root + f'/{device_system}/{subject_id}/read/session{ses+1}/trans_label'
        if os.path.exists(label_path):
            label_file = os.path.join(label_path, 'trans_label.npz')
            label_read = np.load(label_file)

            if ses == 0:
                word_label = label_read['word']
                initial_class_label = label_read['initial_class']
                tone_label = label_read['tone']
                initial_label = label_read['initial']
                final_label = label_read['final']
            else:
                word_label = np.concatenate((word_label, label_read['word']), axis=0)
                initial_class_label = np.concatenate((initial_class_label, label_read['initial_class']), axis=0)
                tone_label = np.concatenate((tone_label, label_read['tone']), axis=0)
                initial_label = np.concatenate((initial_label, label_read['initial']), axis=0)
                final_label = np.concatenate((final_label, label_read['final']), axis=0)
        else:
            print("no such file")
            continue

    word_label = word_label.reshape(-1)
    initial_class_label = initial_class_label.reshape(-1)
    tone_label = tone_label.reshape(-1)
    initial_label = initial_label.reshape(-1)
    final_label = final_label.reshape(-1)

    label_dict = {'word': word_label, 
                  'initial_class': initial_class_label, 
                  'tone': tone_label, 
                  'initial': initial_label, 
                  'final': final_label}
    
    all_label = label_dict[label_type]
    return all_data, all_label, all_data.shape[1]


def EA(x):
    """EA alignment

    Parameters
    -----------
    x: 
        brain data
    """

    print('EA processing')
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5) + (0.00000001) * np.eye(x.shape[1])
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    
    return XEA


class MakeSet(Dataset):
    def __init__(self, seeg, wav_features, label):
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

        return self.seeg[index], self.wav_features[index], self.label[index]
    
    
    def __len__(self):
        """Return the length of the dataset
        """
        
        return len(self.seeg)


if __name__ == '__main__':
    seeg, label, in_channels = seeg_dataset(dataset='SEEG', 
                                            subject_id='S08', 
                                            session_num=4, 
                                            channel_path='read_good', 
                                            label_type='word', 
                                            fs=200, 
                                            vad_half_window=0.3, 
                                            sliding_step=0.1)
    print('seeg.shape:', seeg.shape)
    print('label.shape:', label.shape)
    print('in_channels', in_channels)