import os
import numpy as np
from torch.utils.data import Dataset

def seeg_dataset(device_system='Natus/V_48',
                 subject_id='S08', 
                 session_num=4, 
                 channel_path='read_good', 
                 root='/mnt/data1/whb/Ours/SACM_Data/processed'):
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
    root: 
        Path to load the processed data
    """
    
    # Load SEEG
    for ses in range(session_num):
        file_path = root + f'/{device_system}/{subject_id}/read/session{ses+1}/{channel_path}/seeg_vad'
        if os.path.exists(file_path):
            word_file_name = file_path + '/envelope_word.npy'
            rest_file_name = file_path + '/envelope_rest.npy'

            print('loading:----{}----session{}----'.format(subject_id, ses))
            word_data = np.load(word_file_name)
            rest_data = np.load(rest_file_name)

            if ses == 0:
                all_word_data = word_data
                all_rest_data = rest_data
            else:
                all_word_data = np.concatenate((all_word_data, word_data), axis=0)
                all_rest_data = np.concatenate((all_rest_data, rest_data), axis=0)
        else:
            print("no such file")
            continue
    
    word_label = np.zeros(all_word_data.shape[0], dtype=int)
    rest_label = np.ones(all_rest_data.shape[0], dtype=int)

    all_data = np.concatenate((all_word_data, all_rest_data), axis=0)
    all_detect_label = np.concatenate((word_label, rest_label), axis=0)

    return all_data, all_detect_label, all_data.shape[1]


class MakeSet(Dataset):
    def __init__(self, data, label):
        """Create dataset

        Parameters
        -----------
        data: 
            SEEG data
        label:
            Label data
        """

        super().__init__()
        self.data = data
        self.label = label


    def __getitem__(self, index):
        """Return the required items in order 

        Parameters
        -----------
        index: 
            Item index
        """

        return self.data[index], self.label[index]
    

    def __len__(self):
        """Return the length of the dataset
        """

        return len(self.data)


if __name__ == '__main__':
    all_data, detect_label, in_channels = seeg_dataset(device_system='Natus/V_48',
                                                       subject_id='S08', 
                                                       session_num=4, 
                                                       channel_path='read_good_single_320')
    print('all_data:', all_data)
    print('all_data.shape:', all_data.shape)
    print('detect_label', detect_label)
    print('detect_label.shape:', detect_label.shape)
    print('in_channels:', in_channels)