import os
import numpy as np


def load_wav_features(root_path='/mnt/data1/whb/SACM_Data/processed', 
                      device_system='Natus/V_48',
                      subject_id='S08', 
                      feature_type='hubert_8', 
                      session_num=4, 
                      label_type='word', 
                      print_info=True):
    """Load the audio features

    Parameters
    -----------
    root_path: 
        Path to load the audio features
    device_system: 
        The data acquisition system, Natus/V_48 | Neuracle/V_48
    subject_id: 
        Subject participating in the training
    feature_type: 
        Audio feature type
    session_num: 
        The number of sessions each subject participated in for the training
    label_type: 
        Label type, represents what is being classified
    print_info: 
        Whether to print the info
    """

    wav_features_file = f'{root_path}/{device_system}/{subject_id}/read/wav_features/{feature_type}/features.npy'
    wav_features = np.load(wav_features_file) # (1920, 768, 320)

    for ses in range(session_num):
        label_path = f'{root_path}/{device_system}/{subject_id}/read/session{ses+1}/trans_label'
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

    label_dict = {'word': word_label, 
                  'initial_class': initial_class_label, 
                  'tone': tone_label, 
                  'initial': initial_label, 
                  'final': final_label}
    
    wav_label = label_dict[label_type] # (1920,)

    if print_info == True:
        print('wav_features.shape:', wav_features.shape)
        print('wav_label.shape:', wav_label.shape)

    return wav_features, wav_label


if __name__ == '__main__':
    features, labels = load_wav_features()
