import torch
import numpy as np
import argparse

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from wav_features_load import load_wav_features


def ml_clf(args):
    """LDA for the audio features

    Parameters
    -----------
    args: 
        Parameters for LDA
    """

    features, labels = load_wav_features(args.root_path, 
                                         args.device_system, 
                                         args.subject_id, 
                                         args.feature_type, 
                                         args.session_num, 
                                         args.label_type,
                                         args.print_info)
    features = np.mean(features, axis=-1)
    print(features.shape)

    train_index = []
    test_index = []
    for i in range(args.session_num):
        train_start = i * 480
        train_end = train_start + 432
        train_index.append(np.arange(train_start, train_end))
    
        test_start = train_end
        test_end = test_start + 48
        test_index.append(np.arange(test_start, test_end))

    tr_feature = features[train_index]
    te_feature = features[test_index]
    tr_label = labels[train_index]
    te_label = labels[test_index]

    lda_model = LinearDiscriminantAnalysis()
    lda_model.fit(tr_feature, tr_label)
    lda_pred = lda_model.predict(te_feature)
    lda_accuracy = round(accuracy_score(te_label, lda_pred), 3)
    print('LDA', lda_accuracy)

    # np.savez('/home/HBWang/MyWork/Speech_SEEG/MyCode_SEEG/V_48/A_Preprocessing/Figure_code/Fig8_wav02.npz', S02_pred_speech=lda_pred, S02_true_speech=te_label)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WAV_train')
    parser.add_argument('--root_path', type=str, default='/mnt/data1/whb/SACM_Data/processed', help='Path to load the audio features')
    parser.add_argument('--device_system', type=str, default='Natus/V_48', help='The data acquisition system, Natus/V_48 | Neuracle/V_48')
    parser.add_argument('--label_type', type=str, default='word', help='Label type, represents what is being classified, \
                                                                        word (48 classes), initial_class (24 classes), \
                                                                        tone (4 classes), tone (4 classes), final (35 classes with imbalanced categories)')
    parser.add_argument('--feature_type', type=str, default='hubert_8', help='Audio feature type')
    parser.add_argument('--subject_id', type=str, default='S09', help='Subject participating in the training')
    parser.add_argument('--session_num', type=int, default=4, help='The number of sessions each subject participated in for the training')
    parser.add_argument('--print_info', type=bool, default=True, help='Whether to print the info')

    args = parser.parse_args()

    # device
    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ml_clf(args)