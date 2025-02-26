import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_cm_seeg(true, pred, name, acc):
    """Fig: Confusion matrix of seeg
    
    Parameters
    -----------
    true: 
        Truth labels
    pred: 
        Predicted labels
    name:
        Figure name
    """

    plt.figure(figsize=(10, 10))
    cm = confusion_matrix(y_true=true, y_pred=pred, normalize='true')
    plt.imshow(cm, cmap='Oranges', vmin=0, vmax=0.4)
    plt.title(f'Top-1 Accuracy: {acc}%', fontsize=34)
    plt.xlabel('Predicted label', fontsize=34)
    plt.ylabel('True label', fontsize=34)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()

    cbar = plt.colorbar(shrink=0.84, ticks=[0, 0.1, 0.2, 0.3, 0.4])
    cbar.ax.tick_params(labelsize=34)

    plt.savefig(f'{name}.eps', bbox_inches='tight', dpi=300)
    plt.savefig(f'{name}.png', bbox_inches='tight', dpi=300)

    plt.show()


def plot_cm_speech(true, pred, name, acc):
    """Fig: Confusion matrix of speech classification

    Parameters
    -----------
    true: 
        Truth labels
    pred: 
        Predicted labels
    name:
        Figure name
    """

    plt.figure(figsize=(10, 10))
    cm = confusion_matrix(y_true=true, y_pred=pred, normalize='true')
    plt.imshow(cm, cmap='Blues')
    plt.title(f'Top-1 Accuracy: {acc}%', fontsize=34)
    plt.xlabel('Predicted label', fontsize=34)
    plt.ylabel('Truth label', fontsize=34)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()

    cbar = plt.colorbar(shrink=0.84)
    cbar.ax.tick_params(labelsize=34)

    plt.savefig(f'{name}.eps', bbox_inches='tight', dpi=300)
    plt.savefig(f'{name}.png', bbox_inches='tight', dpi=300)

    plt.show()


if __name__ == '__main__':
    data = np.load('./Fig8_data.npz')
    S02_pred = data['S02_pred']
    S02_true = data['S02_true']
    S07_pred = data['S07_pred']
    S07_true = data['S07_true']
    S02_pred_speech = data['S02_pred_speech']
    S02_true_speech = data['S02_true_speech']
    S07_pred_speech = data['S07_pred_speech']
    S07_true_speech = data['S07_true_speech']

    Figname_list = ['Fig8a', 'Fig8b', 'Fig8c', 'Fig8d']
    ACC_list = [9.29, 1.74, 98.96, 45.83]

    plot_cm_seeg(S02_true, S02_pred, Figname_list[0], ACC_list[0])
    plot_cm_seeg(S07_true, S07_pred, Figname_list[1], ACC_list[1])
    plot_cm_speech(S02_true_speech, S02_pred_speech, Figname_list[2], ACC_list[2])
    plot_cm_speech(S07_true_speech, S07_pred_speech, Figname_list[3], ACC_list[3])