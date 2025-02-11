import matplotlib.pyplot as plt
import numpy as np

def plot_hubertlayer(c, b, c_std, b_std, type):
    """Fig: Effect of Parameters: hubertlayer

    Parameters
    -----------
    c: 
        accuracy of CAR
    b: 
        accuracy of Bipolar
    c_std:
        STDEV.S of CAR
    b_std:
        STDEV.S of Bipolar
    type:
        word | ini
    """
    
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.plot(x, c, color='b', linestyle='-', linewidth=2, marker='*', markersize=15, label='CAR')
    ax.plot(x, b, color='r', linestyle='--', linewidth=2, marker='s', markersize=15, label='Bipolar')

    ax.fill_between(x, b-1.05*b_std, b+1.05*b_std, alpha=0.1, color='r')
    ax.fill_between(x, c-1.05*c_std, c+1.05*c_std, alpha=0.1, color='b') # 2.571 / power(6, 1/2)

    plt.rcParams.update({'font.size':30})
    plt.xticks(range(1, x.__len__()+1), x, fontsize=30)
    plt.yticks(fontsize=30)

    plt.grid(visible=True,
             color='grey', 
             axis='y',   
             linestyle='--',
             linewidth=1,
             alpha=0.6)
    
    plt.xlabel('Layer Depth of HuBERT Model', fontsize=30)

    if type == 'word':
        plt.axhline(y=10.42, color='red', linestyle='-', linewidth=2, label="Chance level")
        plt.ylabel('Top-5 Accuracy (%)', fontsize=30)
        plt.ylim(9, 18)
        plt.legend(bbox_to_anchor=(0.5, 0.33), loc='center')
        plt.savefig('Fig9a.eps', bbox_inches='tight', dpi=300)
        plt.savefig('Fig9a.png', bbox_inches='tight', dpi=300)
        plt.show()
    elif type == 'ini':
        plt.axhline(y=20.83, color='red', linestyle='-', linewidth=2, label="Chance level")
        plt.ylabel('Top-5 Accuracy (%)', fontsize=30)
        plt.ylim(20, 27.5)
        plt.legend(bbox_to_anchor=(0.5, 0.3), loc='center')
        plt.savefig('Fig9b.eps', bbox_inches='tight', dpi=300)
        plt.savefig('Fig9b.png', bbox_inches='tight', dpi=300)
        plt.show()


if __name__ == '__main__':

    word_acc5_c     = np.array([14.65, 14.61, 15.02, 14.69, 15.71, 15.48, 16.25, 16.93, 16.55, 16.47, 15.35, 15.51])
    word_acc5_c_std = np.array([0.79, 0.74, 0.54, 0.88, 0.51, 0.69, 0.57, 0.66, 0.72, 0.77, 0.70, 0.70])
    word_acc5_b     = np.array([14.78, 14.75, 14.87, 14.67, 15.01, 15.36, 16.09, 16.42, 16.08, 15.14, 15.94, 14.45])
    word_acc5_b_std = np.array([0.56, 0.89, 0.59, 0.43, 0.50, 0.53, 0.69, 0.63, 0.66, 1.20, 0.43, 0.84])
    ini_acc5_c      = np.array([24.25, 24.46, 24.28, 24.74, 25.09, 24.75, 25.09, 26.23, 25.38, 25.53, 24.77, 24.20])
    ini_acc5_c_std  = np.array([1.03, 0.65, 0.61, 1.38, 1.02, 0.70, 0.74, 0.87, 0.72, 1.05, 0.27, 1.10])
    ini_acc5_b      = np.array([24.08, 24.31, 24.86, 24.78, 24.51, 24.64, 24.59, 25.21, 24.95, 24.33, 24.33, 23.13])
    ini_acc5_b_std  = np.array([0.45, 0.71, 1.43, 0.92, 1.29, 0.54, 1.32, 0.35, 1.17, 1.01, 1.03, 0.89])

    plot_hubertlayer(word_acc5_c, word_acc5_b, word_acc5_c_std, word_acc5_b_std, 'word')
    plot_hubertlayer(ini_acc5_c, ini_acc5_b, ini_acc5_c_std, ini_acc5_b_std, 'ini')