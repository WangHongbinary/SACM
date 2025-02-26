import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_individual_bar():
    """Fig: Mean squared amplitude for speech and non-speech audio segments
    """

    data = np.load('./Fig6_data.npz')
    Amplitude_word = data['Amplitude_word']
    Amplitude_rest = data['Amplitude_rest']

    word1, word2, word3, word4, word5, word6, word7, word8 = np.array_split(Amplitude_word, 8)
    rest1, rest2, rest3, rest4, rest5, rest6, rest7, rest8 = np.array_split(Amplitude_rest, 8)

    word_means = [np.mean(word1), np.mean(word2), np.mean(word3), np.mean(word4), 
                  np.mean(word5), np.mean(word6), np.mean(word7), np.mean(word8), np.mean(Amplitude_word)]
    rest_means = [np.mean(rest1), np.mean(rest2), np.mean(rest3), np.mean(rest4),
                  np.mean(rest5), np.mean(rest6), np.mean(rest7), np.mean(rest8), np.mean(Amplitude_rest)]

    all_means = []
    labels = []
    colors = []

    for i in range(len(word_means)):
        all_means.append(word_means[i])
        all_means.append(rest_means[i])
        labels.append(f'{i+1}')
        colors.append('#1D43A3')
        colors.append('#FF7F0E')
    labels[-1] = 'Avg.'

    plt.figure(figsize=(20, 10))
    sns.barplot(x=np.arange(len(all_means)), y=all_means, palette=colors)

    plt.yscale('log')
    plt.ylim(0, 150)
    xticks_pos = np.arange(0.5, len(all_means), 2)
    
    plt.xticks(xticks_pos, labels, fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlabel('Subject', fontsize=32)
    plt.ylabel('Mean Squared Amplitude', fontsize=32)

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='#1D43A3', lw=20, label='Speech'),
                       Line2D([0], [0], color='#FF7F0E', lw=20, label='Non-Speech')]
    plt.legend(handles=legend_elements, fontsize=32, borderpad=0.6, ncol=2, loc='upper center')

    plt.tight_layout()
    plt.savefig('Fig6.eps', bbox_inches='tight', dpi=300)
    plt.savefig('Fig6.png', bbox_inches='tight', dpi=300)

    plt.show()

if __name__ == '__main__':
    plot_individual_bar()