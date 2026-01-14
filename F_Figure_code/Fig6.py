'''
Author: WangHongbinary
E-mail: hbwang22@gmail.com
Date: 2026-01-08 17:00:21
LastEditTime: 2026-01-14 10:22:19
Description: Fig.6 of SACM: SEEG-Audio Contrastive Matching for Chinese Speech Decoding
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

if __name__ == '__main__':

    temperature_n = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1])

    # avg
    acc_hustmind  = np.array([13.09, 13.88, 15.46, 17.12, 16.65, 16.44, 15.70, 15.07])
    acc_vocalmind = np.array([17.50, 18.33, 24.17, 30.00, 26.67, 19.17, 26.67, 19.17])

    # std
    std_hustmind  = np.array([0.37, 0.42, 0.67, 0.52, 0.52, 0.47, 0.46, 0.72])
    std_vocalmind = np.array([4.18, 4.08, 4.92, 5.48, 5.16, 5.85, 9.31, 3.76])

    bar_width = 0.35
    x = np.arange(len(temperature_n))

    plt.figure(figsize=(20, 12))

    # HUST-MIND
    plt.bar(
        x - bar_width / 2,
        acc_hustmind,
        width=bar_width,
        yerr=std_hustmind,
        capsize=10,
        label='HUST-MIND',
        color='#1D43A3'
    )

    # VocalMind
    plt.bar(
        x + bar_width / 2,
        acc_vocalmind,
        width=bar_width,
        yerr=std_vocalmind,
        capsize=10,
        label='VocalMind',
        color='#8FA3E6'
    )

    plt.xticks(x, temperature_n)
    plt.tick_params(axis='x', labelsize=30)
    plt.tick_params(axis='y', labelsize=30)
    plt.xlabel('Temperature τ', fontsize=30)
    plt.ylabel('Accuracy (%)', fontsize=30)
    plt.ylim(0, 45)

    plt.grid(axis='y', linestyle='--', alpha=1.0)

    legend_elements = [Line2D([0], [0], color='#1D43A3', lw=30, label='HUST-MIND'),
                       Line2D([0], [0], color='#8FA3E6', lw=30, label='VocalMind')]
    plt.legend(handles=legend_elements, fontsize=30, borderpad=0.8, ncol=2, loc='upper center')

    plt.tight_layout()
    plt.savefig('temperature.eps', bbox_inches='tight', dpi=300)
    plt.savefig('temperature.png', bbox_inches='tight', dpi=300)

    plt.show()