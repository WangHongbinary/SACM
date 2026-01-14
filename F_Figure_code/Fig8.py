'''
Author: WangHongbinary
E-mail: hbwang22@gmail.com
Date: 2026-01-06 16:41:42
LastEditTime: 2026-01-09 15:37:10
Description: Fig.8 of SACM: SEEG-Audio Contrastive Matching for Chinese Speech Decoding
'''

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    data = {"S1": {"A": 0.057, "B": 0.068, "C": 0.071, "D": 0.090, "E": None,  "F": None,  "SMC": 0.275},
            "S2": {"A": 0.001, "B": 0.014, "C": 0.114, "D": 0.072, "E": 0.043, "F": None,  "SMC": 0.397},
            "S3": {"A": 0.006, "B": 0.039, "C": 0.020, "D": 0.039, "E": 0.019, "F": 0.014, "SMC": 0.183},
            "S4": {"A": 0.272, "B": 0.022, "C": 0.060, "D": 0.027, "E": 0.079, "F": None,  "SMC": 0.112},
            "S5": {"A": 0.336, "B": 0.061, "C": 0.038, "D": 0.031, "E": 0.195, "F": 0.172, "SMC": None,},
            "S6": {"A": 0.024, "B": 0.009, "C": 0.047, "D": 0.022, "E": 0.003, "F": 0.051, "SMC": 0.400},
            "S7": {"A": 0.086, "B": 0.038, "C": 0.010, "D": 0.032, "E": 0.048, "F": 0.033, "SMC": None,},
            "S8": {"A": 0.019, "B": 0.003, "C": 0.016, "D": 0.022, "E": None,  "F": 0.205, "SMC": 0.327},
            "S9": {"A": 0.164, "B": 0.173, "C": 0.082, "D": 0.146, "E": None,  "F": 0.188, "SMC": 0.190},
            "S10": {"A": 0.014, "B": 0.018, "C": 0.014, "D": 0.008, "E": None,  "F": 0.021, "SMC": 0.074}}

    subjects = list(data.keys())
    groups = ["A", "B", "C", "D", "E", "F", "SMC"]

    all_values = [v for sub in data.values() for v in sub.values() if v is not None]
    x_max = max(all_values) * 1.1
    placeholder_width = x_max * 0.1

    fig, axes = plt.subplots(2, 5, figsize=(16, 5))
    axes = axes.flatten()

    for ax, subj in zip(axes, subjects):
        subj_data = data[subj]
        y_pos = np.arange(len(groups))

        for i, g in enumerate(groups):
            val = subj_data.get(g)

            if val is None:
                ax.barh(
                    i,
                    placeholder_width,
                    color='white',
                    linewidth=1
                )
                ax.text(
                    placeholder_width / 2,
                    i,
                    "NA",
                    va='center',
                    ha='center',
                    color='gray',
                    fontsize=10
                )
            else:
                color = 'red' if g == "SMC" else 'gray'
                ax.barh(i, val, color=color)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(groups, ha='center')
        ax.invert_yaxis()
        ax.tick_params(axis='y', pad=10, labelsize=11)
        ax.set_xlim(0, x_max)
        ax.set_title(subj, fontsize=12)
        ax.tick_params(axis='x', labelsize=11)

        ax.axvline(0, color='black', linewidth=0.5)

    fig.text(0.5, 0.04, 'Speech-evoked envelope shift', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'SEEG electrode', va='center', rotation='vertical', fontsize=12)

    plt.tight_layout(rect=[0.05, 0.07, 1, 1])
    plt.savefig('Fig_channel_envdiff.png', bbox_inches='tight', dpi=300)
    plt.savefig('Fig_channel_envdiff.eps', bbox_inches='tight', dpi=300)
    plt.show()