import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    data = np.load('./Fig7_data.npz')
    power_A = data['power_A']
    power_B = data['power_B']
    power_C = data['power_C']
    power_D = data['power_D']
    power_SMC = data['power_SMC']
    Audio = data['Audio']
    t1 = data['t']

    fig, ax1 = plt.subplots(nrows=1, figsize=(20, 10))
    line_A = ax1.plot(t1, power_A, color='g', linestyle='-.', linewidth=3, label='A')
    line_B = ax1.plot(t1, power_B, color='b', linestyle=':', linewidth=3, label='B')
    line_C = ax1.plot(t1, power_C, color='m', linestyle='--', linewidth=3, label='C')
    line_D = ax1.plot(t1, power_D, color='black', linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=3, label='D')
    line_SMC = ax1.plot(t1, power_SMC, color='r', linewidth=3, label='SMC')

    ax1.set_ylim(0, 3)
    ax1.tick_params(axis='x', labelsize=30)
    ax1.tick_params(axis='y', labelsize=30)
    ax1.set_xlabel('Time (seconds)', fontsize=30)
    ax1.set_ylabel('SEEG Power', fontsize=30)

    ax2 = ax1.twinx()
    t2 = np.arange(int(1.6*4*16000)) / int(16000)
    line_Audio = ax2.plot(t2, Audio, color='#1D43A3', linewidth=3, label='Audio')
    ax2.set_ylim(0, 1.5)
    ax2.tick_params(axis='y', labelsize=30)
    ax2.set_ylabel('Audio Amplitude', fontsize=30)
    
    lines = line_A + line_B + line_C + line_D + line_SMC + line_Audio
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, fontsize=30, bbox_to_anchor=(0.5, 1.01), loc='lower center', ncol=6)
    
    plt.savefig('Fig7.eps', bbox_inches='tight', dpi=300)
    plt.savefig('Fig7.png', bbox_inches='tight', dpi=300)

    plt.show()