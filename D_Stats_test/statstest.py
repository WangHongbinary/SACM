import numpy as np
from scipy.stats import ttest_rel

data1 = np.loadtxt('./test_data/word_BP_random.csv')
data2 = np.loadtxt('./test_data/word_BP_full.csv')

stat, p = ttest_rel(data1, data2)
print(data1.shape)
print(data2.shape)
print('before control stat={:f}, p={:.128f}'.format(stat, p))

print('stat={:f}, p={:.128f}'.format(stat, p))
if p > 0.05:
    print('!!! Probably the same distribution')
else:
    print('Probably different distributions')
