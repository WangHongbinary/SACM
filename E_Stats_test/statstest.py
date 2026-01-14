'''
Author: WangHongbinary
E-mail: hbwang22@gmail.com
Date: 2026-01-08 15:15:02
LastEditTime: 2026-01-08 16:19:40
Description: 
'''

import numpy as np
from scipy.stats import ttest_rel

data1 = np.loadtxt('./test_data/baseline_DBConformer.csv')
data2 = np.loadtxt('./test_data/SACM_DBConformer.csv')

stat, p = ttest_rel(data1, data2)
print(data1.shape)
print(data2.shape)
print('before control stat={:f}, p={:.128f}'.format(stat, p))

print('stat={:f}, p={:.128f}'.format(stat, p))
if p > 0.05:
    print('!!! Probably the same distribution')
else:
    print('Probably different distributions')
