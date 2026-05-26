#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:51:22 2019

@author: sigbjornmne
"""
import numpy as np
import mne
import os
import os.path as op
import nibabel as nib
import matplotlib.pyplot as plt
import nilearn
from nilearn.plotting import plot_stat_map, plot_glass_brain
from nilearn.image import index_img
import scipy.stats
import helpers.helper_functions as functions

ss = functions.settings_dict()
for key, val in ss.items():
    exec(key+'=val')


#%%
#stc_version = 'v003'
stc_version_list = ['v001', 'v002', 'v003', 'v004', 'v005', 
                    'v006', 'v007', 'v008', 'v009', 'v010', 'v011','v012', 'v013', 'v014', 'v015']
#plt.close('all')
#plt.figure()

group_stat_dir = op.join(ss['stc_dir'], 'group_stat')
act_frac = np.load(op.join(group_stat_dir, 'act_frac.npy'))
act_frac_p = [100*act_frac[ii]/act_frac[3] for ii in range(len(act_frac))]

x = [float(event_name) for event_name in ss['event_name_list']]
x = range(7)
fig, ax = plt.subplots(figsize=(4, 4))

plt.plot(x, act_frac*100, '-o', c='k', lw = 2)
#    plt.bar(x, act_frac*100, color = 'k',width=.7)
#    plt.bar(x, act_frac)

ax.set_xlabel('Duty cycle (%)')
ax.set_xticklabels(ss['event_name_list'], rotation=45)
ax.set_xticks(x)
ax.set_ylabel('Brain response volume (% of brain)')
#    ax.set_ylim([0, max(act_frac)*120])
ax.set_title('stc_version')
#    ax.set_xlim(0, 100)
plt.tight_layout()
