#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:32:53 2019

@author: sigbjornmne
"""


import os
import os.path as op
from pathlib import Path

import numpy as np
import mne
import nibabel as nib
from nilearn.plotting import plot_stat_map, plot_glass_brain
from nilearn import plotting
from nilearn.image import index_img
import matplotlib.pyplot as plt
import helpers.helper_functions as functions

ss = functions.settings_dict()
#%%
stc_version = 'v005'
    


img_list = []

group_stat_dir = op.join(ss['stc_dir'], 'group_stat')
t_img = nib.load(op.join(group_stat_dir, 't.nii'))
bool_img = nib.load(op.join(group_stat_dir, 'bool.nii'))
comb_img = nib.load(op.join(group_stat_dir, 'comb.nii'))
colin = nib.load(Path(ss['atlas_dir']) / 'colin27_t1_tal_lin.nii')
colin_mask = nib.load(Path(ss['atlas_dir']) / 'colin27_t1_tal_lin_mask_TRANS.nii')
#fsaverage_brain.mgz
figsize = (7,8)
cut_coords = (0,0)



#%%



#%%
import matplotlib.gridspec as gridspec

gs1 = gridspec.GridSpec(2, 4)
gs1.update(wspace=0.01,hspace=0.01) #

fh = plt.figure()
for event in range(ss['n_events']):
#for event in range(1):
    img = index_img(comb_img, event)
#    ax = fh.add_subplot(2, 4, event + 1)
    ax = fh.add_subplot(gs1[event])
    title = ss['event_name_list'][event]+ ' %'
    if event == 3:
        cb = True
    else:
        cb = False

    display = plot_stat_map(img, 
                  axes=ax,
#                  bg_img = colin, 
#                  bg_img = None,
                  black_bg = True,
                  alpha = 0.6, 
                  vmax = 30,
                  title = title,
                  threshold = None,
                  cut_coords = cut_coords,
                  display_mode='xz',
                  draw_cross = True,
                  annotate = False,
                  colorbar = cb)

fh.suptitle(stc_version)
plt.show()

#fh.patch.set_facecolor((0,0,0))

#%%

fh = plt.figure()
for event in range(ss['n_events']):
    img = index_img(t_img, event)
    ax = fh.add_subplot(2, 4, event + 1)
    title = ss['event_name_list'][event]
    plot_glass_brain(img, 
                  axes=ax,
                  alpha = .6, 
                  vmax = 30,
                  black_bg = True,
                  title = title,
                  cut_coords = cut_coords,
                  display_mode='xz',
                  draw_cross = True,
                  annotate = False,
                  colorbar = True)
    
fh.suptitle(stc_version)
plt.show()

#%%
fh = plt.figure(figsize=figsize)
for event in range(ss['n_events']):
    img = index_img(t_img, event)
    ax = fh.add_subplot(2, 4, event + 1)
    title = ss['event_name_list'][event]
    plot_stat_map(img, 
                  axes=ax,
                  bg_img = colin,
                  alpha = .5, 
                  vmax = 30,
                  black_bg = True,
                  title = title,
                  cut_coords = cut_coords,
                  display_mode='xz',
                  draw_cross = True,
                  annotate = False,
                  colorbar = True)
    
#fh.suptitle(stc_version)
fh.patch.set_facecolor((0,0,0))

plt.show()
