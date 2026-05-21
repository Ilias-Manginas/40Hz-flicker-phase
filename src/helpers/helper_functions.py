#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 12:40:43 2019

@author: sigbjornmne

"""
import yaml
import mne
import os.path as op
import helpers.helper_functions


def settings_dict(*yaml_fname):
    

    if not yaml_fname:
        yaml_fname = 'helpers/settings.yaml'

    
    with open(yaml_fname, 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    return settings



def load_raw(subject, preload = True):
    
    ss = settings_dict()
#    for key, val in ss.items():
#        exec(key+'=val')

    raw_fname1 = op.join(ss['raw_dir'], ss['raw_path1_list'][subject])
    raw_fname2 = op.join(ss['raw_dir'], ss['raw_path2_list'][subject])
    if subject in [2, 8]:  # subjcets with raw data split in two files
        raw1 = mne.io.read_raw_fif(raw_fname1, preload=preload)
        raw2 = mne.io.read_raw_fif(raw_fname2, preload=preload)
        raw = mne.concatenate_raws([raw1,raw2], on_mismatch='warn')
    else:
        raw = mne.io.read_raw_fif(raw_fname1, preload=preload)  # raw meg data
    
    raw.info['bads'] = ss['bads']
    raw.set_channel_types(mapping={'EMG001': 'eeg'})
    raw.set_channel_types(mapping={'EMG002': 'eeg'})
    raw.set_channel_types(mapping={'EOG003': 'eeg'})
    raw.set_channel_types(mapping={'EOG004': 'eeg'})
    raw.set_channel_types(mapping={'EMG005': 'eeg'})
    raw.set_channel_types(mapping={'MISC001': 'misc'})
    return raw


def load_raw_list(preload = True):
    
    ss = settings_dict()
    raw_list = []
    for subject in ss['subject_idx_list']:
        raw_fname1 = op.join(ss['raw_dir'], ss['raw_path1_list'][subject])
        raw_fname2 = op.join(ss['raw_dir'], ss['raw_path2_list'][subject])
        if subject in [2, 8]:  # subjcets with raw data split in two files
            raw1 = mne.io.read_raw_fif(raw_fname1, preload=preload)
            raw2 = mne.io.read_raw_fif(raw_fname2, preload=preload)
            raw = mne.concatenate_raws([raw1,raw2])
        else:
            raw = mne.io.read_raw_fif(raw_fname1, preload=preload)  # raw meg data
        
        raw.info['bads'] = ss['bads']
        raw.set_channel_types(mapping={'EMG001': 'eeg'})
        raw.set_channel_types(mapping={'EMG002': 'eeg'})
        raw.set_channel_types(mapping={'EOG003': 'eeg'})
        raw.set_channel_types(mapping={'EOG004': 'eeg'})
        raw.set_channel_types(mapping={'EMG005': 'eeg'})
        raw.set_channel_types(mapping={'MISC001': 'eeg'})
        raw_list.append(raw)
    return raw_list


#subject here is the subject id not the subject index
def load_raw_filtered(subject, preload=True):
    ss = settings_dict()

    raw_fname = op.join(ss['raw_filtered_dir'], subject + "-raw.fif")
    raw = mne.io.read_raw_fif(raw_fname, preload=preload)  # raw meg data

    # raw.info['bads'] = ss['bads']
    # raw.set_channel_types(mapping={'EMG001': 'eeg'})
    # raw.set_channel_types(mapping={'EMG002': 'eeg'})
    # raw.set_channel_types(mapping={'EOG003': 'eeg'})
    # raw.set_channel_types(mapping={'EOG004': 'eeg'})
    # raw.set_channel_types(mapping={'EMG005': 'eeg'})
    # raw.set_channel_types(mapping={'MISC001': 'misc'})
    return raw


def load_raw_subset_list():
    
    ss = settings_dict()
    raw_list = []
    for subject in ss['subject_idx_list']:
        raw = mne.io.read_raw_fif(op.join(ss['raw_subset_dir'], ss['subject_id_list'][subject]+'-raw.fif'), preload=True)
        raw_list.append(raw)
        
    return raw_list


def load_events_list():
    ss = settings_dict()
    events_list = []
    
    for sub_idx in ss['subject_idx_list']:
        events = mne.read_events(op.join(ss['events_dir'], ss['subject_id_list'][sub_idx]+'-eve.fif'))    
        events_list.append(events)
    
    return events_list
    
    
def colour_scheme():

    """
    Produce colour pallet
    ---
    input: 

    output: 
        List of colour tuples
    ---
    """

    lux = .8
    lux = 1.0
    lux = .7

    col = [(0, 0, 1), (0, .5, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, .5, 0),
           (1, 0, 0)]
    
    col = [(0,.5,1),
           (0,0,1),
           (.5,.5,1),
           (0,0,0),
           (1,.5,.5),
           (1,0,0),
           (1,.5,0)]
    
    
    col = [(0,.0,1),
           (0,.5,1),
           (0,1,1),
           (0,1,0),
           (1,1,0),
           (1,.5,0),
           (1,0,0)]
    for jj in range(len(col)):
        col[jj] = tuple(lux*kk for kk in col[jj])
    return col



def modules():
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


#%% DISCARD PILE

#
#def subject_info():
#    
#    subject_nrs = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
#                   's11', 's12']
#    
#    subject_ids = ['0005_3SJ', '0002_TCZ', '0009_YGZ', '0010_ZMG', '0011_MEE',
#                   '0012_C3Z', '0014_TAG', '0015_QKW', '0016_XLZ', '0017_QJ5',
#                   '0018_5T3', '0019_COG']
#    
#    raw_paths1 = ['0005/20180514_000000/MEG/001.flicker/files/flicker.fif', 
#                  '0002/20180522_000000/MEG/001.s2/files/s2.fif', 
#                  '0009/20180522_000000/MEG/001.s3p1/files/s3p1.fif', 
#                  '0010/20180523_000000/MEG/001.s4/files/s4.fif', 
#                  '0011/20180524_000000/MEG/001.s5/files/s5.fif', 
#                  '0012/20180524_000000/MEG/001.s6/files/s6.fif', 
#                  '0014/20180525_000000/MEG/001.s7/files/s7.fif', 
#                  '0015/20180525_000000/MEG/001.s8/files/s8.fif', 
#                  '0016/20180525_000000/MEG/001.s9p1/files/s9p1.fif', 
#                  '0017/20180530_000000/MEG/001.s10/files/s10.fif',
#                  '0018/20180530_000000/MEG/001.s11/files/s11.fif', 
#                  '0019/20180530_000000/MEG/001.s12/files/s12.fif']
#    
#    raw_paths2 = ['', 
#                  '', 
#                  '0009/20180522_000000/MEG/002.s3p2/files/s3p2.fif', 
#                  '', 
#                  '', 
#                  '', 
#                  '', 
#                  '', 
#                  '0016/20180525_000000/MEG/002.s9p2/files/s9p2.fif', 
#                  '', 
#                  '', 
#                  '']
#    
#    return subject_nrs, subject_ids, raw_paths1, raw_paths2