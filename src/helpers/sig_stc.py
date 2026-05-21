#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:25:17 2019

@author: sigbjornmne
"""

# its the make_stc script almost unchanged
# except for the fact that the epochs are averaged before making the stc

import os
import os.path as op
import numpy as np
import mne
import nibabel as nib
from mne.export import export_evokeds
from nilearn.plotting import plot_stat_map, plot_glass_brain
from nilearn.image import index_img
from mne.beamformer import (make_lcmv, apply_lcmv_epochs)
from scipy.stats import ranksums
import helpers.helper_functions as functions
import pickle
import warnings

def make_sig_stc(subject_index, event_id):

    ss = functions.settings_dict()

    # %% SETTINGS
    stc_version = 'v012'
    fs = 2000
    ch_type = 'grad'
    fmin, fmax = 38, 42
    tbw = 1  # with of transision band for bp filter
    tspan_base = 0.3  # baseline duration
    tspan_stim = 3.0
    offset_base = -0.0  # avoid onset related data
    offset_stim = 1.0  # avoid onset related data
    tmin = -tspan_base + offset_base
    tmax = tspan_stim + offset_stim - 1 / fs


    subject_id = ss['subject_id_list'][subject_index]
    print('Subject: ' + ss['subject_nr_list'][subject_index] + '/' + subject_id + ' is selected')
    fwd_fname = op.join(ss['fwd_dir'], subject_id + '-fwd.fif')
    events_fname = op.join(ss['events_dir'], subject_id + '-eve.fif')

    # load fwd
    fwd = mne.read_forward_solution(fwd_fname)
    if fwd['src'][0]['subject_his_id'].startswith("fs"):
        fwd['src'][0]['subject_his_id'] = subject_id

    raw = functions.load_raw(subject_index)
    raw.info['bads'] = ss['bads']
    raw.pick_types(meg=ch_type)

    # filter
    raw.filter(fmin,
               fmax,
               n_jobs=1,
               l_trans_bandwidth=tbw,
               h_trans_bandwidth=tbw,
               fir_design='firwin')

    # load events
    events = mne.read_events(events_fname)

    # covariance of all epochs
    epochs_all = mne.Epochs(raw,
                            events,
                            ss['event_id_list'],
                            tmin=tmin,
                            tmax=tmax,
                            baseline=None,
                            preload=True)
    epochs_all._raw = None

    # time stamps in epochs (shifted by tmin relative to raw/event time)
    tmin_base = 0
    tmax_base = tmin_base + tspan_base
    tmin_stim = -tmin + offset_stim
    tmax_stim = -tmin + tmax

    cov_base = mne.compute_covariance(epochs_all, tmin=tmin_base, tmax=tmax_base)
    cov_stim = mne.compute_covariance(epochs_all, tmin=tmin_stim, tmax=tmax_stim)
    cov_all = cov_stim + cov_base
    #    cov_all = mne.compute_covariance(epochs_all)

    # BEAMFORMER/SPATIAL FILTER
    filters = make_lcmv(epochs_all.info,
                        fwd,
                        data_cov=cov_all,
                        pick_ori='max-power',
                        weight_norm='nai',
                        reg=0.05)

    # apply spatial filter
    epochs = mne.Epochs(raw,
                        events,
                        event_id,
                        tmin=tmin,
                        tmax=tmax,
                        baseline=None,
                        preload=True)

    epochs._raw = None  # care for memory
    # stc = apply_lcmv_epochs(epochs, filters)

    evoked = epochs.average()
    stc = mne.beamformer.apply_lcmv(evoked,filters)

    return stc