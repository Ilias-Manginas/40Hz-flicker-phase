import numpy as np
from nilearn.plotting import find_xyz_cut_coords, plot_stat_map
from nilearn.image import index_img
import os.path as op
import matplotlib.pyplot as plt
import mne

def extract_voxel_coords_from_stc(stc,voxel,src):
    # isolate voxel
    stc_single = get_stc_single_voxel(stc,voxel)

    img = stc_single.as_volume(src=src)

    # choose time index
    img3d = index_img(img, 0)

    coords = find_xyz_cut_coords(img3d)

    return coords

def get_stc_single_voxel(stc,voxel):
    # isolate voxel
    data = np.zeros_like(stc.data)
    data[voxel, :] = stc.data[voxel, :]

    stc_single = mne.VolSourceEstimate(
        data,
        vertices=stc.vertices,
        tmin=stc.tmin,
        tstep=stc.tstep,
        subject=stc.subject,
    )

    return stc_single

def get_masked_stc(stc, mask):
    masked_data = stc.copy().data

    masked_data[mask, :] = 0

    masked_stc = stc.copy()
    masked_stc.data = masked_data

def plot_different_stcs(stcs,
                        srcs,
                        t1s,
                        title,
                        fig_titles,
                        nrows,
                        ncols,
                        figsize,
                        display_mode="ortho",
                        draw_cross=False,
                        cut_coords=(-26, -49, -6),
                        dpi=100,
                        vmin = 0,
                        vmax = 1):



    img_list = []
    for stc, src in zip(stcs, srcs):

        img = index_img(stc.as_volume(src), 0)

        img_list.append(img)

    fig = plt.figure(figsize=figsize)

    fig.suptitle(f"{title}",
                 fontsize=12,
                 color="white"
                 )

    for img, fig_title, t1, i in zip(img_list, fig_titles, t1s, range(len(img_list))):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        plot_stat_map(
            img,
            bg_img=t1,
            display_mode=display_mode,
            cut_coords=cut_coords,
            vmin=-vmin,
            vmax=vmax,
            draw_cross=draw_cross,
            axes=ax,
            colorbar=(i+1 == ncols),
        )

        ax.set_title(fig_title, fontsize=10, color="white", loc="left")

    fig.patch.set_facecolor((0, 0, 0))
    fig.subplots_adjust(top=0.90)
    plt.show()

def plot_same_stc(stcs,
                  src,
                  t1,
                  title,
                  fig_titles,
                  nrows,
                  ncols,
                  figsize,
                  display_mode="ortho",
                  draw_cross=False,
                  cut_coords=(-26, -49, -6),
                  dpi=100,
                  vmin = 0,
                  vmax = 1):



    img_list = []
    for stc in stcs:

        img = index_img(stc.as_volume(src), 0)

        img_list.append(img)

    fig = plt.figure(figsize=figsize)

    fig.suptitle(f"{title}",
                 fontsize=12,
                 color="white"
                 )

    for img, fig_title, i in zip(img_list, fig_titles, range(len(img_list))):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        plot_stat_map(
            img,
            bg_img=t1,
            display_mode=display_mode,
            cut_coords=cut_coords,
            vmin=-vmin,
            vmax=vmax,
            draw_cross=draw_cross,
            axes=ax,
            colorbar=(i+1 == ncols),
        )

        ax.set_title(fig_title, fontsize=10, color="white", loc="left")

    fig.patch.set_facecolor((0, 0, 0))
    fig.subplots_adjust(top=0.90)
    plt.show()