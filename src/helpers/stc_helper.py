import numpy as np
from nilearn.plotting import find_xyz_cut_coords
from nilearn.image import index_img
import mne

def extract_voxel_coords_from_stc(stc,voxel,src):
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

    img = stc_single.as_volume(src=src)

    # choose time index
    img3d = index_img(img, 0)

    coords = find_xyz_cut_coords(img3d)

    return coords