from nilearn import plotting
from nilearn.image import index_img
import nibabel as nib
import numpy as np
import cmocean

def plot_phase_volume(stc_phase, src, subject, subjects_dir,
                      initial_time=0.0, bg_img=None):

    t_idx = stc_phase.time_as_index(initial_time)[0]
    stc_t = stc_phase.copy().crop(
        tmin=stc_phase.times[t_idx],
        tmax=stc_phase.times[t_idx]
    )

    # Convert to NIfTI
    img      = stc_t.as_volume(src, mri_resolution=True)
    phase_3d = index_img(img, 0)

    # Build binary source space mask
    mask_stc         = stc_t.copy()
    mask_stc.data[:] = 1.0
    mask_img = index_img(mask_stc.as_volume(src, mri_resolution=True), 0)
    mask_bin = nib.Nifti1Image(
        (mask_img.get_fdata() > 0).astype(np.float32),
        mask_img.affine
    )

    if bg_img is None:
        import os
        bg_img = os.path.join(subjects_dir, subject, 'mri', 'T1.mgz')

    display = plotting.plot_stat_map(
        phase_3d,
        bg_img=bg_img,
        display_mode='mosaic',
        cut_coords=7,
        colorbar=True,
        cmap=cmocean.cm.phase,
        vmin=-np.pi,
        vmax=np.pi,
        symmetric_cbar=False,
        threshold=None,
        black_bg=True,
        transparency=mask_bin,          # 0 = fully transparent, 1 = fully opaque
        transparency_range=[0.5, 1.0],  # anything below 0.5 is transparent
    )

    plotting.show()
    return display