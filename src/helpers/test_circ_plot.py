import gc
import mne
import numpy as np
import matplotlib.pyplot as plt
import os.path as op
from nilearn.image import index_img
from nilearn.plotting import plot_stat_map
import cmocean
import helpers.helper_functions as hf
import matplotlib.animation as animation
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import helpers.helper_functions as hf


def plot_circ_stat_map(stc,
                       subject,
                       subjects_dir,
                       time=0.0,
                       cut_coords=(-26, -49, -6),
                       title="",
                       draw_cross=False,):
    ss = hf.settings_dict()
    fwd_fname = op.join(ss['fwd_dir'], subject + '-fwd.fif')
    fwd = mne.read_forward_solution(fwd_fname)
    src = fwd['src']

    if src[0]['subject_his_id'].startswith("fs"):
        src[0]['subject_his_id'] = subject

    t_idx = stc.time_as_index(time)[0]
    stc_t = stc.copy().crop(
        tmin=stc.times[t_idx],
        tmax=stc.times[t_idx]
    )

    stc_img = stc_t.as_volume(fwd['src'])
    t1 = op.join(subjects_dir, subject, 'mri', 'T1.mgz')

    fh = plt.figure(figsize=(6, 8))

    img = index_img(stc_img, 0)
    #    img = index_img(stc_z_img, ii)

    plot_stat_map(img,
                  bg_img=t1,
                  title=title,
                  cut_coords=cut_coords,
                  cmap=cmocean.cm.phase,
                  vmin=-np.pi,
                  vmax=np.pi,
                  draw_cross=draw_cross)
    # fh.suptitle(suptitle)
    fh.patch.set_facecolor((0, 0, 0))

    plt.show()

def _render_img(args):
    """Render a single pre-computed volume and return (index, RGB array).
    Must be a module-level function for ProcessPoolExecutor to pickle it.
    """
    t_idx, t, img, t1, subject, cut_coords, dpi, draw_cross, title = args

    fig, ax = plt.subplots(figsize=(8, 3), dpi=dpi)
    plot_stat_map(img,
                  bg_img=t1,
                  title=f"{title}  |  t = {t * 1000:.1f} ms",
                  cut_coords=cut_coords,
                  cmap=cmocean.cm.phase,
                  vmin=-np.pi,
                  vmax=np.pi,
                  draw_cross=draw_cross,
                  axes=ax,
                  colorbar=True)
    fig.patch.set_facecolor((0, 0, 0))
    fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0)
    fig.canvas.draw()

    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()
    plt.close(fig)
    return t_idx, buf

def plot_circ_stat_map_video(stc,
                              subject,
                              subjects_dir,
                              output_path,
                              title,
                              cut_coords=(-26, -49, -6),
                              draw_cross = False,
                              fps=10,
                              time_indices=None,
                              dpi=100,
                              n_workers=4):
    """
    Generate an MP4 video of cyclic phase stat maps across time.

    Parameters
    ----------
    stc : SourceEstimate
        The source estimate object.
    subject : str
        Subject ID.
    subjects_dir : str
        Path to the FreeSurfer subjects directory.
    output_path : str
        Path to save the output MP4 file.
    cut_coords : tuple
        MNI cut coordinates (x, y, z).
    fps : int
        Frames per second in the output video.
    time_indices : list or None
        List of time indices to include. If None, all time points are used.
        Use e.g. np.arange(0, len(stc.times), 5) to subsample every 5th frame.
    dpi : int
        Resolution of each frame (pixels = figsize * dpi).
    n_workers : int
        Number of parallel processes for rendering. Default 10 (leaves 2 cores
        free on a 12-core CPU).
    """
    ss = hf.settings_dict()
    fwd_fname = op.join(ss['fwd_dir'], subject + '-fwd.fif')
    fwd = mne.read_forward_solution(fwd_fname)
    src = fwd['src']

    if src[0]['subject_his_id'].startswith("fs"):
        src[0]['subject_his_id'] = subject

    t1 = op.join(subjects_dir, subject, 'mri', 'T1.mgz')

    if time_indices is None:
        time_indices = np.arange(len(stc.times))

    # --- Step 1: Pre-compute all volumes sequentially ---
    print(f"Pre-computing {len(time_indices)} volumes...")
    imgs = []
    for t_idx in tqdm(time_indices):
        t = stc.times[t_idx]
        stc_t = stc.copy()  # still needed but crop immediately
        stc_t._data = stc.data[:, t_idx:t_idx + 1]  # only keep one time point
        stc_t.tmin = t
        stc_t.tstep = stc.tstep
        img = index_img(stc_t.as_volume(src), 0)

        del stc_t   # explicitly free after volume conversion
        gc.collect()
        imgs.append((t_idx, t, img, t1, subject, cut_coords, dpi, draw_cross, title))

    # --- Step 2: Render frames in parallel ---
    print(f"Rendering frames with {n_workers} processes...")
    frames_unordered = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for result in tqdm(executor.map(_render_img, imgs), total=len(imgs)):
            frames_unordered.append(result)

    # Free volumes immediately after rendering
    del imgs
    gc.collect()

    # Restore time order
    frames = [buf for _, buf in sorted(frames_unordered, key=lambda x: x[0])]
    del frames_unordered
    gc.collect()

    # --- Step 3: Assemble video ---
    print("Assembling video...")
    h, w = frames[0].shape[:2]
    fig_vid, ax_vid = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig_vid.patch.set_facecolor("black")
    ax_vid.axis("off")
    fig_vid.subplots_adjust(0, 0, 1, 1)

    im = ax_vid.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    ani = animation.FuncAnimation(
        fig_vid, update, frames=frames, interval=1000 / fps, blit=True
    )

    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    ani.save(str(output_path), writer=writer)
    plt.close(fig_vid)

    # Free frames after video is saved
    del frames, ani
    gc.collect()

    print(f"Video saved to: {output_path}")

def _render_dual_imgs(args):
    """Render two pre-computed volumes stacked vertically, return (index, RGB array).
    Must be a module-level function for ProcessPoolExecutor to pickle it.
    """
    t_idx, times1, times2, img1, img2, t1, subject, cut_coords, dpi, draw_cross, title1, title2 = args

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), dpi=dpi)

    plot_stat_map(img1,
                  bg_img=t1,
                  title=f"{title1}  |  t = {times1 * 1000:.1f} ms",
                  cut_coords=cut_coords,
                  cmap=cmocean.cm.phase,
                  vmin=-np.pi,
                  vmax=np.pi,
                  draw_cross=draw_cross,
                  axes=ax1,
                  colorbar=True)

    plot_stat_map(img2,
                  bg_img=t1,
                  title=f"{title2}  |  t = {times2 * 1000:.1f} ms",
                  cut_coords=cut_coords,
                  cmap=cmocean.cm.phase,
                  vmin=-np.pi,
                  vmax=np.pi,
                  draw_cross=draw_cross,
                  axes=ax2,
                  colorbar=True)

    fig.patch.set_facecolor((0, 0, 0))
    fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0, hspace=0.1)
    fig.canvas.draw()

    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()
    plt.close(fig)
    return t_idx, buf

def plot_dual_circ_stat_map_video(stc1,
                              stc2,
                              subject,
                              subjects_dir,
                              title1,
                              title2,
                              output_path,
                              draw_cross=False,
                              cut_coords=(-26, -49, -6),
                              fps=10,
                              time_indices=None,
                              dpi=100,
                              n_workers=4):
    """
    Generate an MP4 video of two cyclic phase stat maps stacked vertically across time.

    Parameters
    ----------
    stc1 : SourceEstimate
        First source estimate object.
    stc2 : SourceEstimate
        Second source estimate object. Must have the same times as stc1.
    subject : str
        Subject ID.
    subjects_dir : str
        Path to the FreeSurfer subjects directory.
    output_path : str
        Path to save the output MP4 file.
    stc1_label : str
        Label shown in the title of the top plot.
    stc2_label : str
        Label shown in the title of the bottom plot.
    cut_coords : tuple
        MNI cut coordinates (x, y, z).
    fps : int
        Frames per second in the output video.
    time_indices : list or None
        List of time indices to include. If None, all time points are used.
        Use e.g. np.arange(0, len(stc1.times), 5) to subsample every 5th frame.
    dpi : int
        Resolution of each frame (pixels = figsize * dpi).
    n_workers : int
        Number of parallel processes for rendering. Default 10 (leaves 2 cores
        free on a 12-core CPU).
    """
    assert len(stc1.times) == len(stc2.times), \
        "stc1 and stc2 must have the same number of time points"

    ss = hf.settings_dict()
    fwd_fname = op.join(ss['fwd_dir'], subject + '-fwd.fif')
    fwd = mne.read_forward_solution(fwd_fname)
    src = fwd['src']

    if src[0]['subject_his_id'].startswith("fs"):
        src[0]['subject_his_id'] = subject

    t1 = op.join(subjects_dir, subject, 'mri', 'T1.mgz')

    if time_indices is None:
        time_indices = np.arange(len(stc1.times))

    # --- Step 1: Pre-compute all volumes sequentially ---
    print(f"Pre-computing {len(time_indices)} volumes...")
    imgs = []
    for t_idx in tqdm(time_indices):
        times1 = stc1.times[t_idx]
        times2 = stc2.times[t_idx]

        # Slice the data directly instead of copying the whole stc
        stc1_t = stc1.copy()  # still needed but crop immediately
        stc1_t._data = stc1.data[:, t_idx:t_idx + 1]  # only keep one time point
        stc1_t.tmin = times1
        stc1_t.tstep = stc1.tstep

        stc2_t = stc2.copy()
        stc2_t._data = stc2.data[:, t_idx:t_idx + 1]
        stc2_t.tmin = times2
        stc2_t.tstep = stc2.tstep

        img1 = index_img(stc1_t.as_volume(src), 0)
        img2 = index_img(stc2_t.as_volume(src), 0)

        del stc1_t, stc2_t  # explicitly free after volume conversion
        gc.collect()

        imgs.append((t_idx, times1, times2 , img1, img2, t1, subject, cut_coords, dpi, draw_cross, title1, title2))

    # --- Step 2: Render frames in parallel ---
    print(f"Rendering frames with {n_workers} processes...")
    frames_unordered = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for result in tqdm(executor.map(_render_dual_imgs, imgs), total=len(imgs)):
            frames_unordered.append(result)

    # Free volumes immediately after rendering
    del imgs
    gc.collect()

    # Restore time order
    frames = [buf for _, buf in sorted(frames_unordered, key=lambda x: x[0])]
    del frames_unordered
    gc.collect()

    # --- Step 3: Assemble video ---
    print("Assembling video...")
    h, w = frames[0].shape[:2]
    fig_vid, ax_vid = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig_vid.patch.set_facecolor("black")
    ax_vid.axis("off")
    fig_vid.subplots_adjust(0, 0, 1, 1)

    im = ax_vid.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    ani = animation.FuncAnimation(
        fig_vid, update, frames=frames, interval=1000 / fps, blit=True
    )

    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    ani.save(str(output_path), writer=writer)
    plt.close(fig_vid)

    # Free frames after video is saved
    del frames, ani
    gc.collect()

    print(f"Video saved to: {output_path}")