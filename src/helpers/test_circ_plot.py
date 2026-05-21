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
from concurrent.futures import ThreadPoolExecutor, as_completed

import helpers.helper_functions as hf


def plot_circ_stat_map(stc,
                       subject,
                       subjects_dir,
                       time=0.0,
                       cut_coords=(-26, -49, -6),
                       title=""):
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

    title = f"{subject}"
    plot_stat_map(img,
                  bg_img=t1,
                  alpha=.6,
                  title=title,
                  cut_coords=cut_coords,
                  cmap=cmocean.cm.phase,
                  vmin=-np.pi,
                  vmax=np.pi,
                  draw_cross=False)
    # fh.suptitle(suptitle)
    fh.patch.set_facecolor((0, 0, 0))


def plot_circ_stat_map_video(stc,
                              subject,
                              subjects_dir,
                              output_path="",
                              cut_coords=(-26, -49, -6),
                              fps=10,
                              time_indices=None,
                              dpi=100,
                              n_workers=10):
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
        Number of parallel threads for rendering. Default 10 (leaves 2 cores free
        on a 12-core CPU).
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

    # --- Render frames in parallel ---
    print(f"Rendering {len(time_indices)} frames with {n_workers} threads...")
    frames_unordered = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_render_frame, t_idx, stc, src, t1, subject, cut_coords, dpi): t_idx
            for t_idx in time_indices
        }
        for future in tqdm(as_completed(futures), total=len(time_indices)):
            frames_unordered.append(future.result())

    # Restore time order
    frames = [buf for _, buf in sorted(frames_unordered, key=lambda x: x[0])]

    # --- Assemble video ---
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
    ani.save(output_path, writer=writer)
    plt.close(fig_vid)
    print(f"Video saved to: {output_path}")

def _render_frame(t_idx, stc, src, t1, subject, cut_coords, dpi):
    t = stc.times[t_idx]
    stc_t = stc.copy().crop(tmin=t, tmax=t)
    stc_img = stc_t.as_volume(src)
    img = index_img(stc_img, 0)

    fig, ax = plt.subplots(figsize=(8, 3), dpi=dpi)
    plot_stat_map(img, bg_img=t1, alpha=0.6,
                  title=f"{subject}  |  t = {t * 1000:.1f} ms",
                  cut_coords=cut_coords, cmap=cmocean.cm.phase,
                  vmin=-np.pi, vmax=np.pi, draw_cross=False,
                  axes=ax, colorbar=True)
    fig.patch.set_facecolor((0, 0, 0))
    fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0)
    fig.canvas.draw()

    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()
    plt.close(fig)
    return t_idx, buf