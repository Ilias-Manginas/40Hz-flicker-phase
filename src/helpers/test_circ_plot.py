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


def plot_circ_stat_map(stcs,
                       src,
                      subject,
                      subjects_dir,
                      title,
                      fig_titles,
                      nrows,
                      ncols,
                      figsize,
                      display_mode="ortho",
                      draw_cross=False,
                      cut_coords=(-26, -49, -6),
                      dpi=100):

    t1 = op.join(subjects_dir, subject, 'mri', 'T1.mgz')

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
        plot = plot_stat_map(
            img,
            bg_img=t1,
            display_mode=display_mode,
            cut_coords=cut_coords,
            cmap=cmocean.cm.phase,
            vmin=-np.pi,
            vmax=np.pi,
            draw_cross=draw_cross,
            axes=ax,
            colorbar=(i+1 == ncols),
        )

        ax.set_title(fig_title, fontsize=10, color="white", loc="left")

    fig.patch.set_facecolor((0, 0, 0))
    fig.subplots_adjust(top=0.9)

    plt.show()



def _render_multi_imgs(args):
    """Render two pre-computed volumes stacked vertically, return (index, RGB array).
    Must be a module-level function for ProcessPoolExecutor to pickle it.
    """
    t_idx, times, img_list, t1, subject, cut_coords, dpi, draw_cross, title, nrows, ncols, figsize = args
    #figsize=(8, 6)
    fig = plt.figure(figsize=figsize)

    fig.suptitle(f"{title}",
                 fontsize=12,
                 color="white"
                 )

    for img, i in zip(img_list, range(len(img_list))):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        plot_stat_map(
            img,
            bg_img=t1,
            cut_coords=cut_coords,
            cmap=cmocean.cm.phase,
            vmin=-np.pi,
            vmax=np.pi,
            draw_cross=draw_cross,
            axes=ax,
            colorbar=(i+1 == ncols),
        )

    fig.patch.set_facecolor((0, 0, 0))
    fig.subplots_adjust(top=0.9)
    fig.canvas.draw()

    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()
    plt.close(fig)
    return t_idx, buf

def plot_multi_circ_stat_map_video(stcs,
                                   src,
                                  subject,
                                  subjects_dir,
                                  title,
                                  output_path,
                                  nrows,
                                  ncols,
                                  figsize,
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
    assert all(len(stc.times) == len(stcs[0].times) for stc in stcs), \
        "All STCs must have the same number of time points"

    t1 = op.join(subjects_dir, subject, 'mri', 'T1.mgz')

    if time_indices is None:
        time_indices = np.arange(len(stcs[0].times))

    # --- Step 1: Pre-compute all volumes sequentially ---
    print(f"Pre-computing {len(time_indices)} volumes...")
    imgs = []
    for t_idx in tqdm(time_indices):
        times = stcs[0].times[t_idx]

        img_list = []
        for stc in stcs:

            # Slice the data directly instead of copying the whole stc
            stc_cropped = stc.copy()  # still needed but crop immediately
            stc_cropped._data = stc.data[:, t_idx:t_idx + 1]  # only keep one time point
            stc_cropped.tmin = times
            stc_cropped.tstep = stc.tstep

            img = index_img(stc_cropped.as_volume(src), 0)

            img_list.append(img)
            del stc_cropped  # explicitly free after volume conversion
            gc.collect()

        imgs.append((t_idx, times, img_list, t1, subject, cut_coords, dpi, draw_cross, title, nrows, ncols, figsize))

    # --- Step 2: Render frames in parallel ---
    print(f"Rendering frames with {n_workers} processes...")
    frames_unordered = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for result in tqdm(executor.map(_render_multi_imgs, imgs), total=len(imgs)):
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