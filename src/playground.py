import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import helpers.helper_functions as hf


def plot_meg_multiview(raw, show_names=False, sensor_color='steelblue'):
    """
    Plot MEG sensors as 2D topomaps from 3 angles (top, front, side)
    with a fitted head circle outline.
    """
    info = raw.info
    picks = mne.pick_types(info, meg=True)
    ch_pos = np.array([info['chs'][i]['loc'][:3] for i in picks])
    ch_names = [info['ch_names'][i] for i in picks]

    x, y, z = ch_pos[:, 0], ch_pos[:, 1], ch_pos[:, 2]

    # Fit a sphere radius for each projection plane
    def fit_radius(a, b):
        return np.max(np.sqrt(a**2 + b**2)) * 1.1  # 10% padding

    views = [
        {
            'title': 'Top view\n(looking down Z)',
            'h': x, 'v': y,
            'xlabel': 'X (left ← → right)',
            'ylabel': 'Y (back ← → front)',
            'radius': fit_radius(x, y),
            'nose_dir': (0, 1),   # nose points in +Y
        },
        {
            'title': 'Front view\n(looking along Y)',
            'h': x, 'v': z,
            'xlabel': 'X (left ← → right)',
            'ylabel': 'Z (down ← → up)',
            'radius': fit_radius(x, z),
            'nose_dir': None,
        },
        {
            'title': 'Side view\n(looking along X)',
            'h': y, 'v': z,
            'xlabel': 'Y (back ← → front)',
            'ylabel': 'Z (down ← → up)',
            'radius': fit_radius(y, z),
            'nose_dir': (1, 0),   # nose points in +Y direction
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.patch.set_facecolor('#f8f8f8')

    for ax, view in zip(axes, views):
        h, v = view['h'], view['v']
        r = view['radius']

        ax.set_facecolor('#f0f4f8')

        # --- Head circle ---
        head_circle = Circle((0, 0), r, fill=False,
                              color='#333333', linewidth=2.5, zorder=2)
        ax.add_patch(head_circle)

        # --- Nose triangle (top and side views only) ---
        if view['nose_dir'] is not None:
            nx, ny = view['nose_dir']
            nose = plt.Polygon(
                [(nx * r * 0.98,       ny * r * 0.98),
                 (nx * r * 1.12 ,      ny * r * 1.12),
                 (-nx * r * 0.05 + ny * r * 1.05, -ny * r * 0.05 + nx * r * 1.05)],
                closed=False, fill=False, color='#333333', linewidth=2.5, zorder=2
            )

            # Simpler nose: just a small triangle tip
            tip_x = nx * (r + r * 0.15)
            tip_y = ny * (r + r * 0.15)
            ear_offset = 0.12 * r
            nose_pts = np.array([
                [-ny * ear_offset + nx * r * 0.9,  nx * ear_offset + ny * r * 0.9],
                [nx * (r + r * 0.18),               ny * (r + r * 0.18)],
                [ ny * ear_offset + nx * r * 0.9, -nx * ear_offset + ny * r * 0.9],
            ])
            ax.plot(nose_pts[:, 0], nose_pts[:, 1],
                    color='#333333', linewidth=2.5, zorder=2)

        # --- Ear bumps (top and side views) ---
        if view['nose_dir'] is not None:
            nx, ny = view['nose_dir']
            # ears are perpendicular to nose direction
            for side in [-1, 1]:
                ear_x = side * ny * r  # perpendicular
                ear_y = side * nx * r  # perpendicular (signed)
                # actually ears sit at 90deg from nose
                ex = -ny * side * r
                ey =  nx * side * r
                ear = Circle((ex * 1.02, ey * 1.02), r * 0.09,
                             fill=False, color='#333333',
                             linewidth=2.5, zorder=2)
                ax.add_patch(ear)

        # --- Sensors ---
        ax.scatter(h, v, s=40, color=sensor_color,
                   edgecolors='white', linewidths=0.5,
                   zorder=4, alpha=0.85)

        # --- Channel names ---
        if show_names:
            for hi, vi, name in zip(h, v, ch_names):
                ax.text(hi, vi, name, fontsize=4.5,
                        ha='center', va='bottom', color='#222222',
                        zorder=5)

        # --- Crosshair at origin ---
        ax.axhline(0, color='gray', linewidth=0.6, alpha=0.4, zorder=1)
        ax.axvline(0, color='gray', linewidth=0.6, alpha=0.4, zorder=1)

        # --- Styling ---
        ax.set_xlim(-r * 1.35, r * 1.35)
        ax.set_ylim(-r * 1.35, r * 1.35)
        ax.set_aspect('equal')
        ax.set_title(view['title'], fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel(view['xlabel'], fontsize=10)
        ax.set_ylabel(view['ylabel'], fontsize=10)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.tick_params(labelsize=8)

        # Convert axis ticks to mm for readability
        ticks = ax.get_xticks()
        ax.set_xticklabels([f'{t*1000:.0f}' for t in ticks])
        ticks = ax.get_yticks()
        ax.set_yticklabels([f'{t*1000:.0f}' for t in ticks])
        ax.set_xlabel(view['xlabel'] + ' [mm]', fontsize=10)
        ax.set_ylabel(view['ylabel'] + ' [mm]', fontsize=10)

    plt.suptitle('MEG Sensor Layout — Multi-angle View',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()
    return fig


# --- Run it ---
raw = hf.load_raw(subject=1, preload=True)

fig = plot_meg_multiview(raw, show_names=True)