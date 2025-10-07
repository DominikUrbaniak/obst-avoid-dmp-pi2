import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from dmp import dmp_cartesian as dmp
from pi2 import PI2

# ----------------------------------------------------------------------
# Experiment configuration
# ----------------------------------------------------------------------
mode = "box_acc"
name = "random_configs"

n_dmps = 3
n_bfs = 10
K = 25
basis = "rbf"
visualize_progress = 1

pi2 = PI2()
optn = 1
trajectory_file = "poly_traj_3.txt"
pi2.sigma_min = 0.0007
pi2.sigma_max = 0.13


pi2.sigma_steepness = 2
pi2.plot_dmp_init = False
pi2.plot_frequency = 10

# ----------------------------------------------------------------------
# DMP initialization
# ----------------------------------------------------------------------
f_target = pi2.initialize_dmp(trajectory_file, n_dmps, n_bfs, K=K, basis=basis)
init_means = pi2.dmp.w
pi2.dmp.rescale = "rotodilatation"

# Runtime configuration
max_runs = 10
# Get training directory from command line
if len(sys.argv) > 1:
    dirs = f"results/{sys.argv[1]}"
    if len(sys.argv) > 2:
        max_runs = sys.argv[2]
        if len(sys.argv) > 3:
            visualize_process = sys.argv[3]
else:
    dirs = "results/default/"
print(f"max_runs: {max_runs}, dirs: {dirs}")

max_val = 0.15
max_iters = 15000
n_samples = 10
covar_decay_factor = 1
saving = 0

pi2.param_pertub = [0, 1, 1]  # perturb forcing terms in y- and z-directions
pi2.bounds = [0.003, 0.003]

# ----------------------------------------------------------------------
# Experiment loop
# ----------------------------------------------------------------------
time_0 = time.time()
run_time = 0
run_id = 0

while run_id < max_runs:
    print(f"Run #{run_id}/{max_runs}, elapsed minutes: {run_time}")

    z_heights = np.sort(0.005+0.14*np.random.rand(2))
    print(f"z_heights: {z_heights}, max_val: {max_val}")

    # Prepare directories

    directory = f"{dirs}train{run_id}"
    img_folder = os.path.join(dirs, "graphics/img_train/")
    pdf_folder = os.path.join(dirs, "graphics/pdf_train/")
    os.makedirs(directory, exist_ok=True)
    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(pdf_folder, exist_ok=True)

    # Figures
    fig1, ax1 = plt.subplots(figsize=(8, 7 * max_val / 0.15))
    try:
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.wm_geometry("+0+0")
    except Exception:
        pass

    fig2, (ax6, ax7) = plt.subplots(1, 2, figsize=(10, 7 * max_val / 0.15), facecolor="w")
    try:
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.wm_geometry("+1100+0")
    except Exception:
        pass
    axes = [ax1, ax6, ax7]

    # Reset weights
    pi2.dmp.w = init_means

    # Run optimization
    start_time = time.time()
    costs, h, training_data, labels, n_episodes, duration, y_locations, final_x_track = (
        pi2.optimize_continuous_box_acc(
            max_iters,
            n_samples,
            z_heights,
            covar_decay_factor,
            saving,
            directory,
            max_val,
            mode,
            axes,
            visualize_progress=visualize_progress,
        )
    )

    # Trim unused data
    training_data = training_data[:n_episodes, :, :]
    labels = labels[:n_episodes, :]
    y_locations = y_locations[:n_episodes, :]

    # Save data
    np.savez(
        f"{directory}/train_{run_id}.npz",
        training_data=training_data,
        labels=labels,
        n_episodes=n_episodes,
        costs=costs,
        t_pi2=duration,
        z_heights=z_heights,
        max_val=max_val,
        y_locations=y_locations,
        final_x_traj=final_x_track,
    )

    # Save figures
    fig1.savefig(f"{pdf_folder}fig_{run_id}.pdf", format="pdf")
    fig1.savefig(f"{img_folder}img_{run_id}.png", format="png")
    plt.close(fig1)

    fig2.savefig(f"{pdf_folder}param_fig_{run_id}.pdf", format="pdf")
    fig2.savefig(f"{img_folder}param_img_{run_id}.png", format="png")
    plt.close(fig2)

    # Update run time
    run_time = int((time.time() - time_0) / 60)
    run_id += 1
