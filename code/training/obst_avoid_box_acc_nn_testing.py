#!/usr/bin/env python3
"""
Evaluate a trained neural network policy by generating DMP rollouts
for different obstacle configurations and visualizing the results.

Usage:
    python obst_avoid_box_acc_nn_testsing.py <results_directory> <ins_offset> <nn_model_filename>
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import logging
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
from pi2 import PI2


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def generate_3d_vector_list(length, max_x, min_y, max_z, seed=42):
    """Generate random 3D vectors [x, y, z] with y < z."""
    if length <= 0:
        raise ValueError("Length must be positive.")
    if min_y >= max_z:
        raise ValueError("min_y must be smaller than max_z.")
    if seed is not None:
        np.random.seed(seed)

    vectors = []
    while len(vectors) < length:
        batch_size = (length - len(vectors)) * 2
        x_vals = np.random.uniform(0, max_x, batch_size)
        y_vals = np.random.uniform(min_y, max_z, batch_size)
        z_vals = np.random.uniform(min_y, max_z, batch_size)
        for x, y, z in zip(x_vals, y_vals, z_vals):
            if y < z:
                vectors.append([x, y, z])
                if len(vectors) == length:
                    break
    return np.array(vectors)


def generate_3d_vector_list_extrapolation(length, max_x, max_x_extra, min_y, max_z, seed=42):
    """Generate 3D vectors in extrapolation region with y < z."""
    if length <= 0:
        raise ValueError("Length must be positive.")
    if min_y >= max_z:
        raise ValueError("min_y must be smaller than max_z.")
    if seed is not None:
        np.random.seed(seed)

    vectors = []
    while len(vectors) < length:
        batch_size = (length - len(vectors)) * 2
        x_vals = np.random.uniform(max_x, max_x_extra, batch_size)
        y_vals = np.random.uniform(min_y, max_z, batch_size)
        z_vals = np.random.uniform(min_y, max_z, batch_size)
        for x, y, z in zip(x_vals, y_vals, z_vals):
            if y < z:
                vectors.append([x, y, z])
                if len(vectors) == length:
                    break
    return np.array(vectors)


# ----------------------------------------------------------------------
# Command-line configuration
# ----------------------------------------------------------------------


if len(sys.argv) > 1:
    ins_offset = float(sys.argv[1])
else:
    ins_offset = 0.0

if len(sys.argv) > 2:
    dirs = f"results/{sys.argv[2]}"
else:
    dirs = "results/default/"  # Default directory

if not dirs.endswith("/"):
    dirs += "/"

if len(sys.argv) > 3:
    model_filename = sys.argv[3]
else:
    model_filename = "model.pth"

print(f"Using results directory: {dirs}")
if model_filename:
    print(f"Using model: {model_filename} and ins_offset: {ins_offset}")

# ----------------------------------------------------------------------
# Experiment configuration
# ----------------------------------------------------------------------
test = "learned"
data_size = 100
n_ins = 3

max_in1, min_in2, max_in3, max_in4 = 1.0, 0.03, 0.96, 0.99
n_samples = 16
add_offsets = True
offset_in1, offset_in2, offset_in3 = ins_offset, -ins_offset, ins_offset


max_in1_orig = max_in1
#max_in1 = 2.0
max_in3 = 0.97
n_tests_up, n_tests = 10, 10

ins_random, ins_rand_list = False, True
along_in1, along_in2, along_in3, along_in4 = True, False, False, False
ins_list = []

# create random list of nn inputs
id = f"list_{n_tests_up}_{n_tests}_off{ins_offset}"
if max_in1 > max_in1_orig:
    ins_list = generate_3d_vector_list_extrapolation(
        n_tests_up * n_tests, max_in1_orig, max_in1, min_in2, max_in3
    )
else:
    ins_list = generate_3d_vector_list(n_tests_up * n_tests, max_in1, min_in2, max_in3)

# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
model_path = os.path.join(dirs, model_filename)

print(f"Loading model: {model_path}")
loaded_model = torch.jit.load(model_path)
loaded_model.eval()

# ----------------------------------------------------------------------
# Initialize DMP system
# ----------------------------------------------------------------------
trajectory_file = "poly_traj_3.txt"
n_dmps, n_bfs, basis = 3, 10, "rbf"
max_val = 0.15
mode = "default"

pi2 = PI2()
pi2.sigma_min = 0.001
pi2.sigma_max = 0.05
pi2.sigma_steepness = 2
pi2.plot_dmp_init = False

f_target = pi2.initialize_dmp(trajectory_file, n_dmps, n_bfs, basis=basis)
init_means = pi2.dmp.w
pi2.dmp.rescale = "rotodilatation_xy" if n_ins == 3 else "rotodilatation"

L_demo = pi2.learned_L
max_f = max_val / L_demo
pi2.dmp.x_goal[1] = L_demo

# ----------------------------------------------------------------------
# Experiment setup
# ----------------------------------------------------------------------

test_data = np.linspace(0, max_val, n_tests_up)
all_costs = np.zeros([n_tests_up, 5])

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(message)s", filename="measurement_log.txt", filemode="w")

# ----------------------------------------------------------------------
# Output directories
# ----------------------------------------------------------------------
subfolders = f"tests/{id}/"
folders = {
    "img": f"{dirs}{subfolders}graphics/img_new/",
    "pdf": f"{dirs}{subfolders}graphics/pdf_new/",
    "cost_img": f"{dirs}{subfolders}costs/img_new/",
    "cost": f"{dirs}{subfolders}costs/",
    "img_acc": f"{dirs}{subfolders}graphics/img_new/acc/",
    "pdf_acc": f"{dirs}{subfolders}graphics/pdf_new/acc/",
}

for folder in folders.values():
    os.makedirs(folder, exist_ok=True)

# ----------------------------------------------------------------------
# Main evaluation loop
# ----------------------------------------------------------------------
count_bad_error = 0
max_bad_error, max_good_error = 0.0, 0.0
bad_errors, good_errors = [], []
bad_error_ids = []
counter = 0

for i in range(n_tests):
    log_messages = []
    directory = dirs + "test"

    fig1, ax1 = plt.subplots(figsize=(8, 11))
    fig2, (ax6, ax7) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.grid(False)
    ax1.plot([-0.0045, -0.0045], [0, 0.25], "r--", linewidth=1)
    ax1.plot([0.1545, 0.1545], [0, 0.25], "r--", linewidth=1)
    ax1.plot([-0.05, 0.25], [0, 0], "k", linewidth=0.5)
    ax1.set_xlim([-0.02, 0.17])
    ax1.set_ylim([-0.01, 0.32])
    ax1.set_xlabel("X (m)")

    for j in range(n_tests_up):
        ins = np.array([ins_list[i*n_tests_up+j,0],ins_list[i*n_tests_up+j,1],ins_list[i*n_tests_up+j,2],0])

        ins_offset_vec = np.array([ins[0] + offset_in1, ins[1] + offset_in2, ins[2] + offset_in3, ins[3]])

        obst_dims = np.array([ins[1] * L_demo, ins[2] * L_demo, ins[0] * L_demo])
        ins_torch = torch.tensor(ins_offset_vec[:n_ins], dtype=torch.float32)

        # NN inference
        with torch.no_grad():
            start_time = time.time()
            new_means = loaded_model(ins_torch)
            nn_inference_ms = (time.time() - start_time) * 1000

        new_means = np.reshape(new_means, (pi2.n_bfs_p, 2))
        pi2.dmp.w[1, :] = new_means[:, 0]
        pi2.dmp.w[2, :] = new_means[:, 1]

        # Rollout and evaluation
        x_learned, dx_learned, ddx_learned, t_learned, _, _ = pi2.dmp.rollout(tau=pi2.tau)
        policy_cost, _, _, y_borders, min_max_x_z = pi2.evaluate_rollout_continuous_box_acc(
            x_learned, ddx_learned, ins[1:3] * L_demo, ins[0] * L_demo, mode
        )

        all_costs[j, :] = policy_cost
        height_error = -policy_cost[1] / L_demo - ins[0]

        # Error tracking
        if height_error > 0:
            good_errors.append(height_error)
            max_good_error = max(max_good_error, abs(height_error))
        else:
            count_bad_error += 1
            bad_errors.append(abs(height_error))
            bad_error_ids.append(counter)
            max_bad_error = max(max_bad_error, abs(height_error))

        # Visualization
        if height_error > 0.0:
            col = np.array([0, 1.0, 0.0])
        else:
            col = np.array([1, 0.0, 0.0])
        pi2.plot_rollout(ax1, col, 4)
        ax1.plot(ins[1]*L_demo,ins[0]*L_demo,'*',color=col)
        ax1.plot(ins[2]*L_demo,ins[0]*L_demo,'*',color=col)
        ax6.plot(t_learned, ddx_learned[:, 1], "r")
        ax7.plot(t_learned, ddx_learned[:, 2], "r")

        counter += 1

    # Save logs and figures
    print(f"completed test {i+1}/{n_tests} with offset: {ins_offset}")
    np.savetxt(f"{folders['cost']}/{i}_logging.txt", all_costs, fmt="%.6f", delimiter=",")

    fig1.savefig(f"{folders['pdf']}fig_{i}.pdf")
    fig1.savefig(f"{folders['img']}img_{i}.png")
    fig2.savefig(f"{folders['pdf_acc']}fig_{i}.pdf")
    fig2.savefig(f"{folders['img_acc']}img_{i}.png")

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
bad_error_mean = np.mean(bad_errors) if bad_errors else 0
good_error_mean = np.mean(good_errors) if good_errors else 0
success_rate = 100 - (count_bad_error / counter * 100)

print(f"Completed {counter} tests.")
print(f"Success rate: {success_rate:.2f}%")
print(f"Bad error mean: {bad_error_mean:.4f}, max: {max_bad_error:.4f}")
print(f"Good error mean: {good_error_mean:.4f}, max: {max_good_error:.4f}")
print(f"Bad error IDs: {bad_error_ids}")

plt.ion()
plt.show()
