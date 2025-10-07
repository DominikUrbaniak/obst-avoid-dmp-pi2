import sys
import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim


# ----------------------------------------------------------------------
# Dataset class
# ----------------------------------------------------------------------
class TrajectoryDataset(Dataset):
    """PyTorch dataset for trajectory learning."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ----------------------------------------------------------------------
# Neural network model
# ----------------------------------------------------------------------
class RBFNet(nn.Module):
    """Simple feed-forward network (MLP) to approximate DMP forcing term parameters."""

    def __init__(self, input_size, output_size):
        super(RBFNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # output is linear for regression
        return x


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
id = "01"

sample_ratio = 0  # if >0, use ratio instead of fixed n_samples
n_samples = 1000  # number of samples per run
if sample_ratio:
    n_samples = -1  # dynamically set later

n_rbf = 11
n_dims = 2
n_in = 3
L_demo = 0.15
# Get training directory from command line
if len(sys.argv) > 1:
    dirs = f"results/{sys.argv[1]}"
else:
    dirs = "results/default/"
if len(sys.argv) > 2:
    model_name = f"{sys.argv[2]}"
else:
    model_name = f"model"
data_size = sum(os.path.isdir(os.path.join(dirs, d)) for d in os.listdir(dirs))-1
print(f"Considered data size: {data_size}")
# ----------------------------------------------------------------------
# Load PI2 data
# ----------------------------------------------------------------------
iters = np.zeros(data_size, dtype=int)
individual_nn_output = {}
individual_nn_input = {}
non_zero_counter = 0

for j in range(data_size):
    file_path = f"{dirs}train{j}/train_{j}.npz"
    if not os.path.exists(file_path):
        continue

    data = np.load(file_path, allow_pickle=True)
    iters[j] = int(data["n_episodes"])

    if iters[j] == 0:
        continue

    nn_output = data["training_data"]
    individual_nn_output[non_zero_counter] = np.reshape(
        nn_output[:, :, 1:3], (iters[j], n_rbf * n_dims)
    )

    z_min = np.min(data["labels"], axis=1)
    label_set = z_min / L_demo
    label_set_all = np.column_stack(
        (
            label_set,
            data["y_locations"] / L_demo,
        )
    )
    individual_nn_input[non_zero_counter] = label_set_all
    non_zero_counter += 1

# ----------------------------------------------------------------------
# Build training matrices
# ----------------------------------------------------------------------
min_samples = int(np.min(iters[np.nonzero(iters)]))
print("min_samples:", min_samples)

if n_samples > min_samples:
    n_samples = min_samples
elif sample_ratio:
    n_samples = max(1, int(iters[j] * n_samples))

nn_output_all = np.zeros((non_zero_counter * n_samples, n_rbf * n_dims))
nn_input_all = np.zeros(
    (non_zero_counter * n_samples, individual_nn_input[0].shape[1])
)

for i in range(non_zero_counter):
    nn_output = individual_nn_output[i]
    label_set = individual_nn_input[i]
    samples = (
        np.round(np.linspace(0, iters[i] - 1, n_samples)).astype(int)
        if n_samples > 1
        else [-1]
    )

    nn_output_all[i * n_samples : (i + 1) * n_samples, :] = nn_output[samples, :]
    nn_input_all[i * n_samples : (i + 1) * n_samples, :] = label_set[samples, :]

print(f"Dataset prepared: X={nn_input_all.shape}, Y={nn_output_all.shape}")

# ----------------------------------------------------------------------
# Create PyTorch datasets and loaders
# ----------------------------------------------------------------------
dataset = TrajectoryDataset(nn_input_all, nn_output_all)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ----------------------------------------------------------------------
# Model, loss, and optimizer
# ----------------------------------------------------------------------
model = RBFNet(n_in, n_rbf * n_dims)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
num_epochs = 100
start_time = time.time()

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_targets).item()
    val_loss /= len(val_loader)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}"
    )

training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds.")

# ----------------------------------------------------------------------
# Save model and training data summary
# ----------------------------------------------------------------------
scripted_model = torch.jit.script(model)
os.makedirs(dirs, exist_ok=True)

max_in1 = np.round(np.max(nn_input_all[:, 0]), 2)
min_in2 = np.round(np.min(nn_input_all[:, 1]), 2)
max_in3 = np.round(np.max(nn_input_all[:, 2]), 2)
#file_name = f"{data_size}_{max_in1}_{min_in2}_{max_in3}"

# Save model
model_path= os.path.join(dirs, f"{model_name}.pth")
scripted_model.save(model_path)

# Save dataset summary
np.savez(
    os.path.join(dirs, f"{model_name}_data.npz"),
    nn_input=nn_input_all,
    nn_output=nn_output_all,
    iters=iters,
    t_nn=training_time,
    max_min_ins=[max_in1, min_in2, max_in3],
)

print(f"Model saved to {model_path}")
