import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import pandas as pd
import os

from config import config_vars as cfg


data_path = cfg["data_path"]
if not os.path.exists(data_path):
    os.mkdir(data_path)

# Load the .ply file
ply_file = cfg["ply_file"]
mesh = o3d.io.read_triangle_mesh(ply_file)
vertices = np.asarray(mesh.vertices)

# Define the function to generate color matrices
def generate_color_matrix(vertices, gaussian_center, sigma, i):
    N = vertices.shape[0]
    M = np.zeros((N, 3))

    # Compute distances from the center point
    distances = np.linalg.norm(vertices - gaussian_center, axis=1)*0.1

    # Apply Gaussian function to simulate the spot
    colors = np.exp(-distances**2 / (2 * sigma**2))
    colors = colors/(np.max(colors))

    if i==0:
        print(vertices.shape)
        print(distances.shape)
        print(colors.shape)
        print((vertices-gaussian_center).shape)
        print(gaussian_center * np.mean(vertices, axis=0))

    # Assign colors (normalized to [0, 1])
    M[:, 0] = colors  # Red channel
    M[:, 1] = colors  # Green channel
    M[:, 2] = colors  # Blue channel

    return M

# Generate series of matrices
torus_center = np.mean(vertices, axis=0)
dists = np.linalg.norm(vertices-torus_center, axis=1)
torus_outer_radius = np.max(dists)

name = cfg["name"]
labels = {}
sigma = cfg["sigma"]  # Width of the spot
color_matrices = []
number_of_samples = cfg["number_of_samples"]

# Move the spot along the vertices in a circular manner
for i, spot_angle in enumerate(tqdm(np.linspace(-1, 1, number_of_samples))):
    angle = np.pi * spot_angle
    gaussian_center = (np.array([np.cos(angle), np.sin(angle), 0]) * torus_outer_radius) + torus_center
    #print(f"{np.pi}    {angle}    {gaussian_center}")
    M = generate_color_matrix(vertices, gaussian_center, sigma, i)
    color_matrices.append(M)

    filename = f"{name}_{i:04d}"
    labels[filename] = spot_angle

torch.save(labels, data_path+"/labels.pt")
pd.DataFrame(list(labels.items()), columns=['shape', 'label']).to_csv(data_path+"/labels.csv")

for i, M in enumerate(color_matrices):
    np.save(f"{data_path}/{name}_{i:04d}.npy", M)

# Visualize the first frame (optional)
#colors = color_matrices[0]
#mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
#o3d.visualization.draw_geometries([mesh])

# Save color matrices to files if needed


