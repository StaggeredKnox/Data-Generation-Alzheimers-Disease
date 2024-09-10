import pymeshlab
import numpy as np
from scipy.spatial import KDTree
import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import json

from config import config_vars as cfg


data_path = cfg["data_path"]
simplified_data_path = cfg["simplified_data_path"]
template = cfg["template"]
simplified_template = cfg["simplified_template"]

if not os.path.exists(simplified_data_path):
    os.makedirs(simplified_data_path, exist_ok=True)

ms = pymeshlab.MeshSet()

ms.load_new_mesh(template)
original_mesh = ms.current_mesh()
original_vertices = np.array(original_mesh.vertex_matrix())

ms.load_new_mesh(simplified_template)
simplified_mesh = ms.current_mesh()
simplified_vertices = np.array(simplified_mesh.vertex_matrix())

tree = KDTree(original_vertices)
distances, indices = tree.query(simplified_vertices)
#indices = tree.query_ball_point(simplified_vertices, 0.0)
#print(indices.shape)
#print(indices)
#print(indices.dtype)
#print(distances)
#exit()

fps = glob(osp.join(data_path, '*'))

for idx, fp in enumerate(tqdm(fps)):
    subject = fp.split("/")[-1].split(".")[0]
    if subject=="labels" or subject=="new_labels":
        continue
    with open(fp, 'r') as file:
        dat = np.array(json.load(file))
        simplified_dat = dat[indices]
        #simplified_dat = np.nan_to_num(simplified_dat)
        #print(np.isnan(simplified_dat).any())
        np.save(f"{simplified_data_path}/{subject}.npy", simplified_dat)




