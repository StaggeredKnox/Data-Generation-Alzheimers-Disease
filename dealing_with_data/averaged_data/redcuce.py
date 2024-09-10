import pymeshlab
import numpy as np
from scipy.spatial import KDTree
import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import json
import nibabel as nib
from nibabel.freesurfer.io import write_geometry, write_morph_data


data_path = "/home/rajk/Desktop/dealing_with_data/averaged_data/"
simplified_data_path = "/home/rajk/Desktop/dealing_with_data/avg_simplified/"
template = "/home/rajk/Desktop/dealing_with_data/combined_mesh.ply"
simplified_template = "/home/rajk/Desktop/dealing_with_data/decimated_mesh.ply"

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


fps = glob(osp.join(data_path, '*'))


for idx, fp in enumerate(tqdm(fps)):
    subject = fp.split("/")[-1].split(".")[0]
    if subject=="redcuce" or subject=="temp":
        continue
    dat = nib.freesurfer.read_morph_data(subject+".thickness")
    simplified_dat = dat[indices]
    write_morph_data(f"{simplified_data_path}/{subject}.thickness", simplified_dat)




