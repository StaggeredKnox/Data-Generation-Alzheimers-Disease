import trimesh as tm
import nibabel as nib
import numpy as np
from nibabel.freesurfer.io import write_geometry, write_morph_data

#mesh = tm.load_mesh("decimated_mesh.ply")

#vertices = mesh.vertices
#faces = mesh.faces
#write_geometry('temp.inflated', vertices, faces)

sample_path = "gen_1.000000.npy"

thickness_data = np.load('/home/rajk/Desktop/generated_data/'+sample_path)
write_morph_data('gen_sDAT.thickness', thickness_data)
