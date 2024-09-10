import trimesh as tm
import nibabel as nib
import numpy as np
from nibabel.freesurfer.io import write_geometry, write_morph_data

sample_path = "gen_1.000000"

base_path = "/home/rajk/Desktop/IE_synthetic_data_work/Explaining_Shape_Variability-master/src/DeepLearning/compute_canada/guided_vae"

thickness_data = np.load(base_path+"/generated_data/"+sample_path+".npy")
write_morph_data('gen_1.thickness', thickness_data)
