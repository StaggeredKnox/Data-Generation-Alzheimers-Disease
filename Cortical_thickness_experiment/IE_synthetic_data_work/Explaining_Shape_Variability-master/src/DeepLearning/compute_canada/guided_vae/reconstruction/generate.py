import torch
from reconstruction import AE
import numpy as np
import os
import os.path as osp

device = torch.device('cuda', 0)

base_path = "/home/rajk/Desktop/IE_synthetic_data_work/Explaining_Shape_Variability-master/src/DeepLearning/compute_canada/guided_vae/"

model_path = base_path+"models/141"  

model_state_dict = torch.load(f"{model_path}/model_state_dict.pt")
in_channels = torch.load(f"{model_path}/in_channels.pt")
out_channels = torch.load(f"{model_path}/out_channels.pt")
latent_channels = torch.load(f"{model_path}/latent_channels.pt")
spiral_indices_list = torch.load(f"{model_path}/spiral_indices_list.pt")
up_transform_list = torch.load(f"{model_path}/up_transform_list.pt")
down_transform_list = torch.load(f"{model_path}/down_transform_list.pt")

model = AE(in_channels, out_channels, latent_channels, spiral_indices_list, down_transform_list, up_transform_list)          # Create an instance of the model
model.load_state_dict(model_state_dict)
model.to(device)
model.eval()     # Set the model to evaluation mode

std = torch.load(f"{model_path}/std.pt").to(device)
mean = torch.load(f"{model_path}/mean.pt").to(device)

Z = torch.zeros(1, latent_channels).to(device)
#Z = torch.randn_like(Z)
print(Z.shape)


# possible values of factor for cortical thickness data are : 0.0, 1/6, 2/6, 3/6, 4/6, 5/6 and 1.
factor = 6/6.0    # for gaussian_spot_data it is spot_angle = [-1, 1] and for cortical_thickness_data it is disease severity = [0, 1] or 7 categorical values

Z[0] = factor


print("generated ------------->")
X = model.decoder(Z).squeeze(0)
print(X)
print(X.shape)

X = X * std + mean
print("After readjusting ----->")
print(X)
print(X.shape)


import trimesh as tm
import numpy as np
import pyvista as pv

template_file = base_path+"/data/CoMA/template/template.ply"

color_3c = X.detach().cpu()

gen_dat_path = base_path+"generated_data"
if not os.path.exists(gen_dat_path):
    os.makedirs(gen_dat_path, exist_ok=True)

np.save(f"{gen_dat_path}/gen_{factor:04f}.npy", color_3c[:, 0])



