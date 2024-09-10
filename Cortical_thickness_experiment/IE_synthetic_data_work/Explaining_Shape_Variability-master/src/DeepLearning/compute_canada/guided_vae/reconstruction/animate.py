import numpy as np
import pyvista as pv
import matplotlib.pyplot as cm
from glob import glob
import os.path as osp
import torch
from reconstruction import AE
import os

dataset_name = "torus"
name = "cortical_thickness"
model_number = "141"
ground_truth = False    # if True, plots ground truth animation.
factor_lower_bound = 0
factor_upper_bound = 1

base_path = "/home/rajk/Desktop/IE_synthetic_data_work/Explaining_Shape_Variability-master/src/DeepLearning/compute_canada/guided_vae"
saving_path = base_path+"/animations/"
template_file_path = base_path+"/data/CoMA/template/template.ply"

if not os.path.exists(saving_path):
    os.mkdir(saving_path)

plotter = pv.Plotter()

if ground_truth==True:
    plotter.open_gif(saving_path+name+"(ground_truth).gif")
else:
    plotter.open_gif(saving_path+name+"(generated_via_Z).gif")    

colormap = cm.get_cmap('inferno')

def update_colors(M):
    colors = (colormap(M)[:, :3] * 256/2.45)  # Convert to RGB
    maxs = np.max(colors[:, 0])
    print(maxs)
    mesh = pv.read(template_file_path)
    mesh.point_data['colors'] = colors
    plotter.add_mesh(mesh)#, scalars='colors', show_scalar_bar=True)#, rgb=True, show_scalar_bar=False)
    plotter.write_frame()

def main(plot_ground_truth=True):

    if plot_ground_truth==True:
        fps = sorted(glob(osp.join(base_path, f"data/CoMA/raw/{dataset_name}/*.npy")))[0:700:10]
        print(len(fps))
        for idx, fp in enumerate(fps):
            #print(fp)
            plotter.clear()
            X = np.load(fp)
            color_matt = np.zeros((len(X), 3))  # for cortical data
            color_matt[:, 0] = X          # for cortical data
            #color_matt = X        # for spot data
            update_colors(color_matt)
    else:
        #factors = np.linspace(factor_lower_bound, factor_upper_bound, 5000)[0:5000:100]
        factors = np.array([0, 1/6.0, 2/6.0, 3/6.0, 4/6.0, 5/6.0, 6/6.0])

        model_path = base_path+f"/models/{model_number}"

        model_state_dict = torch.load(f"{model_path}/model_state_dict.pt")
        in_channels = torch.load(f"{model_path}/in_channels.pt")
        out_channels = torch.load(f"{model_path}/out_channels.pt")
        latent_channels = torch.load(f"{model_path}/latent_channels.pt")
        spiral_indices_list = torch.load(f"{model_path}/spiral_indices_list.pt")
        up_transform_list = torch.load(f"{model_path}/up_transform_list.pt")
        down_transform_list = torch.load(f"{model_path}/down_transform_list.pt")
 
        device = torch.device('cpu')
        if torch.cuda.is_available()==True:
            device = torch.device('cuda', 0)

        model = AE(in_channels, out_channels, latent_channels, spiral_indices_list, down_transform_list, up_transform_list)          # Create an instance of the model
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()     # Set the model to evaluation mode

        std = torch.load(f"{model_path}/std.pt").to(device)
        mean = torch.load(f"{model_path}/mean.pt").to(device)

        Z = torch.zeros(1, latent_channels).to(device)
        for factor in factors:
            Z[0] = factor
            X = model.decoder(Z).squeeze(0)
            X = X * std + mean
            X = X.detach().cpu()
            color_matt = torch.zeros(len(X), 3)         # for cortical data
            color_matt[:, 0] = X[:, 0]               # for cortical data
            #color_matt = X               # for spot data
            update_colors(color_matt)     

main(plot_ground_truth=ground_truth)

plotter.close()

