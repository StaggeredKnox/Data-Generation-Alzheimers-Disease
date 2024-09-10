#import torch
#from torch_geometric.data import Data
#from torch_geometric.utils import to_undirected
#import openmesh as om

#def read_mesh(path):
#    mesh = om.read_trimesh(path)
#    face = torch.from_numpy(mesh.face_vertex_indices()).T.type(torch.long)
#    x = torch.tensor(mesh.points().astype('float32'))
#    edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
#    edge_index = to_undirected(edge_index)
    
#    labels = torch.load("/home/rajk/Desktop/IE_synthetic_data_work/Explaining_Shape_Variability-master/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/labels.pt")
#    subject = str(path.split("/")[-1].split(".")[0])
#    y = torch.Tensor([labels[subject]])


#    return Data(x=x, edge_index=edge_index, face=face, y=y)






import json
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import openmesh as om

def read_mesh(file_path):

    labels_path = "/home/rajk/Desktop/IE_synthetic_data_work/Explaining_Shape_Variability-master/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/labels.pt"

    template_path = "/home/rajk/Desktop/IE_synthetic_data_work/Explaining_Shape_Variability-master/src/DeepLearning/compute_canada/guided_vae/data/CoMA/template/template.ply"

    mesh = om.read_trimesh(template_path)
    face = torch.from_numpy(mesh.face_vertex_indices()).T.type(torch.long)
    #x = torch.tensor(mesh.points().astype('float32'))
    edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
    edge_index = to_undirected(edge_index)
    
    labels = torch.load(labels_path)
    subject = str(file_path.split("/")[-1].split(".")[0])
    y = torch.Tensor([labels[subject]])

    x = torch.from_numpy(np.load(file_path)).type(torch.float32)
    x = torch.reshape(x, (-1, 1))

    #thickness = torch.from_numpy(np.load(file_path)).type(torch.float32)
    #x = torch.zeros(len(thickness), 3)
    #x[:, 0] = thickness

    #with open(file_path, 'r') as file:
    #    dat = json.load(file)
    #    x = torch.zeros(len(dat), 3)
    #    thickness = torch.tensor(dat).type(torch.float32)
    #    x[:, 0] = thickness


    return Data(x=x, edge_index=edge_index, face=face, y=y)



