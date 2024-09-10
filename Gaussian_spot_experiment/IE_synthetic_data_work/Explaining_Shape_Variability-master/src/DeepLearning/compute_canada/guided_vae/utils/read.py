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






import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import numpy as np
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
    #x = torch.tensor(np.load(file_path), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, face=face, y=y)



