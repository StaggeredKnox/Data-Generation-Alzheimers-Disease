import trimesh as tm
import nibabel as nib
import numpy as np
import pyvista as pv

from config import config_vars as cfg


path = cfg["path"]
subject = cfg["subject"]
template_file = cfg["template_file"]

color_3c = np.load(path+subject)

template_mesh = tm.load(path+template_file)

colors = np.zeros((len(color_3c), 4), dtype=np.float32)

colors[:, :3]= color_3c

template_mesh.visual.vertex_colors = colors

template_mesh.export(path+"dummy.ply", encoding="ascii")

template_mesh = pv.read(path+"dummy.ply")

plot = pv.Plotter()

plot.add_mesh(template_mesh)

plot.show()




