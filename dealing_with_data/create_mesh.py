import nibabel as nib
import openmesh as om
import numpy as np

from config import config_vars as cfg


def read_freesurfer_mesh(file_path):
    surf = nib.freesurfer.read_geometry(file_path)
    vertices, faces = surf[0], surf[1]
    return vertices, faces

def create_openmesh(vertices, faces):
    mesh = om.TriMesh()
    
    vertex_handles = []
    for vertex in vertices:
        vertex_handle = mesh.add_vertex(vertex)
        vertex_handles.append(vertex_handle)
    
    for face in faces:
        face_vhandles = [vertex_handles[idx] for idx in face]
        mesh.add_face(face_vhandles)
    
    return mesh

def combine_meshes(mesh1, mesh2):
    
    combined_mesh = om.TriMesh()
    
    vh_map = {}
    
    for vh in mesh1.vertices():
        point = mesh1.point(vh)
        new_vh = combined_mesh.add_vertex(point)
        vh_map[(1, vh.idx())] = new_vh
    
    for vh in mesh2.vertices():
        point = mesh2.point(vh)
        new_vh = combined_mesh.add_vertex(point)
        vh_map[(2, vh.idx())] = new_vh
    
    for fh in mesh1.faces():
        fvhs = [vh_map[(1, vh.idx())] for vh in mesh1.fv(fh)]
        combined_mesh.add_face(fvhs)
    
    for fh in mesh2.faces():
        fvhs = [vh_map[(2, vh.idx())] for vh in mesh2.fv(fh)]
        combined_mesh.add_face(fvhs)

    combined_mesh.request_vertex_normals()
    combined_mesh.update_normals()
    
    return combined_mesh


def main():
    lh_pial_path = cfg["lh_path"]
    rh_pial_path = cfg["rh_path"]
    output_path = cfg["output_path_cm"]
    
    # Read FreeSurfer mesh files
    lh_vertices, lh_faces = read_freesurfer_mesh(lh_pial_path)
    rh_vertices, rh_faces = read_freesurfer_mesh(rh_pial_path)
    
    # Create OpenMesh meshes
    lh_mesh = create_openmesh(lh_vertices, lh_faces)
    rh_mesh = create_openmesh(rh_vertices, rh_faces)
    
    # Combine the left and right hemisphere meshes
    combined_mesh = combine_meshes(lh_mesh, rh_mesh)
    
    # Save the combined mesh to a .ply file
    om.write_mesh(output_path, combined_mesh)
    print(f'Combined mesh saved to {output_path}')

if __name__ == "__main__":
    main()

