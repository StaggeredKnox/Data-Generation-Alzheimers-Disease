import openmesh as om

from config import config_vars as cfg


def read_mesh(file_path):
    mesh = om.read_trimesh(file_path)
    if not mesh.has_face_normals():
        mesh.request_face_normals()
        mesh.update_face_normals()
    if not mesh.has_vertex_normals():
        mesh.request_vertex_normals()
        mesh.update_vertex_normals()
    return mesh

def decimate_mesh(mesh, target_face_count):
    decimater = om.TriMeshDecimater(mesh)
    
    # Standard setup for decimation, you can add other modules as needed
    module_handle = om.TriMeshModQuadricHandle()
    decimater.add(module_handle)
    decimater.initialize()
    
    # Perform decimation
    decimater.decimate_to_faces(target_face_count)
    mesh.garbage_collection()
    
    return mesh

def save_mesh(mesh, file_path):
    om.write_mesh(mesh, file_path)

def main():
    input_path = cfg["input_path"]
    output_path = cfg["output_path_dm"]
    target_face_count = cfg["target_face_count"]  # Adjust this number as needed
    
    # Read the original mesh
    mesh = read_mesh(input_path)
    
    # Decimate the mesh
    simplified_mesh = decimate_mesh(mesh, target_face_count)
    
    # Save the simplified mesh
    save_mesh(output_path, simplified_mesh)
    print(f'Simplified mesh saved to {output_path}')

if __name__ == "__main__":
    main()

