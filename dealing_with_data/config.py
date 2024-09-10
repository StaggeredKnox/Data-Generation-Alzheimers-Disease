config_vars = {


###  configuration for create_mesh.py  ###

    "lh_path" : '/home/rajk/Desktop/dealing_with_data/MNI152_template/FSL_MNI152/surf/lh.pial',

    "rh_path" : '/home/rajk/Desktop/dealing_with_data/MNI152_template/FSL_MNI152/surf/rh.pial',

    "output_path_cm" : '/home/rajk/Desktop/dealing_with_data/combined_mesh.ply',



###  configuration for decimate_mesh.py  ###

    "input_path" : '/home/rajk/Desktop/dealing_with_data/combined_mesh.ply',

    "output_path_dm" : '/home/rajk/Desktop/dealing_with_data/decimated_mesh.ply',

    "target_face_count" : 29784,  # adjust this number as needed



###  configuration for get_data.py  ###

    "tar_file_path" : '/home/rajk/projects/rrg-mfbeg-ad/faisal_group/ARCHIVE/PROJECTS/ADNI/PROCESSED_DATA/CorticalMeasures/FreesurferThickness/Registered/FSL_MNI152',

    "extract_path" : '/home/rajk/Desktop/dealing_with_data/dummy',

    "store_data_path" : '/home/rajk/Desktop/dealing_with_data/cortical_thickness',

    "adni_labels_path" : '/home/rajk/Desktop/dealing_with_data/ADNI-STRATIFIED-MRI-7740',

    "random_seed" : 50,  # change only if needed

    "value_instances" : 600,  # change this to adjust no. of samples from each class



###  configuration for simplify_data.py  ###

    "data_path" : "/home/rajk/Desktop/dealing_with_data/cortical_thickness",

    "simplified_data_path" : "/home/rajk/Desktop/dealing_with_data/simplified_cortical_thickness",

    "template" : "combined_mesh.ply",

    "simplified_template" : "decimated_mesh.ply",

    }
