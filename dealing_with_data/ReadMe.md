Folder named "dealing_with_data" deals with brain mesh and cortical thickness data, it has 4 python files:

		(1)  create_mesh.py --->  creates mesh(in .ply format) from 
							      freesurfer surfaces files.


        (2)  decimate_mesh.py --->  reduces mesh nodes and edges for 
									quick processing.

		
		(3)  get_data.py --->  extracts cortical thickness in .json format 
							   for various samples for ADNI dataset.


		(4)  simplify_data --->  simplifies cortical thickness data as per
								 simplified mesh and saves in .npy format.


config.py file contains all vaiables important to the task.

running the files in following order : (1) - (2) - (3) - (4)

after running all the files, simplified mesh and simplified data with "new_labels.pt" taken for model training.



EXTRA NOTES:

		(1)  freesurfer_format.py script takes up a .npy file for thickness data and converts
			 it to freesurfer friendly format i.e. .thickness for visualization via freeview.
             (look freeview tutorial for loading and visualizing data)

		
		(2)  avg_it.py takes values instances for each class over which it averages the ground truth data classwise/categorywise
             and produces 7 .npy files one for each class. It saves all 7 files in the folder named "averaged_data".
			 
		 	 This folder contains "redcuce.py" which reduces the 7 files according to 
			 "temp.inflated" (it could be any template which has been decimated) and converts the averaged thickness data
             into a freesurfer friendly format for visualization.

			 The reduced thickness data files are saved in "avg_simplified" folder after running "redcuce.py".

















	
