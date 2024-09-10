To generate cortical thickness data run, "generate.py" under reconstruction folder.
One needs to adjust parameters like model name, factor and rest to get desired result.
It generates data one-by-one but the process could be automated using for loop.
Files generated are saved in "generated_data" folder under guided_vae.

After Generating all files;
In order to convert .npy generated files into .thickness file, one needs to run "freesurfer_format.py"
with appropriate subject name. It converts files one-by-one.
.thickness files are used to visualize data using freeview tool.
