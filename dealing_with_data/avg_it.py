import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import torch
import shutil as sh
import tarfile
import numpy as np
import nibabel as nib
import pandas as pd
import random
import numpy as np
from nibabel.freesurfer.io import write_geometry, write_morph_data

from config import config_vars as cfg



# Specify the path to the zip file
tar_file_path = cfg["tar_file_path"]

adni_labels_path = cfg["adni_labels_path"]

extract_path = cfg["extract_path"]

if not os.path.exists(extract_path):
    os.makedirs(extract_path, exist_ok=True)



# Creating labels for entire ADNI-STRATIFIED-MRI-7740 dataset
labels = {}

codes = ["sNC", "uNC", "sMCI", "pNC", "pMCI", "eDAT", "sDAT"]  # first 3 = DAT- , last 4 = DAT+

fps = glob(osp.join(adni_labels_path, 'Target*'))

for idx, fp in enumerate(tqdm(fps)):
    code = fp.split("/")[-1].split("-")[-1]
    if code in codes:
        print(code)
        lines = []
        with open(fp, 'r') as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]
        for key in lines:
            labels[key] = codes.index(code)

# Shuffle the labels and save
random_seed = cfg["random_seed"]
random.seed(random_seed)

items = list(labels.items())
random.shuffle(items)
labels = dict(items)




# Create labels with specific number of instances and it is random because of above shuffling
new_labels = {}
value_instances = cfg["value_instances"]
lookup = np.ones(len(codes))*value_instances
for k, v in labels.items():
    if lookup[v]>0:
        lookup[v] = lookup[v]-1
        new_labels[k] = v/(len(codes)-1.0)    

print("\nInstances stored corresponding to each value :\n")
for item in codes:
    print(f"{item} ---> {value_instances-lookup[codes.index(item)]}")
print("\n")




# Creating the dataset from the new_labels
fps = glob(osp.join(tar_file_path, '*.tar'))

avg_dat = {"sNC": 0, "uNC": 0, "sMCI": 0, "pNC": 0, "pMCI": 0, "eDAT": 0, "sDAT": 0}

book = np.array([0.0, 1/6.0, 2/6.0, 3/6.0, 4/6.0, 5/6.0, 1.0])

verify_lookup = {"sNC": 0, "uNC": 0, "sMCI": 0, "pNC": 0, "pMCI": 0, "eDAT": 0, "sDAT": 0}

for idx, fp in enumerate(tqdm(fps)):
    with tarfile.open(fp, 'r') as tar:

        file_list = tar.getnames()          # e.g. ['2_bl', '2_bl/0', '2_bl/0/lh.thickness', '2_bl/0/rh.thickness', '2_bl/15', '2_bl/15/lh.thickness', '2_bl/15/rh.thickness']

        if file_list[0] not in new_labels.keys():
            continue

        members = [m for m in tar.getmembers() if m.name.startswith(file_list[1])]
    
        tar.extractall(path=extract_path, members=members)

        lh_fp = extract_path+'/'+file_list[2]
        rh_fp = extract_path+'/'+file_list[3]
    
        lh_thickness = nib.freesurfer.read_morph_data(lh_fp)
        rh_thickness = nib.freesurfer.read_morph_data(rh_fp)
    
        data = np.array(lh_thickness.tolist() + rh_thickness.tolist())
        cls = codes[int(labels[file_list[0]])]
        verify_lookup[cls] = verify_lookup[cls] + 1
        avg_dat[cls] = avg_dat[cls] + (data / (value_instances-lookup[codes.index(cls)]))

        temp = extract_path+'/'+file_list[0]
        if os.path.exists(temp):
            sh.rmtree(temp)

if os.path.exists(extract_path):
    sh.rmtree(extract_path)

print("\nPrinting verification lookup  ----->\n")
for k, v in verify_lookup.items():
    print(f"{k} ------------- {v}")
print("verify the table from previous table...")


avg_dat_path = "/home/rajk/Desktop/dealing_with_data/averaged_data"

if not os.path.exists(avg_dat_path):
    os.makedirs(avg_dat_path, exist_ok=True)


for k, v in avg_dat.items(): 
    write_morph_data(f"{avg_dat_path}/avg_{k}.thickness", v)








