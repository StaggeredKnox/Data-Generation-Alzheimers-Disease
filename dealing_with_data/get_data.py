import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import json
import torch
import shutil as sh
import tarfile
import numpy as np
import nibabel as nib
import pandas as pd
import random

from config import config_vars as cfg



# Specify the path to the zip file
tar_file_path = cfg["tar_file_path"]

extract_path = cfg["extract_path"]

store_data_path = cfg["store_data_path"]

adni_labels_path = cfg["adni_labels_path"]

# Create the directory if it doesn't exist
if not os.path.exists(extract_path):
    os.makedirs(extract_path, exist_ok=True)

if not os.path.exists(store_data_path):
    os.makedirs(store_data_path, exist_ok=True)




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

torch.save(labels, store_data_path+"/labels.pt")
pd.DataFrame(list(labels.items()), columns=['shape', 'label']).to_csv(store_data_path+"/labels.csv")




# Create labels with specific number of instances and it is random because of above shuffling
new_labels = {}
value_instances = cfg["value_instances"]
lookup = np.ones(len(codes))*value_instances
for k, v in labels.items():
    if lookup[v]>0:
        lookup[v] = lookup[v]-1
        new_labels[k] = v/(len(codes)-1.0)    

torch.save(new_labels, store_data_path+"/new_labels.pt")
pd.DataFrame(list(new_labels.items()), columns=['shape', 'label']).to_csv(store_data_path+"/new_labels.csv")

print("\nInstances stored corresponding to each value :\n")
for item in codes:
    print(f"{item} ---> {value_instances-lookup[codes.index(item)]}")
print("\n")




# Creating the dataset from the new_labels
fps = glob(osp.join(tar_file_path, '*.tar'))

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
    
        data = lh_thickness.tolist() + rh_thickness.tolist()
        with open(store_data_path+'/'+file_list[0]+".json", 'w') as json_file:
            json.dump(data, json_file)

        temp = extract_path+'/'+file_list[0]
        if os.path.exists(temp):
            sh.rmtree(temp)   

if os.path.exists(extract_path):
    sh.rmtree(extract_path) 







