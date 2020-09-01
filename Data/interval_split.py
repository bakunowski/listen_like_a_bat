import os
import random
import pandas as pd
import numpy as np

# Path to folder containing csv files with IRs
folderpath = "/homes/kb316/FinalProject/Data/Train_Test_Validation/Validation/Original/"

new_folderpath = "/homes/kb316/FinalProject/Data/Train_Test_Validation/Validation/Interval10/"

num_inputs=10

for dirpath, dirnames, filenames in os.walk(folderpath):
    for filename in [f for f in filenames]:
        data = pd.read_csv(folderpath + filename, header=None)
        data = data.fillna(0)
        while data.size >= num_inputs:
           interval = random.randint(1, 8)
           keys = np.arange(0, interval*num_inputs, interval)
           if keys[-1] >= data.size:
               continue
           split = data.iloc[keys, ]
           split.to_csv(new_folderpath + filename + '%d' % random.randint(0, 1000), header=False, index=False)
           data = data.drop(split.index)
