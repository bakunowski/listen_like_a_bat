import os
import pandas as pd

# Path to folder with folders containing .csv files with impulse responses
folderpath = "/homes/kb316/FinalProject/Data/Train_Test_Split/Train/Original/"

new_folderpath = "/homes/kb316/FinalProject/Data/Train_Test_Split/Train/Random10/"

for dirpath, dirnames, filenames in os.walk(folderpath):
    for filename in [f for f in filenames if f.endswith(".csv")]:
        data = pd.read_csv(folderpath + filename, sep=';', header=None)
        data = data.fillna(0)
        split = data.sample(frac=1)
        split.to_csv(new_folderpath + filename, header=False, index=False)
