import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from natsort import natsorted
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import os, sys

path = 'phenotyping_results'
files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.csv')]

for i, file in tqdm(enumerate(files)):
    
    path_csv = join(path, file)
    df_temp = pd.read_csv(path_csv)
    
    if i == 0:
        df_concat = df_temp.copy()
        
    if i > 0:
        df_concat = pd.concat((df_concat, df_temp))

try:
    df_new = df_new.drop(columns=['Unnamed: 0'])
except:
    print('No Unnamed columns')

# Omitted cells not phenotyped
df = df_concat.copy()
n_0 = len(df)
df_new = df[df['area_nuc'] != -1]
n_1 = len(df_new)

print('Original cells:', n_0, '\nCells omitted:', n_0 - n_1, '\nFraction of retained cells:', n_1/n_0)

print('done')
df_new.to_csv('Cells_Phenotyped.csv', index=False)
