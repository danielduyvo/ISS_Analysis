import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from natsort import natsorted
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import os, sys

# Concatenate genotyped cells list,
# 'Cells_Genotyped_Complete_Well_x.csv', from every well into a single
# complete genotyped cell list

path = 'genotyping_results'
well_dirs = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

for well in tqdm(well_dirs):
    i = int(well.split('_')[-1])
    
    path_csv = join('genotyping_results', 'well_' + str(i), 'Cells_Genotyped_Complete_Well_' + str(i) + '.csv')

    df_temp = pd.read_csv(path_csv)
    df_temp['well'] = i*np.ones([len(df_temp)])
    
    if i == 1:
        df_concat = df_temp.copy()
        
    if i > 1:
        df_concat = pd.concat((df_concat, df_temp))

print(df_concat.shape)
df_concat.to_csv('Cells_Genotyped.csv', index=False)
