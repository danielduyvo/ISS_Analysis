import sys, os
from datetime import datetime
import matplotlib.pyplot as plt
import In_Situ_Functions as isf
import pandas as pd
import numpy as np
import argparse

if __name__ == '__main__':

    t_start=datetime.now()

    # Take job number from shell script, requires 5 digits in the form: well (_), tile (_,_,_,_)
    # Example well 2 tile 348 to be inputed as 20348
    # This allows running the segmentation on all tiles in parallel
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=int)
    args = parser.parse_args()
    n_job = args.job
    n_well = int(np.floor(n_job / 10000))
    n_tile = int(np.around(10000 * ((n_job / 10000) % 1)))

    save_path = 'genotyping_results/well_' + str(n_well)

    # Load barcode-sgRNA library
    # Subdivide the library according to well, if necessary
    df_lib = pd.read_csv('RBP_F2_Bulk_Optimized_Final.csv')
    if n_well == 1:
        df_lib = df_lib[df_lib['group'] == 1]
    if n_well == 2 or n_well == 3:
        df_lib = df_lib[df_lib['group'] == 3]
    if n_well >= 4:
        df_lib = df_lib[df_lib['group'] == 2]

    # Creating saving dirs in case they don't exist
    # Results will be split by well
    isf.Manage_Save_Directories(save_path)

    # Compline a tile (of a certain well) from multiple cycle dirs
    # Output is Cy x Ch x H x W, Ch
    # Usually channel order is: DAPI, G, T, C, A (in cycle 1 syto12 is added at the end for cellular segmentation at 10X)
    data = isf.InSitu.Assemble_Data_From_ND2(n_tile, n_well, 'genotyping')

    # Feldman et al. Peak calling code from Feldman et al. 2019
    maxed, peaks, _ = isf.InSitu.Find_Peaks(data, verbose=False)

    # Loading pre-segmented nuclear and cellular masks
    nucs = np.load(os.path.join('segmented', '10X', 'nucs', 'well_' + str(n_well), 'Seg_Nuc-Well_'+ str(n_well) + '_Tile_' + str(n_tile) + '.npy'))[1]
    cells = np.load(os.path.join('segmented', '10X', 'cells', 'well_' + str(n_well), 'Seg_Cells-Well_'+ str(n_well) + '_Tile_' + str(n_tile) + '.npy'))[1]

    # Calling bases from peaks. Peak THRESHOLD_STD can be optimized manually, typically between 200-400
    df_reads, _ = isf.InSitu.Call_Bases(cells, maxed, peaks, 200)
    
    df_reads_amb = isf.InSitu.Assign_Simple_Ambiguity(df_reads, lim=0.3)

    df_cell_genotype = isf.Lookup.Choose_Barcodes(df_reads_amb, df_lib, nucs, cells, verbose=True)

    isf.Save(save_path, n_tile, n_well,  df_reads=df_reads, df_reads_amb=df_reads_amb, df_cell_genotype=df_cell_genotype)

    t_end=datetime.now()
    print('Time:',t_end-t_start, ' Start:', t_start, ' Finish:', t_end)
