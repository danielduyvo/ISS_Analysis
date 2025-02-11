from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import nd2reader as nd2
from cellpose import models as cellpose_models
import skimage.measure as sm
from os import listdir
from os.path import isfile, join
import In_Situ_Functions as isf
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    t_start = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=int)
    args = parser.parse_args()
    n_job = args.job

    n_well = int(np.floor(n_job / 10000))
    n_tile = int(np.around(10000 * ((n_job / 10000) % 1)))

    path_name = 'genotyping/cycle_1'
    img = isf.InSitu.Import_ND2_by_Tile_and_Well(n_tile, n_well, path_name)

    nucs = isf.Segment.Segment_Nuclei(img[0], nuc_diameter=30)
    cells = isf.Segment.Segment_Cells(img[-1], NUC=img[0], cell_diameter=60)
    nucs, cells = isf.Segment.Label_and_Clean(nucs, cells)

    save_name_nuc = 'segmented/10X/nucs/well_' + str(n_well) + '/Seg_Nuc-Well_' + str(n_well) + '_Tile_' + str(n_tile) + '.npy'
    save_name_cell = 'segmented/10X/cells/well_' + str(n_well) + '/Seg_Cells-Well_' + str(n_well) + '_Tile_' + str(n_tile) + '.npy'
    np.save(save_name_nuc, nucs)
    np.save(save_name_cell, cells)

    t_end = datetime.now()

    print('Well: ' + str(n_well) + ' Tile: ' + str(n_tile) + ' Time Start: ' + str(t_start), ' Time End: ' + str(t_end) + ' Duration: ' + str(t_end - t_start))


