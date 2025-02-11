import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import In_Situ_Functions as isf
import Album_Functions as af
import matplotlib
import argparse
matplotlib.use('Agg')

if __name__ == '__main__':

    # Split single cell image generation into batches to parallelize computations,
    # Indicated number of batch in shell script
    parser = argparse.ArgumentParser()
    parser.add_argument('batch', type=int)
    args = parser.parse_args()
    B = args.batch

    # Indicate how many cells per batch
    batch_length = 100

    # Load the phenotyped cells dataframe
    path_name = ''
    path_df = 'Cells_Phenotyped.csv'
    save_path = 'datasets/single_cells'
    seg_path = os.path.join(path_name, 'segmented/40X')
    path_40X = os.path.join(path_name, 'phenotyping')

    df = pd.read_csv(os.path.join(path_name, path_df))
    df_sub = df.iloc[B * batch_length: (B + 1) * batch_length]

    # Iterate by row, every row contains a single cell information
    image_list = -1*np.ones([len(df_sub)], dtype=object)
    for i in tqdm(range(len(df_sub))):

        # extract cell coordinates
        n_well = int(df_sub['well'].iloc[i])
        T_40X = int(df_sub['tile_40X'].iloc[i])
        I_40X = int(df_sub['i_nuc_40X'].iloc[i])
        J_40X = int(df_sub['j_nuc_40X'].iloc[i])
        sgRNA = df_sub['sgRNA'].iloc[i]
        Cell = int(df_sub['cell'].iloc[i])
        T_10X = int(df_sub['tile'].iloc[i])

        cell_name = sgRNA + '_W-' + str(n_well) + '_T-' + str(T_10X) + '_C-' + str(Cell)

        try:

            # load cellular and nuclear tile masks
            cell_mask_path = os.path.join(seg_path, 'cells/well_' + str(n_well))
            nuc_mask_path = os.path.join(seg_path, 'nucs/well_' + str(n_well))
            nucs_mask = np.load(os.path.join(nuc_mask_path, 'Seg_Nuc-Well_' + str(n_well) + '_Tile_' + str(T_40X) + '.npy'))
            cells_mask = np.load(os.path.join(cell_mask_path, 'Seg_Cells-Well_' + str(n_well) + '_Tile_' + str(T_40X) + '.npy'))

            # load image tile
            img_40X = isf.InSitu.Import_ND2_by_Tile_and_Well(T_40X, n_well, path_40X)

            # crop cells and combine with masks
            img_cell, cells_mask_crop, nucs_mask_crop = af.Make_Cell_Image_From_CSV(I_40X, J_40X, img_40X, cells_mask[1], nuc_mask=nucs_mask[1], padding=12, crop_background=False)

            # Create cell image by combining phenotyping image with cellular and nuclear masks
            if img_cell.any() != None:

                # cell_name = sgRNA + '_W-' + str(n_well) + '_T-' + str(T_10X) + '_C-' + str(Cell)
                image_list[i] = cell_name + '.png'

                _, h, w = img_cell.shape

                img_cell_final = np.zeros([h, w, 4])
                img_cell_final[:, :, 0] = af.Norm(img_cell[2]) # Red channel: RBP Halotag (JFX549)
                img_cell_final[:, :, 1] = af.Norm(img_cell[1]) # Green channel: G3BP1/cellular dye (mClov3)
                img_cell_final[:, :, 2] = af.Norm(img_cell[0]) # Blue channel: Nuclear stain (DAPI)
                # nuclear mask and cellular mask ar combined into the alpha channel
                # in png 8 bit image is produced, so that 0.8*255 = 204 is the nuclear mask, img[:, :, 3] > 200 extracts nuclear mask
                # in png 8 bit image is produced, so that 0.7*255 = 178 is the nuclear mask, img[:, :, 3] > 150 extracts cellular mask
                img_cell_final[:, :, 3] = 0.1 * nucs_mask_crop + 0.2 * cells_mask_crop + 0.5

                # Save cell image
                af.Save_Cell_Image(img_cell_final, cell_name, save_path)

        except:

            print(cell_name, 'Cannot be found')

    # save dataframe batch
    df_sub['image'] = image_list
    df_sub.to_csv('datasets/single_cell_dataframes/single_cell_dataframes_' + str(B) + '.csv', index=False)

