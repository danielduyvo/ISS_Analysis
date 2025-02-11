import numpy as np
import pandas as pd
import argparse
import os
import In_Situ_Functions as isf
import Mapping_Functions as mf
import Album_Functions as af
from tqdm import tqdm

if __name__ == '__main__':

    def Crop_Cell(_p_40X, _img, _cells):

        # Cut a box around a cell
        # _p_40X: numpy array of dimensions 1 x 3 of single cell coordinates, contains the following structure: [[tile, i, j]]
        # _img: image tile where the cell is imaged
        # _cells: segmented cell mask of the tile
        # returns: cropped cell image (Ch x H x W), and i min/max j min/max of the box that contains the cell

        _cell_number = _cells[_p_40X[0, 1], _p_40X[0, 2]]
        _single_cell_mask = _cells == _cell_number

        i_project = np.sum(_single_cell_mask, axis=1) > 0
        i_min = np.argmax(i_project)
        i_max = np.argmax(np.cumsum(i_project))

        j_project = np.sum(_single_cell_mask, axis=0) > 0
        j_min = np.argmax(j_project)
        j_max = np.argmax(np.cumsum(j_project))

        _cell_crop = np.array(_img[:, i_min:i_max, j_min:j_max] * _single_cell_mask[i_min:i_max, j_min:j_max], dtype=int)

        return _cell_crop, i_min, i_max, j_min, j_max

    def Get_DOF(_n_well):

        # Degrees of Freedom by well, to copied from the Mapping jupyter notebooks.
        # 5 floats required, [di, dj, angle, scale i, scale j] for low -> high mag affine transformation

        if _n_well == 1:
            _DOF = [621.232776, 667.464359, 0.0253159474, 4.00321261, 4.00259586]

        return _DOF

    def norm(_x):
        return (_x - _x.min()) / (_x.max() - _x.min())

    def Colocalize(_n_cell, _rgb, _cells, _nucs, mask='cell', ch_1=0, ch_2=1):

        # Preprocessing for R2 colocalization score calculation, flatten and clean image pixels of zero value pixels
        # _n_cell: the cell number from the Genotyped dataframe
        # _rgb: image of box-croped single cell image
        # _cells: segmented cell mask of the tile
        # _nucs: segmented nuclei mask of the tile
        # mask: determine which mask to use for the flattening and cleaning of single cell image
        # ch_1: the first channel involved in the colocalization score
        # ch_2: the second channel involved in the colocalization score
        # return: flat and clean channel 1, flat and clean channel 2, cell image

        find_cell = _cells == _n_cell
        find_nuc = _nucs == _n_cell
        x, y = np.where(find_cell)
        a, b, c, d, = af.Crop_Cell_Window(x[int(len(x) / 2)], y[int(len(y) / 2)], _cells)

        cells3 = np.zeros([b - a, d - c, 3])
        find_nuc_crop = af.Smooth_Cell(find_nuc[a:b, c:d])
        find_cell_crop = af.Smooth_Cell(find_cell[a:b, c:d])
        find_cyto_crop = (find_cell_crop + find_nuc_crop) == 1

        final_crop = np.ones([b - a, d - c])
        if mask == 'cell':
            final_crop = find_cell_crop
        if mask == 'cyto':
            final_crop = find_cyto_crop
        if mask == 'nuc':
            final_crop = find_nuc_crop

        cells3[:, :, 0] = final_crop
        cells3[:, :, 1] = final_crop
        cells3[:, :, 2] = final_crop

        _cell_img = _rgb[a:b, c:d, :] * cells3

        _flat_R = np.ravel(_cell_img[:, :, ch_1])
        _flat_G = np.ravel(_cell_img[:, :, ch_2])

        _flat_R = _flat_R[_flat_R > 0]
        _flat_G = _flat_G[_flat_G > 0]

        return _flat_G, _flat_R, _cell_img

    def get_r2(_X, _Y):

        # Calculated R2 correlation for a 2D scatter

        corr_matrix = np.corrcoef(_X, _Y)
        corr = corr_matrix[0, 1]
        r_squared = corr ** 2

        return r_squared

    # Split cell phenotyping into batches to parallelize computations,
    # Indicated number of batch in shell script
    parser = argparse.ArgumentParser()
    parser.add_argument('batch', type=int)
    args = parser.parse_args()
    B = args.batch

    # Indicate how many cells per batch
    batch_length = 100

    # Load tile map
    M_10X = np.load('M_10X.npy')
    M_40X = np.load('M_40X.npy')

    # Load genotyped cells list
    df_name = 'Cells_Genotyped.csv'
    df_path = ''
    full_path = os.path.join(df_path, df_name)
    path_40X = ''

    df_geno = pd.read_csv(full_path)
    df_geno = df_geno.iloc[B * batch_length : (B + 1) * batch_length]

    # Collection of phenotyping parameters to calculate, preparing empty data arrays

    N = len(df_geno)

    tile_40X = -1 * np.ones([N], dtype=int)
    i_nuc_40X = -1 * np.ones([N], dtype=int)
    j_nuc_40X = -1 * np.ones([N], dtype=int)

    area_nuc = -1 * np.ones([N], dtype=int)
    area_cell = -1 * np.ones([N], dtype=int)

    ints_tot_DAPI = -1 * np.ones([N], dtype=int)
    ints_tot_Clov = -1 * np.ones([N], dtype=int)
    ints_tot_Halo = -1 * np.ones([N], dtype=int)

    ints_avg_DAPI = -1 * np.ones([N], dtype=int)
    ints_avg_Clov = -1 * np.ones([N], dtype=int)
    ints_avg_Halo = -1 * np.ones([N], dtype=int)

    ints_std_DAPI = -1 * np.ones([N], dtype=int)
    ints_std_Clov = -1 * np.ones([N], dtype=int)
    ints_std_Halo = -1 * np.ones([N], dtype=int)

    i_min_list = -1 * np.ones([N], dtype=int)
    i_max_list = -1 * np.ones([N], dtype=int)
    j_min_list = -1 * np.ones([N], dtype=int)
    j_max_list = -1 * np.ones([N], dtype=int)

    coloc_list = -1 * np.ones([N])

    # Iterate through the cells in the genotyped cells dataframe batch
    for i in tqdm(range(len(df_geno))):

        n_well = int(df_geno['well'].iloc[i])
        T_10X = int(df_geno['tile'].iloc[i])
        I_10X = int(df_geno['i_nuc'].iloc[i])
        J_10X = int(df_geno['j_nuc'].iloc[i])

        DOF = Get_DOF(n_well)

        try:
            # Find position of cell in phenotyping high mag images, using low mag genotyping coordinates
            P_10X = mf.Local_to_Global(np.array([[T_10X, I_10X, J_10X]]), M_10X, [2304, 2304])
            P_40X = mf.model_TRS(P_10X, DOF, angle='degree')
            p_40X = mf.Global_to_Local(P_40X, M_40X, [2304, 2304])

            p_40X = np.squeeze(p_40X)
            T_40X = p_40X[0]
            I_40X = p_40X[1]
            J_40X = p_40X[2]

            tile_40X[i] = int(T_40X)
            i_nuc_40X[i] = int(I_40X)
            j_nuc_40X[i] = int(J_40X)

            if T_40X >= 0:
                # Find high mag image phenotyping tile containing cell
                img_40X = isf.InSitu.Import_ND2_by_Tile_and_Well(T_40X, n_well, os.path.join(path_40X, 'phenotyping'))

                # Find high mag segmented cell and nuclear tile containing cell
                cells = np.load(os.path.join(path_40X, 'segmented/40X/cells/well_' + str(n_well), 'Seg_Cells-Well_' + str(n_well) + '_Tile_' + str(T_40X) + '.npy'))
                nucs = np.load(os.path.join(path_40X, 'segmented/40X/nucs/well_' + str(n_well), 'Seg_Nuc-Well_' + str(n_well) + '_Tile_' + str(T_40X) + '.npy'))

                # create a box-crop of single cell from original phenotyping high mag image
                crop_nuc, _, _, _, _ = Crop_Cell(np.array([[T_40X, I_40X, J_40X]]), img_40X, nucs[1])
                crop, i_min, i_max, j_min, j_max = Crop_Cell(np.array([[T_40X, I_40X, J_40X]]), img_40X, cells[1])
                c, h, w = crop.shape

            # eliminate objects that are larger than 500 x 500, probably not cells
            if h < 500 and w < 500:
                # Perform phenotype calculations on DAPI (nuclear), mClov3 (G3BP1), JFX549 (Halotag)

                crop_DAPI = crop_nuc[0]
                crop_Clov = crop[1]
                crop_Halo = crop[2]

                flat_DAPI = np.ravel(crop_DAPI)
                flat_DAPI = flat_DAPI[flat_DAPI != 0]

                flat_Clov = np.ravel(crop_Clov)
                flat_Clov = flat_Clov[flat_Clov != 0]

                flat_Halo = np.ravel(crop_Halo)
                flat_Halo = flat_Halo[flat_Halo != 0]

                area_nuc[i] = len(flat_DAPI)
                area_cell[i] = len(flat_Clov)

                ints_tot_DAPI[i] = int(np.sum(flat_DAPI))
                ints_tot_Clov[i] = int(np.sum(flat_Clov))
                ints_tot_Halo[i] = int(np.sum(flat_Halo))

                ints_avg_DAPI[i] = int(np.mean(flat_DAPI))
                ints_avg_Clov[i] = int(np.mean(flat_Clov))
                ints_avg_Halo[i] = int(np.mean(flat_Halo))

                ints_std_DAPI[i] = int(np.std(flat_DAPI))
                ints_std_Clov[i] = int(np.std(flat_Clov))
                ints_std_Halo[i] = int(np.std(flat_Halo))

                i_min_list[i] = i_min
                i_max_list[i] = i_max
                j_min_list[i] = j_min
                j_max_list[i] = j_max

                rgb = np.zeros([2304, 2304, 3])
                rgb[:, :, 0] = norm(img_40X[2])
                rgb[:, :, 1] = norm(img_40X[1])

                n_cell = int(cells[1, I_40X, J_40X])
                G, R, I = Colocalize(n_cell, rgb, cells[1], nucs[1], mask='cyto')
                coloc_list[i] = get_r2(G, R)

        except:
            print('Cannot find specific cell. Well:', n_well, 'Tile:', T_10X)

    # construct dataframe with column names
    df_geno['tile_40X'] = tile_40X
    df_geno['i_nuc_40X'] = i_nuc_40X
    df_geno['j_nuc_40X'] = j_nuc_40X

    df_geno['area_nuc'] = area_nuc
    df_geno['area_cell'] = area_cell

    df_geno['ints_tot_DAPI'] = ints_tot_DAPI
    df_geno['ints_tot_Clov'] = ints_tot_Clov
    df_geno['ints_tot_Halo'] = ints_tot_Halo

    df_geno['ints_avg_DAPI'] = ints_avg_DAPI
    df_geno['ints_avg_Clov'] = ints_avg_Clov
    df_geno['ints_avg_Halo'] = ints_avg_Halo

    df_geno['ints_std_DAPI'] = ints_std_DAPI
    df_geno['ints_std_Clov'] = ints_std_Clov
    df_geno['ints_std_Halo'] = ints_std_Halo

    df_geno['i_min'] = i_min_list
    df_geno['i_max'] = i_max_list
    df_geno['j_min'] = j_min_list
    df_geno['j_max'] = j_max_list

    df_geno['coloc_r2'] = coloc_list

    file_name = 'Cells_Phenotyped'
    df_geno.to_csv('phenotyping_results/' + file_name + '_' + str(B) + '.csv', index=False)









