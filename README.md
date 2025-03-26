# Shalem Lab In-Situ Sequencing Analysis Boilerplate

This repository contains the functions and scripts for analyzing optical
screens.

## Installation

Certain packages are required for running the Python code. Please
install conda and then run the following commands to install the
required packages.

```{bash}
conda create -n iss python=3.8.10
conda activate iss
pip install torch==1.10.1
pip install cellpose==3.0.8
conda install pandas=1.5.2
conda install matplotlib
conda install scikit-image
conda install seaborn
pip install nd2reader
pip install pims_nd2
pip install csbdeep
pip install argparse
pip install decorator

# Packages necessary for running the environment in Jupyter notebooks
pip install ipykernel
conda install ipywidgets
python -m ipykernel install --user --name iss --display-name iss
```

## Workflow

The general workflow for running the analysis is as follows:

* Segment the images at both 40X and 10X (only need one cycle from the
genotyping)
* Genotype the cells
* QC the genotypes
* Create a layout matrix mapping each tile to their physical postions on
the well
* Manually map fiducial cells to calculate and optimize the
transformation between the 40X and 10X images
* Phenotype the cells
* Create images of the single cells
* Generate albums
* Concatenate image

More detailed instructions follows:

### Input file placement

You will want to place the images in the following folders within this
project directory. To avoid unnecessary moving or copying, you can
create links to your images using
`ln -s <original genotyping directory> genotyping`.
If you would like to load in example data, you can run
`ln -s /mnt/isilon/shalemlab/Tutorials/Demo_In-Situ-Seq_Analysis/Data/* .`

```
.
├── genotyping
│   ├── cycle_1 # 10X images
│   │   ├── ...
│   │   └── Well<well>_Point1_<tile>_ChannelDAPI,G-ISS,T-ISS,A-ISS,C-ISS,mClov3_Seq<seq>.nd2
│   ├── ...
│   └── cycle_9
│       └── ...
├── phenotyping # 40X images
│   ├── ...
│   └── Well<well>_Point1_<tile>_ChannelDAPI,mClov3,TMR_Seq<seq>.nd2
└── RBP_F2_Bulk_Optimized_Final.csv
```

### Activate environment

```{bash}
conda activate iss
```

### Segment cells

First, we will segment the cells in both the 40X and the 10X images. By
default, the scripts will segment images found in the following
directories:  

* 10X magnification: `genotyping/cycle_1`
* 40X magnification: `phenotyping`

You can change the path to the images by setting the `path_name`
variable in the `Segment_{10,40}X.py` scripts. 

```{bash}
# Run the following command to segment the images from genotyping
# sbatch --mem 16G -c 1 -t 1:00:00 -a <well><tile> Segment_10X.sh
# Example:
# Segment genotyping images from well 1, tiles 34, 35, 49 and 50
sbatch --mem 16G -c 1 -t 1:00:00 -a [10034,10035,10049,10050] Segment_10X.sh

# Run the following command to segment the images from phenotyping
# sbatch --mem 16G -c 1 -t 1:00:00 -a <well><tile> Segment_40X.sh
# Example:
# Segment genotyping images from well 1, tiles 495 to 504, 560 to 569,
# 619 to 628, 688 to 697, 751 to 760, 824 to 833, 890 to 899, 965 to
# 974, 1034 to 1043
sbatch --mem 16G -c 1 -t 1:00:00 -a [10495-10504,10560-10569,10619-10628,10688-10697,10751-10760,10824-10833,10890-10899,10965-10974,11034-11043] Segment_40X.sh
```

This generates files describing the segmented nuclei and cells masks here:
`segmented/{10X,40X}/{nucs,cells}/well_<well>/Seg_Nuc-Well_<well>_Tile_<tile>.npy`.
You can change this with the `save_name_nuc` and `save_name_cell`
variables in the `Segment_{10,40}X.py` scripts.

### Genotype cells

Next, we will run the genotyping algorithm on the 10X magnification
images. First, you will need to provide a CSV with information on the
library and the wells. This file will be read into the genotyping
script, which is set by the `df_lib` variable in
`Genotyping_Pipeline.py`. Additionally, you will need to edit lines
28-33 of `Genotyping_Pipeline.py` to map each well to the corresponding
group within the library.  

This script requires that the segmentation for the 10X images has been
completed.

```{bash}
# Run the following command to genotype the 10X magnification images
# sbatch --mem 16G -c 1 -t 1:00:00 -a <well><tile> Genotype_Pipeline.sh
# Example:
# Genotype cells from well 1, tiles 34, 35, 49 and 50
sbatch --mem 16G -c 1 -t 1:00:00 -a [10034,10035,10049,10050] Genotype_Pipeline.sh
```

This generates genotypes here by default:
`genotyping_results/well<well>/`. You can change the path using the
`save_path` variable in the `Genotyping_Pipeline.py` script.

### QC genotypes

Next, we will QC and filter the genotyped cells. We will need to set the
following variables on lines 18 through 28 of `Overall_QC.sh` to match
the previous output paths.  

* If you have not changed the default output paths, these following
variables do not need to be changed
    * `save_path`:
        * Default: `"genotyping_results/well_"`
        * Prefix for the path to the genotyping output, as set by
        `save_path` in `Genotyping_Pipeline.py`
    * `nuc_save_path`:
        * Default: `"segmented/10X/nucs/well_"`
        * Prefix for the path to the 10X magnification segmentation, as
        set by `save_path_nuc` in `Segment_10X.py`
    * `cell_save_path`:
        * Default: `"segmented/10X/cells/well_"`
        * Prefix for the path to the 10X magnification segmentation, as
        set by `save_path_cell` in `Segment_10X.py`
* The following variables/lines need to be changed to match the
information on the library, which should match what was given to the
`Genotyping_Pipeline.py` script
    * `df_lib`:
        * Path to the CSV containing information on the library
        * Should match the `df_lib` variable set in
        `Genotyping_Pipeline.py`
    * Lines 23 through 28
        * Map the well to the group within the CSV

```{bash}
# Run the following command to QC the genotyped cells
# sbatch --mem 16G -c 1 -t 1:00:00 -a <well> Overall_QC.sh
# Example:
# QC the genotyped cells from well 1
sbatch --mem 16G -c 1 -t 1:00:00 -a 1 Overall_QC.sh
```

This generates the following files

* `Cell_Genotype.csv`
* `Cell_Genotyped_Complete_Well_<well>.csv`
* `QC_Results.csv`
* `QC_Report.txt`
* `Quality_Scores.png`
* `sgRNAs_by_Rank.png`
* `sgRNAs_in_Gene.png`
* `sgRNAs_in_Intron.png`
* `Introns_in_Gene.png`

These will be output into `genotyping_results/well_<well>` by default.
This is the same location as the input from the genotyping pipeline, as
set by the `save_path` in `Overall_QC.py`.

### Concatenate genotyping results

We will concatenate the genotyping results. This has been done through
the Jupyter Notebook `Concatenate.ipynb` by running blocks 1 and 2, with
the `df_concat.to_csv('Cells_Genotyped.csv', index=False)` line
uncommented. Alternatively, run the following command to start a job on
SLURM.

```{bash}
# Run the following command to concatenate the QC'd genotypes into one
# CSV
srun --mem 16G -c 8 -t 1:00:00 Concatenate_Genotypes.sh
```

The output will be a CSV file `Cells_Genotyped.csv` with results from
all wells all together.

### Generate matrix of tile positions

This step requires interactive use of the `Mapping_1_Cal_M_Matrix.ipynb`
Jupyter notebook for each well. The goal here is to use the metadata
from the ND2 image files to create a map of the physical locations of
each tile within the well. By default, the map will be generated from
the images in the `genotyping/cycle_1` and `phenotyping` directories for
10X and 40X magnification respectively. You can change this by setting
the `path_10X` and `path_40X` variables.  

In addition to mapping the tile positions, this notebook will help you
see if the tiles are rotated or flipped. This is dependent on the user
profile settings on the microscope, and will need to be accounted for
when mapping cells to their physical locations. Plot a 2x2 square of
tiles and transform the image arrays to ensure the tiles are properly
aligned.  

The output from this notebook is:

* `M_10X.npy`: a numpy array with the physical tile layout for the 10X
magnification images
* `M_40X.npy`: a numpy array with the physical tile layout for the 40X
magnification images
* `M_10X.csv`: a CSV with the physical tile layout for the 10X
magnification images
* `M_40X.csv`: a CSV with the physical tile layout for the 40X
magnification images

### Finding fiducials and mapping coordinates between 10X and 40X

This step requires interactive use of the
`Mapping_2_Find_Fiducials.ipynb` and
`Mapping_3_Optimize_Mapping_DOF.ipynb` Jupyter notebooks for each well.
The goal here is to generate a list of coordinates in both the 40X and
10X magnification for at least 3 cells (5 recommended). Edit the
variable `p_10X` to add to the list of coordinates within the 10X
magnification space. Points are in the form `(<tile>, <i>, <j>)`. Once 3
points are provided, an affine transformation function will be optimized
for converting between the two coordinate systems. This function is
described by 5 parameters, which are output as the variable `DOF`.

### Phenotyping cells

We will phenotype the cells using the images at 40X magnification. The
script `Phenotype_Cells.py` will be run for each cell. As input, we need:

* `DOF`:
    * The optimized `DOF` from the previous mapping step
    * Set on line 41
* If you have not changed the default output paths, these following
variables do not need to be changed
    * `df_name`:
        * Default: `"Cells_Genotyped.csv"`
        * Filename for the concatenated genotyping output
    * `df_path`:
        * Default: `""`
        * Path to the directory containing concatenated genotyping output
    * `path_40X`:
        * Default: `""`
        * Path to the directory containing the `phenotyping` directory
        containing the 40X magnification images and the `segmented`
        directory with the cell and nuclei segmentation

```{bash}
# Run the following command to phenotype the cells at 40X magnification
# in batches of 100 cells
# sbatch --mem 16G -c 1 -t 1:00:00 -a <batch number> Overall_QC.sh
# Example:
# Phenotype cells 1 through 3201 in batches of 100 cells from the
# Cells_Genotyped.csv file
sbatch --mem 16G -c 1 -t 1:00:00 -a 0-31 Overall_QC.sh
```

This script outputs CSV files with the phenotyping results, one file for
each batch. The files are named as such:
`phenotyping_results/Cells_Phenotyped_<batch>.csv`.

### Concatenate phenotyping results

We will concatenate the phenotyping results. This has been done through
the Jupyter Notebook `Concatenate.ipynb` by running blocks 1 and 3, with
the `df_new.to_csv('Cells_Phenotyped.csv', index=False)` line
uncommented. Alternatively, run the following command to start a job on
SLURM.

```{bash}
# Run the following command to concatenate the phenotypes into one CSV
srun --mem 16G -c 8 -t 1:00:00 Concatenate_Phenotypes.sh
```

The output will be a CSV file `Cells_Phenotyped.csv` with results from
all wells all together.

### Making single cell images

We will generate single cell images from the images at 40X
magnification. The script `Make_Single_Cells.py` will be run for each
cell. As input, we need:

* `path_name`:
    * Default: `""`
    * Path to the directory containing concatenated phenotyping output,
    as well as the directories containing the segmentation
    (`segmented/40X`) and the images at 40X magnification
    (`phenotyping`)
* `path_df`:
    * Default: `"Cells_Phenotyped.csv"`
    * Filename for the concatenated phenotyping output

```{bash}
# Run the following command to generate single cell images at 40X
# magnification in batches of 100 cells
# sbatch --mem 16G -c 1 -t 1:00:00 -a <batch number> Make_Single_Cells.sh
# Example:
# Phenotype cells 1 through 2401 in batches of 100 cells from the
# Cells_Genotyped.csv file
sbatch --mem 16G -c 1 -t 1:00:00 -a 0-23 Make_Single_Cells.sh
```

This script outputs single cell images at the following path:
`datasets/single_cells/<sgRNA>_W-<well>_T<tile>_C-<cell>`. Additionally,
a CSV with the image paths is output for each batch at the following
path:
`datasets/single_cell_dataframes/single_cell_dataframes_<batch>.csv`.

### Concatenate single cells

We will concatenate the phenotyping results. This has been done through
the Jupyter Notebook `Concatenate.ipynb` by running blocks 1 and 4.
Alternatively, run the following command to start a job on SLURM.

```{bash}
# Run the following command to concatenate the QC'd genotypes into one
# CSV
srun --mem 16G -c 8 -t 1:00:00 Concatenate_Images.sh
```

The output will be a CSV file `Cells_Imaged.csv` with results from
all wells all together.

### Making albums

This step requires interactive use of the `Make_Albums.ipynb`
Jupyter notebook for each well. The output from this notebook
is:

* Embedded images of the log mean single cell image intensity
* `albums/<sgRNA>.png`
    * A grid of images of cells for a particular sgRNA, limited to a
    range of +/- 1 standard deviation from the weighted mean of means
    of a Gaussian mixture model fit to the log intensity

