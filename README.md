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

We will want to segment the cells in both the 40X and the 10X images. 

```
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
