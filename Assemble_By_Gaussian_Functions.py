import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import scipy.ndimage as ndi
import scipy.signal as ss
import skimage.filters as sf
from skimage.morphology import disk
from sklearn import mixture
import itertools
import os, sys


def cluster_by_gaussian(_X, n_cluster=2, seed=None, verbose=False):

    # Decouple a 1D numpy array of values into gaussians peaks
    # _X: 1D numpy array of values
    # return: 1D numpy array of integer labels

    _X = np.expand_dims(_X, -1)

    if seed != 'None':
        np.random.seed(seed)

    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, n_cluster + 1)
    cv_types = ["spherical", "tied", "diag", "full"]
    for cv_type in cv_types:
        for n_components in n_components_range:
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
            gmm.fit(_X)
            bic.append(gmm.bic(_X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    color_iter = itertools.cycle(['black', 'purple', 'blue', 'green', 'gold', 'orange', 'red', 'pink', 'brown', 'grey'])
    # color_iter = itertools.cycle(['Greys_r', 'Purples_r', 'Blues_r', 'Greens_r', 'Oranges_r', 'Reds_r'])
    clf = best_gmm

    _Y = clf.predict(_X)

    if verbose:
        for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_, color_iter)):
            if not np.any(_Y == i):
                continue
            # plot_2D_hist(_X[_Y == i, 0], _X[_Y == i, 1], colormap=color)
            plt.scatter(_X[_Y == i, 0], _X[_Y == i, 1], s=0.2, color=color)

    return _Y


def Find_Intensity_Peaks(_z, bins=50, normalized_height=0.2, filter_width_coef=2, pad_left=True, pad_right=True,
                         peak_choice=-1, verbose=False):

    # Function to detect log mean single cell image intensity
    # Useful in extracting single cells of a certain sgRNA close the mean of the log normal distribution
    # Cells close to the high or low end of the distribution tend to exhibit anomalous protein localization pattern,
    # This function creates a historgram of the mean log intensities, and uses a smoothing gaussian filter to detect peaks.
    # based on the number of peaks, it then uses gaussian mixture to fit gaussian curves to the data.
    # This enable the function to handle single gaussian distributions, as well as multimodal distributions.
    # In the case of multimodal distributions, the function will focus on the highest intensity gaussian.
    # Once a gaussian is identified, are range will be caluclated of +/- one standard deviation to select cells from.

    # _z: 1D numpy array of the mean log single cell image intensities
    # bins: number of bins used to create the histrogram
    # normalized_height: when identifying peaks after hieght normalization, below this value peaks will be ignored.
    # designed to eliminate confusion between low frequency peaks and noise.
    # filter_width_coef: width of gaussian filter
    # pad_left: add zero value class at the lower end of the histogram, helps stabilize filter
    # pad_right: add zero value class at the higher end of the histogram, helps stabilize filter
    # peak_choice: In case of a multimodal distribution, and direct towards any peaks of interest.
    # By default, in a multimodal distribution, the the highest intensity gaussian peak will be chosen.
    # verbose: Plot the gaussian fitting process
    # return: range of +/- one std of log intensities to choose cells from, coordinates of identified peaks

    # Create histogram
    _h, _b = np.histogram(_z, bins=bins)
    _h = _h / _h.max()
    _db = _b[1] - _b[0]

    if pad_left:
        _b = np.concatenate(([_b[0] - _db], _b))
        _h = np.concatenate(([0], _h))

    if pad_right:
        _b = np.concatenate((_b, [_b[-1] + _db]))
        _h = np.concatenate((_h, [0]))

    _b = _b[:-1] / 2 + _b[1:] / 2

    # Perform gaussian smoothing
    _hs = ndi.gaussian_filter(_h, sigma=filter_width_coef * (_b.max() - _b.min()))

    # Identify peaks
    _peaks, _ = ss.find_peaks(_hs, height=normalized_height)

    # Address possibility of multimodal distribution
    if len(_peaks) > 1:
        _Y = cluster_by_gaussian(_z, n_cluster=len(_peaks))
        _labels = np.unique(_Y)

        _means = np.zeros([len(_labels), 2])
        for i in _labels:
            _means[i] = [i, np.mean(_z[_Y == i])]

        df_ints = pd.DataFrame(_means, columns=['group', 'mean']).sort_values(by='mean')

        _z = _z[_Y == df_ints['group'].iloc[peak_choice]]

    # Calculate range of +/- one std around gaussian mean of  peak of interest
    _h_, _b_ = np.histogram(_z, bins=bins)
    _b_ = _b_[:-1] / 2 + _b_[1:] / 2
    _h_ = _h_ / _h_.max()
    u_ = np.sum(_b_ * _h_) / np.sum(_h_)
    n_ = len(_h_[_h_ > 0])
    m_ = ((n_ - 1) / n_)
    s_ = np.sqrt(np.sum(_h_ * (u_ - _b_) ** 2) / (m_ * np.sum(_h_)))
    r_min_ = u_ - s_
    r_max_ = u_ + s_

    if verbose:
        plt.plot(_b, _h, '.-', c='grey')
        plt.plot(_b_, _h_, '.-', c='black')
        plt.plot(_b, _hs, c='red')
        plt.plot(_b[_peaks], _hs[_peaks], 'x', c='green')
        plt.plot([r_min_, r_min_], [_h.min(), _h.max()], c='red')
        plt.plot([r_max_, r_max_], [_h.min(), _h.max()], c='red')

        plt.show()

    return np.array([r_min_, r_max_]), _b[_peaks]


def Make_Album_50(_df, _channel, _range_log10, _path_name):

    # Make a 5 x 10 album of single cell images
    # _df: dataframe containing cell information including coordinates and image name path
    # _channel: imaging channel of interest for the albums
    # _range_log10: 1D two element array of style: [min, max],
    # such that the minimum average single cell image intensity is 10**min, and maximum intensity is 10**max
    # _path_name: path where single cell images png files are stored

    _df_album = _df.copy()
    _df_album = _df_album[_df_album[_channel] >= 10 ** _range_log10[0]]
    _df_album = _df_album[_df_album[_channel] <= 10 ** _range_log10[1]]
    print('Cells:', len(_df_album))

    if len(_df_album) > 50:
        _df_album = _df_album.sample(n=50, random_state=2023)

    for _c, _file in enumerate(_df_album['image']):
        _img = np.moveaxis(np.array(Image.open(os.path.join(_path_name, _file)).resize((128, 128)), dtype=float), -1, 0)
        _mask = _img[2] > 0
        # mask = Erode_Cell(mask, deg=6)
        _img_final = _img[0] * _mask
        plt.subplot(5, 10, _c + 1)
        plt.imshow(_img_final, cmap='Greys_r')
        plt.axis('off')