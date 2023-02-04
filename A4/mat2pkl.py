"""This script converts .mat into .pkl
Notes:
- some .mat files may require "import mat73.loadmat"
- set MAT_FOLDER to the path to the .mat files
- set PKL_FOLDER to the path to save the .pkl files
- run "python mat2pkl.py" to run this script
"""

# IMPORTS
from scipy.io import loadmat
import pickle
from tqdm import tqdm
import os
from os import walk

##########################################################################
##########################################################################

# PATHS
MAT_FOLDER = '../data/'
PKL_FOLDER = './data/'
if not os.path.isdir(PKL_FOLDER):
    os.makedirs(PKL_FOLDER)

##########################################################################
##########################################################################

# if this script is run as a script rather than imported
if __name__ == "__main__": 

    # Load the filenames of the EEG datasets
    mats = []
    in_dirs = []
    for (dirpath, dirnames, filenames) in walk(MAT_FOLDER):
        filenames = [string for string in filenames if '.mat' in string]
        directories = [dirpath + '/' + string for string in filenames if '.mat' in string]
        if len(filenames) == 0:
            continue
        mats.extend(filenames)
        in_dirs.extend(directories)
    mats.sort()
    in_dirs.sort()

    out_dir = PKL_FOLDER
    for (path, file, i) in zip(in_dirs, mats, tqdm(range(len(mats)), desc='Converting .mat to .pkl...')):
        try:
            data_dict = loadmat(path)
            output_file = open(out_dir + file[:-4] + '.pkl', "wb")
            pickle.dump(data_dict, output_file)
            output_file.close()
        except:
            print(file + ' cannot be loaded') # there are a few files that just can't be loaded
            continue