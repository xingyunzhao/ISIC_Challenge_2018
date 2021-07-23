# imports
import pathlib

import albumentations
import numpy as np
import pandas as pd
import torch
# from neptunecontrib.api import log_table
from skimage.transform import resize
from torch.utils.data import DataLoader

from customdatasets import SegmentationDataSet4
from transformations import ComposeDouble, AlbuSeg2d, FunctionWrapperDouble, create_dense_target, normalize_01


# parameters
checkpoint_location = r'temp_chkp/'  # where checkpoints are saved to
# project_name = '<username>/<project>'  # the project has to be created beforehand in neptune!
# api_key = 'abcdetoken...'  # enter your api key from netpune here


# hyper-parameters
params = {'BATCH_SIZE': 8,
          'DEPTH': 4,
          'ACTIVATION': 'relu',
          'NORMALIZATION': 'group8',
          'UPSAMPLING': 'transposed',
          'LR': 0.0001,
          'WEIGTH_CE': torch.tensor((0.2, 0.8)),
          'WEIGTH_DICE': torch.tensor((0.0, 1.0)),
          'PRECISION': 32,
          'LR_FINDER': False,
          'INPUT_SIZE': (128, 128),
          'CLASSES': 2,
          'SEED': 42,
          'EXPERIMENT': 'carvana',
          'MAXEPOCHS': 10}


# root directory of data
root = pathlib.Path.cwd() / "Data" / "2018"


# function to get file paths
def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames


# input and target files
inputs = get_filenames_of_path(root / 'ISIC2018_Task1-2_Training_Input', ext="*.jpg")
targets = get_filenames_of_path(root / 'ISIC2018_Task1_Training_GroundTruth', ext="*.png")

inputs.sort()
targets.sort()




