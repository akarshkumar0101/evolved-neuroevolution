import numpy as np
import xarray as xr
import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import os
import shutil
from functools import partial

from torchinfo import summary
# import torch.utils.tensorboard as tb

import models_pheno, models_decode, models_breed, models_mutate
import mnist
import ga, neuroevolution, co_neuroevolution, cmaes
import genotype
# import ordinary_ne
import util, viz
from multi_arr import MultiArr

import functions
from functions import *
import xarray as xr
import argparse
from ga import calc_npop_truncate
import optim
from viz import *
from analysis import *

def do_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
do_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
