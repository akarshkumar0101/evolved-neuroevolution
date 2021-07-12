import numpy as np
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

np.random.seed(0)
torch.manual_seed(10);
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
