import argparse
import json
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import importlib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import os

parser = argparse.ArgumentParser(
    prog='TrainModel', 
    description='Trains an image classifier model.'
)
parser.add_argument('data_dir')
parser.add_argument('--save_dir', default='.')
parser.add_argument('--arch', default='vgg11')
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--hidden_units', type=int, default=4096)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()

print(args.epochs)


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'