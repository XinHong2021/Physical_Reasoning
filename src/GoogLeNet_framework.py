import os
import pandas as pd
import numpy as np
from datetime import datetime as datatime
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torchvision import transforms
from torchvision import models

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix,classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

from tqdm import tqdm

class SimpleFineTunedGoogLeNet(nn.Module):
    def __init__(self):
        super(SimpleFineTunedGoogLeNet, self).__init__()

        self.googlenet = models.googlenet(weights = models.GoogLeNet_Weights.DEFAULT)

        num_ftrs = self.googlenet.fc.in_features
        self.googlenet.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.50),
            nn.Linear(128, 3)
        )
    def forward(self, inputs):
      inputs = self.googlenet(inputs)
      inputs = self.fc(inputs)
      return inputs

class ComplexFineTunedGoogLeNet(nn.Module):
    def __init__(self):
        super(ComplexFineTunedGoogLeNet, self).__init__()
        # load the pre-trained model: gogglenet
        self.googlenet = models.googlenet(weights = models.GoogLeNet_Weights.DEFAULT)

        num_params = self.googlenet.fc.in_features
        self.googlenet.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(num_params, 256),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 6)
        )
    def forward(self, inputs):
      inputs = self.googlenet(inputs)
      inputs = self.fc(inputs)
      return inputs