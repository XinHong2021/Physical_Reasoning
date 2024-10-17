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

class ReadData(Dataset):

    def __init__(self, df, img_dir,label_column, transform = None):
        """
        Initialize the dataset
        Args:
        df: pandas dataframe, the data of the dataset
        img_dir: string, the directory of the images
        transform: torchvision.transforms, the transform to apply to the images
        """
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.label_column = label_column
        img_dir_path = Path(self.img_dir)

        if not img_dir_path.exists():
            raise ValueError(f"The directory {img_dir_path} does not exist!")
        if not img_dir_path.is_dir() or not os.access(img_dir_path, os.R_OK):
            raise ValueError(f"The directory {img_dir_path} is not accessible or readable!")
        
    """
    Get the length of data set
    """
    def __len__(self):
        return len(self.df)

    """
    Get the image and label of the data set
    Args:
        index: int, the index of the image
    Returns:  
        image: PIL image, the image of the data set
        label: int, the label of the data set
    """
    def __getitem__(self, index):
        img_name = os.path.join(self.img_dir, str(self.df.iloc[index, 0])) 
        image = Image.open(img_name + ".jpg")

        if self.label_column == 'instability_type':
            label = self.df.iloc[index,4] 
    
        elif self.label_column == 'stable_height': 
            label = self.df.iloc[index,-1] 
        
        elif self.label_column == 'total_height':
            label = self.df.iloc[index, 3]
            
        else:
            try:
                print(self.label_column)
            except:
                print("Not found label column")
            
        if self.transform:
            image = self.transform(image)

        return image, label 

class GetTestDF(Dataset):

    def __init__(self, df, img_dir, transform = None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    '''
    Return the size of the dataset
    '''
    def __len__(self):
        return len(self.df)

    '''
    get the image and related column
    '''
    def __getitem__(self, index):
        img_name = os.path.join(self.img_dir, str(self.df.iloc[index, 0]))
        image = Image.open(img_name + ".jpg")
        if self.transform:
            image = self.transform(image)
        
        return image, self.df.iloc[index, 0]


## create model logs 
def create_model_dir(run_name):
    
    base_path = f'./models/'
    full_path = f'{base_path}_{run_name}'

    if not os.path.exists(full_path):
        os.makedirs(full_path)
        
    counter = 1
    while os.path.exists(full_path):
        full_path = f'{full_path}_{counter}'
        counter += 1
    os.makedirs(full_path)

    return full_path
