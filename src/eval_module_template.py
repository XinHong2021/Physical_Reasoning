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

from src.read_data import ReadData

class EvalTrainDf:
    def __init__(self, model_path, model, label_column, 
                csv_path, img_dir, 
                stratify_column='stable_height', 
                test_size=0.2,
                batch_size=32,
                random_state = 42):

        self.model_path = model_path
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.label_column = label_column
        
        self.img_dir = img_dir
        self.data_frame = pd.read_csv(csv_path)
        self.stratify_column = stratify_column
        self.test_size = test_size
        self.random_state = random_state
        self.batch_size = batch_size

        # preprocess images 
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # split data into train and validation 
        # Create data loader
        self.train_data, self.val_data = self.split_train_valid()
        self.train_loader = self.generate_dataloader(self.transform, self.train_data)
        self.valid_loader = self.generate_dataloader(self.transform, self.val_data,  shuffle=False)
        
        self.model = self.load_model()

    def validate(self):
        self.model.eval()
        num_cor_pred = 0
        num_samples = 0
        gt_labels = []
        pred_labels = []
        pred_outputs = []
        with torch.no_grad():
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).long()
                outputs = self.model(inputs)
                pred_outputs.extend(outputs.cpu().numpy())
                pred_class = torch.argmax(outputs, 1)
                if self.label_column == 'stable_height' or self.label_column == 'total_height':
                    pred_class = pred_class + 1
          #      labels = labels - 1

                gt_labels.extend(labels.cpu().numpy())
                pred_labels.extend(pred_class.cpu().numpy())
                # calculate number of correct predictions
                num_cor_pred += (pred_class == labels).sum().item()
                num_samples += labels.size(0)
        # calculate the accuracy rate
        val_accuracy = num_cor_pred / num_samples
        self.pred_labels = pred_labels
        self.pred_outputs = pred_outputs

    def split_train_valid(self):
        train_data, valid_data = train_test_split(
                                    self.data_frame,
                                    test_size=self.test_size,
                                    random_state=self.random_state,
                                    stratify=self.data_frame[self.stratify_column]
                                )

        print(f"Train dataset size: {len(train_data)}",
                f"Validation dataset size: {len(valid_data)}")
        return train_data, valid_data


    def load_model(self):
      #  self.model.load_state_dict(torch.load(self.model_path, map_location=self.device,weights_only=True))
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device,weights_only=True))

        self.model.eval()
        return self.model


class ModelTemplate:
    def __init__(self, model_path, model, label_column, 
                csv_path, img_dir, 
                stratify_column='stable_height', 
                test_size=0.2,
                batch_size=32,
                random_state = 42):

        self.model_path = model_path
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.label_column = label_column
        
        self.img_dir = img_dir
        self.data_frame = pd.read_csv(csv_path)
        self.stratify_column = stratify_column
        self.test_size = test_size
        self.random_state = random_state
        self.batch_size = batch_size

        # preprocess images 
        self.transform = transforms.Compose([
            # 这个变换将图像从PIL图像或NumPy数组转换为PyTorch张量。转换过程中，图像的像素值会被归一化到[0, 1]区间，
            # 并且通道顺序会被调整为(C, H, W)，即通道数在前，高度和宽度在后。
            transforms.ToTensor(),
            # 这个变换对图像进行标准化。标准化操作是将图像的每个通道减去该通道的均值，然后除以该通道的标准差。
            # 这里的均值和标准差是预训练的模型
            # 在ImageNet数据集上计算得到的，用于确保图像的分布与模型训练时的分布一致，从而提高模型的泛化能力。
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # split data into train and validation 
        # Create data loader
        self.train_data, self.val_data = self.split_train_valid()
        self.train_loader = self.generate_dataloader(self.transform, self.train_data)
        self.valid_loader = self.generate_dataloader(self.transform, self.val_data,  shuffle=False)
        
        self.model = self.load_model()

    def validate(self):
        self.model.eval()
        num_cor_pred = 0
        num_samples = 0
        gt_labels = []
        pred_labels = []

        with torch.no_grad():
            for inputs, labels in self.valid_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).long()
                outputs = self.model(inputs)
                pred_class = torch.argmax(outputs, 1)
                if self.label_column == 'stable_height' or self.label_column == 'total_height':
                    pred_class = pred_class + 1
          #      labels = labels - 1

                gt_labels.extend(labels.cpu().numpy())
                pred_labels.extend(pred_class.cpu().numpy())
                # calculate number of correct predictions
                num_cor_pred += (pred_class == labels).sum().item()
                num_samples += labels.size(0)

        # calculate the accuracy rate
        val_accuracy = num_cor_pred / num_samples
        self.generate_classification_report(np.array(pred_labels), np.array(gt_labels))
        self.pred_labels = pred_labels
        return val_accuracy

    def generate_classification_report(self, outputs, labels):
        matrix = confusion_matrix(labels, outputs)
        print(classification_report(labels, outputs, zero_division=0))
        print(matrix)
    
    def split_train_valid(self):
        train_data, valid_data = train_test_split(
                                    self.data_frame,
                                    test_size=self.test_size,
                                    random_state=self.random_state,
                                    stratify=self.data_frame[self.stratify_column]
                                )

        print(f"Train dataset size: {len(train_data)}",
                f"Validation dataset size: {len(valid_data)}")
        return train_data, valid_data


    def load_model(self):
      #  self.model.load_state_dict(torch.load(self.model_path, map_location=self.device,weights_only=True))
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device,weights_only=True))

        self.model.eval()
        return self.model


    def generate_dataloader(self, transform, data_frame, shuffle=True):
      #  print("label column type", self.label_column)
        dataset = ReadData(data_frame, self.img_dir,self.label_column, transform=transform) 
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
