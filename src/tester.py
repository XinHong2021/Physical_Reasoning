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

from src.read_data import GetTestDF




class TesterTemplate:
    def __init__(self, model_path, model, label_column, 
                csv_path, img_dir, 
                batch_size=32):

        self.model_path = model_path
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.label_column = label_column
        self.img_dir = img_dir
        self.data_frame = pd.read_csv(csv_path)
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.test_loader = self.generate_dataloader(self.transform,self.data_frame, shuffle=False)

    def generate_dataloader(self, transform, data_frame, shuffle=True):
      #  print("label column type", self.label_column)
        dataset = GetTestDF(data_frame, self.img_dir, transform=transform) 
        testloader= DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return testloader
        
    def load_model(self):
      #  self.model.load_state_dict(torch.load(self.model_path, map_location=self.device,weights_only=True))
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device,weights_only=True))

        self.model.eval()
        return self.model
    
    def predict(self):

        image_ids_all = []
        predictions_all = []

        with torch.no_grad():

            for images, image_ids in tqdm(self.test_loader, desc="testing"):
                images = images.to(self.device)
                outputs = self.model(images)
                #outputs = torch.round(outputs).clamp(1,6) #在1-6之间输出
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                #predictions = outputs.squeeze().cpu().numpy()
                predictions = predictions.astype(int) + 1
                #predictions = predictions.astype(int)

                predictions_all.extend(predictions)
                image_ids_all.extend(image_ids.numpy())

        pred_df = pd.DataFrame({
            'id': image_ids_all,
            'stable_height': predictions_all
             })
        return pred_df

    def save_to_path(self, output_csv):
      pred_df = self.predict()
      pred_df.to_csv(output_csv, index=False)
      print(f"Predictions have been saved to path: {output_csv}")