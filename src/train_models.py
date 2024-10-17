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
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix,classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from tqdm import tqdm


from src.read_data import ReadData
from src.read_data import create_model_dir

class BestPrecisionTrainer:
    def __init__(self, csv_path, img_dir, model, store_model_name, label_column, 
                    stratify_column='stable_height', 
                    test_size=0.2, batch_size=32, num_epochs=10, random_state=42,
                    learning_rate=0.001):
                  
        self.img_dir = img_dir
        self.stratify_column = stratify_column
        self.test_size = test_size
        self.random_state = random_state
        self.store_model_name = store_model_name
        self.label_column = label_column
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.data_frame = pd.read_csv(csv_path)
        
        self.transform = transforms.Compose([
            # transforms.Resize((512, 512)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), 
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
      
        # split data into train and validation 
        # Create data loader
        self.train_data, self.val_data = self.split_train_valid()
        self.train_loader = self.generate_dataloader(self.transform, self.train_data)
        self.valid_loader = self.generate_dataloader(self.transform, self.val_data,  shuffle=False)
    

        # set the loss weight
        '''
        Changed some thing here : to see the results
        1. add the class_weights to each class labels with different weights,
           more lables occurece with lower weight, few label occurence with higher weight;
           without the weights, all the labels will be treated equally.

        2. used to apply L2 regularization (also called weight decay).
        The primary purpose of weight decay is to prevent overfitting by penalizing large weights.
         It adds a penalty to the loss function based on the size of the weights
         helps it generalize better to unseen data.
         Lnew = Lold + weight_decay * sum(weight^2)

        3. add scheduler to Reduces the learning rate after every step_size epochs.
          After every 4 epochs, the learning rate will be multiplied by gamma (0.1 here), reducing it by 90%.
        '''
        class_weights = torch.tensor([20,10,5], device= self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights) # CrossEntropy for multi categorical-label predication
        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate , weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = 15, gamma=0.1)



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

    def generate_dataloader(self, transform, data_frame, shuffle=True):
        #  print("label column type", self.label_column)
        dataset = ReadData(data_frame, self.img_dir,self.label_column, transform=transform) 
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def generate_classification_report(self, outputs, labels):
        matrix = confusion_matrix(labels, outputs)
        print(classification_report(labels, outputs, zero_division=0))
        print(matrix)


    def calculate_class3_recall(self, outputs, labels):
        recall_all_class = recall_score(labels, outputs, average=None)
        recall_class3 = recall_all_class[2]
        return recall_class3
    
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        num_cor_pred = 0
        num_samples = 0
        gt_labels = [] 
        pred_labels = []

        with torch.no_grad():
            for inputs, labels in self.valid_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).long()
                labels = labels - 1
                outputs = self.model(inputs)
                pred_class = torch.argmax(outputs, 1)
                # if self.label_column == 'stable_height' or self.label_column == 'total_height':
                #     pred_class = pred_class + 1
                # collecting all the lables and predictions
                gt_labels.extend(labels.cpu().numpy())
                pred_labels.extend(pred_class.cpu().numpy())

                # calculate the loss
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                # calculate the correct predication
                num_cor_pred += (pred_class == labels).sum().item()
                num_samples += labels.size(0)

        val_accuracy = num_cor_pred / num_samples
        self.generate_classification_report(np.array(all_predictions), np.array(all_labels))
        recall_class3 = self.calculate_class3_recall(np.array(all_predictions), np.array(all_labels))
        return val_loss/len(self.valid_loader),val_accuracy, recall_class3


    '''
    The function here is used as the main training function on the image by using the pre-definned models in
    hte first model class.
    '''

    def train(self):
        model_path = create_model_dir(self.store_model_name)

        best_class3_recall = 0.0
        for epoch in range(self.num_epochs):
            self.model.train()
            current_loss = 0.0
            current_acc = 0.0

            # monitor the process
            with tqdm(self.train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{self.num_epochs}")

                for inputs, labels in tepoch:
                    inputs, labels = inputs.to(self.device), labels.to(self.device).long()
                    labels = labels - 1

                    #forward propagation
                    self.optimizer.zero_grad()
                    raw_outputs = self.model(inputs)
                    loss = self.criterion(raw_outputs, labels)

                #  losses = self.criterion(raw_outputs, labels) 
                    
                    # k = max(0.10, 0.4 - epoch * 0.05)

                    # num_hard_examples = int(k * len(losses))
                    # _, hard_example_indices = torch.topk(losses, num_hard_examples)
                    # hard_example_inputs = inputs[hard_example_indices]
                    # hard_example_labels = labels[hard_example_indices]
                    # hard_losses = losses[hard_example_indices]

                    # hard_example_outputs = self.model(hard_example_inputs)
                    # hard_losses_again = self.criterion(hard_example_outputs, hard_example_labels)
                    # hard_losses_again.mean().backward()

                    loss.backward() 
                    self.optimizer.step()

                    # IMPORTANT: Recompute the raw outputs after model weights have been updated
                    # This ensures the accuracy is based on the updated model.
                    # updated_raw_outputs = self.model(inputs)

                    #Loss calculating
                current_loss += loss.item()
                _, pred_class = torch.max(raw_outputs, 1)

                # if self.label_column == 'stable_height' or self.label_column == 'total_height':
                #     pred_class = pred_class + 1
                    # labels = labels - 1

                accuracy = (pred_class == labels).sum().item()/ labels.size(0)
                current_acc += accuracy
                tepoch.set_postfix(loss=current_loss / len(self.train_loader),
                            accuracy=current_acc / len(self.train_loader))


        self.scheduler.step()
        print(self.scheduler.get_last_lr())


        val_loss,val_accuracy, all_labels, all_predictions, class3_recall = self.validate()
        print(f"Epoch {epoch + 1}/{self.num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if class3_recall > best_class3_recall:
            best_class3_recall = class3_recall
            torch.save(self.model.state_dict(), f'{solution_dir}/best_model.pth')
            print('Best model saved!')

        print('Finished Training')
        print(f'Best validation recall: {best_class3_recall:.4f}')
    
class ContinueTrainer:
    '''
    The function here helps to initalize the parameters used in the models and pre-process the image
    '''
    def __init__(self, model_path, model, 
                csv_path, img_dir, store_model_name,label_column,
                stratify_column='stable_height', 
                test_size=0.2,
                batch_size=32, num_epochs=10, learning_rate=0.001 ,random_state=42):
        self.img_dir = img_dir
        self.stratify_column = stratify_column
        self.test_size = test_size
        self.random_state = random_state
        self.store_model_name = store_model_name
        self.label_column = label_column
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.data_frame = pd.read_csv(csv_path)

        self.transform = transforms.Compose([
            # transforms.Resize((512, 512)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), 
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # split data into train and validation 
        # Create data loader
        self.train_data, self.val_data = self.split_train_valid()
        self.train_loader = self.generate_dataloader(self.transform, self.train_data)
        self.valid_loader = self.generate_dataloader(self.transform, self.val_data,  shuffle=False)

        """
        1. Applied class weights to each label, assigning lower weights to more frequent labels and 
            higher weights to less frequent ones. Without these weights, all labels would be treated equally.

        2. Introduced L2 regularization (weight decay) to prevent overfitting. This penalizes large weights by adding a term 
            to the loss function based on the magnitude of the weights, improving generalization to unseen data. 
            The updated loss function is Lnew = Lold + weight_decay * sum(weight^2).

        3. Added a learning rate scheduler that reduces the learning rate after every 'step_size' number of epochs. 
            Specifically, after every 4 epochs, the learning rate is reduced by 90% (multiplied by a gamma of 0.1).
        """
        class_weights = torch.tensor([100/25 , 100/25, 100/20,  100/15, 100/10,  100/5], device= self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights) # CrossEntropy for multi categorical-label predication
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate , weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = 10, gamma=0.15)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        return self.model

    '''
    The function helps to split the data set into the training and validation dataset according to the
    size pre-determined.
    '''

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

    def generate_dataloader(self, transform, data_frame, shuffle=True):
        #  print("label column type", self.label_column)
        dataset = ReadData(data_frame, self.img_dir,self.label_column, transform=transform) 
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def generate_classification_report(self, outputs, labels):
        matrix = confusion_matrix(labels, outputs)
        print(classification_report(labels, outputs, zero_division=0))
        print(matrix)

    """
    validate :
    The function here is used to validate the model by using the pre-definned models.
    """
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        num_cor_pred = 0
        num_samples = 0
        gt_labels = [] 
        pred_labels = []

        with torch.no_grad():
            for inputs, labels in self.valid_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).long()
                labels = labels - 1
                outputs = self.model(inputs)
                pred_class = torch.argmax(outputs, 1)
                if self.label_column == 'stable_height' or self.label_column == 'total_height':
                    pred_class = pred_class + 1
                # collecting all the lables and predictions
                gt_labels.extend(labels.cpu().numpy())
                pred_labels.extend(pred_class.cpu().numpy())

                # calculate the loss
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                # calculate the correct predication
                num_cor_pred += (pred_class == labels).sum().item()
                num_samples += labels.size(0)

        # calculate the accuracy rate
        val_accuracy = num_cor_pred / num_samples
        self.generate_classification_report(np.array(pred_labels), np.array(gt_labels))
        self.pred_labels = pred_labels
        valid_loss = val_loss/len(self.valid_loader)
        return valid_loss, val_accuracy
    '''
    The function here is used as the main training function on the image by using the pre-definned models in
    hte first model class.
    '''

    def train(self):
        model_path = create_model_dir(self.store_model_name)

        best_val_accuracy = 0.0
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            running_accuracy = 0.0

            # monitor the process
            with tqdm(self.train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{self.num_epochs}")

                for inputs, labels in tepoch:
                    inputs, labels = inputs.to(self.device), labels.to(self.device).long()
                    labels = labels - 1

                    #forward propagation
                    self.optimizer.zero_grad()
                    raw_outputs = self.model(inputs)
                    loss = self.criterion(raw_outputs, labels)  # loss calculation

                    loss.backward()  # backward propagation
                    self.optimizer.step()

                    #Loss calculating
                    running_loss += loss.item()
                    _, pred_class = torch.max(raw_outputs, 1)
                    if self.label_column == 'stable_height' or self.label_column == 'total_height':
                        pred_class = pred_class + 1
                        
                    accuracy = (pred_class == labels).sum().item()/ labels.size(0)
                    running_accuracy += accuracy
                    tepoch.set_postfix(loss=running_loss / len(self.train_loader),
                              accuracy=running_accuracy / len(self.train_loader))



            self.scheduler.step()
            print(self.scheduler.get_last_lr())


            valid_loss , val_accuracy = self.validate()
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Validation Loss: {valid_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(self.model.state_dict(), f'{model_path}/best_model.pth')
                current_time = datatime.now().strftime("%Y%m%d-%H%M")
                print('Best model saved!', current_time)

        print('Finished Training')
        print(f'Best validation accuracy: {best_val_accuracy:.4f}')


class BasicTrainer:
    '''
    The function here helps to initalize the parameters used in the models and pre-process the image
    '''
    def __init__(self, csv_path, img_dir, model, store_model_name, label_column, 
                    stratify_column='stable_height', 
                    test_size=0.2, batch_size=32, num_epochs=10, random_state=42,
                    learning_rate=0.001):
            
        self.img_dir = img_dir
        self.stratify_column = stratify_column
        self.test_size = test_size
        self.random_state = random_state
        self.store_model_name = store_model_name
        self.label_column = label_column
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.data_frame = pd.read_csv(csv_path)
        
        self.transform = transforms.Compose([
            # transforms.Resize((512, 512)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), 
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # split data into train and validation 
        # Create data loader
        self.train_data, self.val_data = self.split_train_valid()
        self.train_loader = self.generate_dataloader(self.transform, self.train_data)
        self.valid_loader = self.generate_dataloader(self.transform, self.val_data,  shuffle=False)
    

        """
        1. Applied class weights to each label, assigning lower weights to more frequent labels and 
            higher weights to less frequent ones. Without these weights, all labels would be treated equally.

        2. Introduced L2 regularization (weight decay) to prevent overfitting. This penalizes large weights by adding a term 
            to the loss function based on the magnitude of the weights, improving generalization to unseen data. 
            The updated loss function is Lnew = Lold + weight_decay * sum(weight^2).

        3. Added a learning rate scheduler that reduces the learning rate after every 'step_size' number of epochs. 
            Specifically, after every 4 epochs, the learning rate is reduced by 90% (multiplied by a gamma of 0.1).
        """
        if self.label_column == "stable_height" or self.label_column == "total_height":
            class_weights = torch.tensor([4, 4, 5,  100/15, 10,  20], device= self.device)
        elif self.label_column == "instability_type":
            class_weights = torch.tensor([20,10,5], device= self.device)
        else:
            None 
        self.criterion = nn.CrossEntropyLoss(weight=class_weights) # CrossEntropy for multi categorical-label predication
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate , weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = 15, gamma=0.1)

    """
    validate :
    The function here is used to validate the model by using the pre-definned models.
    """
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        num_cor_pred = 0
        num_samples = 0
        gt_labels = [] 
        pred_labels = []

        with torch.no_grad():
            for inputs, labels in self.valid_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).long()
                labels = labels - 1
                outputs = self.model(inputs)
                pred_class = torch.argmax(outputs, 1)
                # if self.label_column == 'stable_height' or self.label_column == 'total_height':
                #     pred_class = pred_class + 1
                # collecting all the lables and predictions
                gt_labels.extend(labels.cpu().numpy())
                pred_labels.extend(pred_class.cpu().numpy())

                # calculate the loss
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                # calculate the correct predication
                num_cor_pred += (pred_class == labels).sum().item()
                num_samples += labels.size(0)

        # calculate the accuracy rate
        val_accuracy = num_cor_pred / num_samples
        self.generate_classification_report(np.array(pred_labels), np.array(gt_labels))
        self.pred_labels = pred_labels
        valid_loss = val_loss/len(self.valid_loader)
        return valid_loss, val_accuracy

    """
    train :
    The function here is used to train the model by using the pre-definned models.
    """
    def train(self):
        model_path = create_model_dir(self.store_model_name)
        best_valid_acc = 0.0
        
        for epoch in range(self.num_epochs):
            self.model.train()

            current_loss = 0.0
            current_acc = 0.0

            with tqdm(self.train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{self.num_epochs}")

                for inputs, labels in tepoch:
                    inputs, labels = inputs.to(self.device), labels.to(self.device).long()
                     
                    labels = labels - 1
                    
                    self.optimizer.zero_grad()
                    raw_outputs = self.model(inputs)
                    # print(f"Model output shapes: "{raw_outputs.shape})
                    # print(f"Target Labels: "{labels})
                    loss = self.criterion(raw_outputs, labels)

                    loss.backward() 
                    self.optimizer.step()

                    current_loss += loss.item()
                    _, pred_class = torch.max(raw_outputs, 1)

                    # if self.label_column == 'stable_height' or self.label_column == 'total_height':
                    #     pred_class = pred_class + 1
                        # labels = labels - 1

                    accuracy = (pred_class == labels).sum().item()/ labels.size(0)
                    current_acc += accuracy
                    tepoch.set_postfix(loss=current_loss / len(self.train_loader),
                                accuracy=current_acc / len(self.train_loader))

            self.scheduler.step()
            print(self.scheduler.get_last_lr())


            val_loss,val_accuracy  = self.validate()
            
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

            if val_accuracy > best_valid_acc:
                best_valid_acc = val_accuracy
                torch.save(self.model.state_dict(), f'{model_path}/best_model.pth')
                current_time = datatime.now().strftime("%Y%m%d-%H%M")
                print('Best model saved!', current_time)

        print('Finished Training')
        print(f'Best validation accuracy: {best_valid_acc:.4f}')


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

    def generate_dataloader(self, transform, data_frame, shuffle=True):
        #  print("label column type", self.label_column)
        dataset = ReadData(data_frame, self.img_dir,self.label_column, transform=transform) 
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def generate_classification_report(self, outputs, labels):
        matrix = confusion_matrix(labels, outputs)
        print(classification_report(labels, outputs, zero_division=0))
        print(matrix)
