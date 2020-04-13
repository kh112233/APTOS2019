import os

from PIL import Image
import cv2

import torch
import torchvision
from torchvision import transforms 
from torch.utils.data import Dataset

from sklearn.utils import shuffle
import random

 
class RetinopathyDataset(Dataset):
    def __init__(self, csv_df, img_path, test=False):

        """
        csv_df:     Dataframe of input data
        img_path:   Path of the directory that store images
        test:       Dataset is for test or train
        """
        
        self.csv_df = csv_df
        self.img_path = img_path
        self.test = test

        self.img_size = 224

    def transform(self, image):
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        if not self.test:
            if random.random() > 0.5:
                image = cv2.flip(image, 0)
            
            if random.random() > 0.5:
                image = cv2.flip(image, 1)

        image = transforms.functional.to_tensor(image)
            
        return image

    def __len__(self):
        return len(self.csv_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_path, self.csv_df.loc[idx, 'id_code'] + '.png')
        image = cv2.imread(img_name)
        image = self.transform(image)

        if self.test:
            return image
        else:
            label = torch.tensor(self.csv_df.loc[idx, 'diagnosis'])
            return image, label

def train_valid_split(train_data, ratio):
    shuffled_data = shuffle(train_data)
    valid_data = shuffled_data.iloc[0:int(len(train_data)*ratio)]
    valid_data = valid_data.reset_index(drop=True)
    
    train_data = shuffled_data.iloc[int(len(train_data)*ratio):]
    train_data = train_data.reset_index(drop=True)
    return train_data, valid_data
