import argparse

from dataset import RetinopathyDataset, train_valid_split
from model import Trainer, EfficientNetFinetune
from test import tester

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import random

# Set Seed

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def main(args):

    train_dir = args.train_dir
    train_csv = args.train_csv
    test_dir = args.test_dir
    test_csv = args.test_csv

    ratio = args.train_valid_ratio
    batch_size = args.batch_size
    epochs = args.epochs

    train_flag = args.train
    pretrain_weight = args.pretrain_weight
    verbose = args.verbose

    if(train_flag==0):
        if(verbose==2):
            print("Reading Training Data...")

        train_csv = pd.read_csv(train_csv)
        train_csv, valid_csv = train_valid_split(train_csv, ratio)

        train = RetinopathyDataset(train_csv, train_dir)
        valid = RetinopathyDataset(valid_csv, train_dir)

        if(verbose==2):
            print("Creating DataLoader...")

        train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_dataloader = DataLoader(valid, batch_size=batch_size*4, shuffle=False, num_workers=4)

        if(verbose==2):
            print("Creating EfficientNet Model...")

        model = EfficientNetFinetune(level="efficientnet-b5", finetune=False, pretrain_weight = "./weights/pretrained/aptos2018.pth")
        
        trainer = Trainer(model, train_dataloader, valid_dataloader, epochs, early_stop="QK", verbose=verbose)

        if(verbose==2):
            print("Strat Training...")
        trainer.train()
    
    if(train_flag==1):
        if(verbose==2):
            print("Strat Predicting...")

        test_csv = pd.read_csv(test_csv)
        test = RetinopathyDataset(test_csv, test_dir, test=True)
        test_dataloader = DataLoader(test, batch_size=batch_size*4, shuffle=False, num_workers=4)
        model = EfficientNetFinetune(level="efficientnet-b5", finetune=False, test=True, pretrain_weight = pretrain_weight)
        tester(model, test_dataloader, verbose)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # Setting parser of dataset
    parser.add_argument("--train_csv", type=str, default="./dataset/input/train.csv" ,required=False, help='The csv of train data (include imageID and label).')
    parser.add_argument("--test_csv", type=str, default="./dataset/input/test.csv" ,required=False, help='The csv of test data (include imageID).')
    parser.add_argument("--train_dir", type=str, default="./dataset/input/train_images/" ,required=False, help='The directory of train data.')
    parser.add_argument("--test_dir", type=str, default="./dataset/input/test_images/" ,required=False, help='The directory of test data.')

    # Setting training parameter
    parser.add_argument("--train_valid_ratio", type=int, default=0.1, required=False, help='Ratio of train data and valid data.')
    parser.add_argument("--batch_size", type=int, default=8, required=False, help='Batch size of model.')
    parser.add_argument("--epochs", type=int, default=15, required=False, help='epochs number for train one discriminator and generator.')

    # Setting train or test flag
    parser.add_argument("--train", type=int, default=False, required=False, help='0: train, 1: test')

    # Setting test weight
    parser.add_argument("--pretrain_weight", type=str, default="", required=False, help='pretrain weight for test')

    # Setting Log type while model training or testing
    parser.add_argument("--verbose", type=int, default=2, required=False, help='log for 0: nothing, 1: valid only, 2: everything')

    main(parser.parse_args())