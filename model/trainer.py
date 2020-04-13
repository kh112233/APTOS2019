import time
import os

from .efficientNet_finetune import EfficientNetFinetune

import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import cohen_kappa_score

class Trainer():
    def __init__(self, model, 
                 train_dataloader: DataLoader, valid_dataloader: DataLoader = None,
                 epochs=100, lr=0.001, early_stop = "None",
                 verbose = 2, with_cuda: bool = True):          
        """
        model:              Model use on training
        train_dataloader:   Dataloader of training data
        valid_dataloader:   Dataloader of validating data
        epochs:             Training epochs
        early_stop:         Early stop by quadratic weighted kappa(QK), mean square error(MSE) or None
        lr:                 Learning rate
        verbose:            Log type (0: nothing, 1:for valid, 2:everything)
        with_cuda:          Training with GPU or not
        """

        # Setup cuda device for BERT model training
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # Setup model
        self.model = model
        self.model = self.model.cuda()

        # Load data information
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        
        # Setup training parameter
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        assert early_stop in ["None", "QK", "MSE"]
        self.early_stop = early_stop
        self.verbose = verbose

        # Weight saving path
        self.save_path = "./weights/train/"

    # Metrics: quadratic weighted kappa
    def qk(self, y_pred, y):
        return torch.tensor(cohen_kappa_score(torch.round(y_pred.cpu()).detach().numpy(), y.cpu(), weights='quadratic'), device=self.device)

    def valid_func(self):
        outputs_list = torch.zeros([])
        labels_list = torch.zeros([])

        for i, batch in enumerate(self.valid_dataloader):
            images, labels = batch
            images = images.cuda()
            labels = labels.cuda()

            outputs = self.model(images)
            outputs = outputs.squeeze(-1)
            if(i==0):
                outputs_list = outputs
                labels_list = labels
            else:
                outputs_list = torch.cat([outputs_list, outputs], -1) 
                labels_list = torch.cat([labels_list, labels], -1) 
        
        valid_loss = self.criterion(outputs_list, labels_list)
        valid_metrics = self.qk(outputs_list, labels_list)

        return valid_loss, valid_metrics     
            
    # Trainnig 
    def train(self):
        since = time.time()
        best_loss = 100
        best_metrics = 0
        patience = 0
        iteration = 0
        early_stop_flag = False

        for epoch in range(self.epochs):
            for batch in self.train_dataloader:        
                self.model.train()
                images, labels = batch   
                images = images.type(torch.FloatTensor).cuda()
                labels = labels.type(torch.FloatTensor).cuda()
                self.optimizer.zero_grad()
                outputs = self.model(images)
                outputs = outputs.squeeze(-1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                iteration += 1

                if (self.verbose == 2):
                    print(f'iter {iteration} | MSE: {loss:.5f}                \r', end='')
                
                # Validation
                if iteration % 500 == 0:
                    if (self.verbose == 2):
                        print('Validating...                                      \r', end='')

                    self.model.eval()
                    with torch.no_grad():
                        val_loss, val_metrics= self.valid_func()

                    if (self.verbose == 1 or self.verbose == 2):
                        time_elapsed = time.time() - since
                        print(f'iter {iteration} | val_MSE: {val_loss:.5f}, val_QK: {val_metrics:.5f}, time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                    
                    # Earlystopping
                    if(self.early_stop == "MSE"):
                        if val_loss < best_loss:
                            best_loss = val_loss

                            if not os.path.exists(self.save_path):
                                os.mkdir(self.save_path)
                            self.model.save_weight(self.save_path + f"MSE:{best_loss:.5f}.ckpt")
                            patience = 0
                        else:
                             patience+=1

                        if (self.verbose == 2):
                            print(f'iter {iteration} | best_MSE: {best_loss:.5f}, patience: {patience:.0f}')

                    elif(self.early_stop == "QK"):
                        if val_metrics > best_metrics:
                            best_metrics = val_metrics
                            
                            if not os.path.exists(self.save_path):
                                os.mkdir(self.save_path)
                            self.model.save_weight(self.save_path + f"QK:{best_metrics:.5f}.ckpt")
                            patience = 0
                        else:
                             patience+=1

                        if (self.verbose == 2):
                            print(f'iter {iteration} | best_QK: {best_metrics:.5f}, patience: {patience:.0f}')

                    if patience >= 5:
                        early_stop_flag = True
                if early_stop_flag:
                    break
            if early_stop_flag:
                break
            
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))