from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import warnings

class Solver(object):

    def __init__(self, class_weights=None):
        
        self.optim_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.001}
        self.optim = torch.optim.Adam
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        
       
        self.bce_loss = nn.BCELoss(reduction='none')
        self.class_weights = torch.FloatTensor(class_weights).to(self.device)

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []
        
    def _criterion(self, outputs, targets):              # weighted binary cross entropy loss
        
        weights_ = self.class_weights[targets.data.view(-1).long()].view_as(targets)
        loss = self.bce_loss(outputs, targets.type(torch.float))
        loss_weighted = loss * weights_
        loss_weighted = loss_weighted.mean()                            
        return loss_weighted
    
    def train(self, model, train_dataloader, val_dataloader, num_epochs=10):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        
        model.to(self.device)

        print('START TRAIN.')
        epoch_loss_train = []
        epoch_loss_val = []
        
        
        for epoch in range(num_epochs):
            # TRAINING
            total_loss = 0.0
            total_loss_val = 0.0
            for (inputs, targets) in train_dataloader:
                inputs, targets = Variable(inputs), Variable(targets)
                
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optim.zero_grad()
                outputs = model(inputs)
               
                outputs = outputs.view_as(targets)
               
                loss = self._criterion(outputs, targets)
                loss.backward()
                optim.step()
                total_loss += loss.item()*inputs.size(0)
            
            epoch_loss_train.append(total_loss/len(train_dataloader.sampler))
            
            model.eval()
            for (inputs, targets) in val_dataloader:
                inputs, targets = Variable(inputs), Variable(targets)
                
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                outputs = outputs.view_as(targets)
                loss_val = self._criterion(outputs, targets)
                
                total_loss_val += loss_val.item()*inputs.size(0)
            epoch_loss_val.append(total_loss_val/len(val_dataloader.sampler))
            
            print('Epoch ',epoch,'Train Loss ',total_loss/len(train_dataloader.sampler),'Val loss ',total_loss_val/len(val_dataloader.sampler))
            model.train()
        
        plt.figure(0)
        plt.plot(np.array(epoch_loss_train), label ='Train loss')
        plt.plot(np.array(epoch_loss_val), label ='Val loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Train and Validation loss curves')
        plt.grid(True)
        

        print('FINISH.')
    
   