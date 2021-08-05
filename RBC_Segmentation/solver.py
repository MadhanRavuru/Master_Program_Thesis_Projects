from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 class_weights=None):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.class_weights = class_weights 
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weight = torch.FloatTensor(self.class_weights).to(self.device),reduction='mean')
        self.softmax = nn.Softmax(dim=1)
        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []
        
    def weighted_dice_loss(self, outputs, targets, epsilon=1e-6):
        y_true = torch.eye(outputs.shape[1])[targets].permute(0,3,1,2).float().to(self.device)
        y_pred = self.softmax(outputs) 
        wds = 0.0
        
        for c in range(outputs.shape[1]):
            pred = y_pred[:,c]
            true = y_true[:,c]
            dims = (1,2)  #(H,W)
            intersection = self.class_weights[c]*torch.sum(pred * true, dims)
            cardinality = torch.sum(((pred*pred) + (true*true)), dims)
            wds = wds + (2. * intersection/ (cardinality + epsilon))
        wds = wds/ self.class_weights.sum() 
        return torch.mean(1. - wds)
    
    def focal_loss(self, outputs, targets, gamma = 2):
        ce_loss = F.cross_entropy(outputs, targets, reduction = 'none')  #(N,H,W)
        pt = torch.exp(-ce_loss)
        factor = torch.pow((1. - pt), gamma)
        loss = factor * self.cross_entropy_loss(outputs, targets)    # take reduction as none
        return loss.mean()
    
    def _criterion(self, outputs, targets, epsilon=1e-6):  
        
        loss_1 = self.cross_entropy_loss(outputs, targets)
        #dice loss
        #dice_scores = self.dice_coefficient(outputs, targets)
        #loss_3 = 1. - torch.mean(dice_scores)
        
        #weighted dice loss
        #loss_4 = self.weighted_dice_loss(outputs, targets)
        
        #focal loss
        #loss_5 = self.focal_loss(outputs, targets)
        
        #Generalized dice loss
        y_true = torch.eye(outputs.shape[1])[targets].permute(0,3,1,2).float().to(self.device)
        y_pred = self.softmax(outputs) 
        w = 1/((y_true.sum(axis=[-1,-2]))**2).clamp(min=epsilon)
        w.requires_grad = False
        numerator = (2. * w * (y_pred * y_true).sum(axis=[-1,-2])).sum(axis=-1)
        denominator = (w * (y_pred + y_true).sum(axis=[-1,-2])).sum(axis=-1)
        loss_2 = torch.mean(1. - (numerator / (denominator.clamp(min=epsilon))))
        
        return loss_1 + loss_2
    
    def dice_coefficient(self, outputs, targets, epsilon=1e-6):
        y_true = torch.eye(outputs.shape[1])[targets].permute(0,3,1,2).float().to(self.device)
        y_pred = self.softmax(outputs) 
        
        dims = (1,2,3)
        intersection = torch.sum(y_pred * y_true, dims)
        # here we can use standard dice extension (y_pred + ytrue).sum(-1) or extension like below (See V-Net)
        cardinality = torch.sum(((y_pred*y_pred) + (y_true*y_true)), dims)
        
        dice_score = 2. * intersection/ (cardinality.clamp(epsilon)) 
        return torch.mean(dice_score)
       
        
    
    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
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
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        epoch_loss_train = []
        epoch_loss_val = []
        epoch_dice_coeff = []
        
        for epoch in range(num_epochs):
            # TRAINING
            running_loss_train = 0.0
            running_loss_val = 0.0
            dice_coeff = 0.0
            for i, (inputs, targets) in enumerate(train_loader, 1):
                inputs, targets = Variable(inputs), Variable(targets)
                
                inputs, targets = inputs.to(device), targets.to(device)

                optim.zero_grad()
                outputs = model(inputs)
                loss = self._criterion(outputs, targets)
                loss.backward()
                optim.step()
                running_loss_train += loss.item()*inputs.size(0)
                self.train_loss_history.append(loss.data.cpu().numpy())
                if log_nth and i % log_nth == 0:
                    last_log_nth_losses = self.train_loss_history[-log_nth:]
                    train_loss = np.mean(last_log_nth_losses)
                    print('[Iteration %d/%d] TRAIN loss: %.3f' % \
                        (i + epoch * iter_per_epoch,
                         iter_per_epoch * num_epochs,
                         train_loss))

            _, preds = torch.max(outputs, 1)
            epoch_loss_train.append(running_loss_train/len(train_loader.sampler))
            # Only allow images/pixels with label >= 0 e.g. for segmentation
            targets_mask = targets >= 0
            train_acc = np.mean((preds == targets)[targets_mask].data.cpu().numpy())
            self.train_acc_history.append(train_acc)
            if log_nth:
                print('[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   train_acc,
                                                                   train_loss))
            # VALIDATION
            val_losses = []
            val_scores = []
            model.eval()
            for inputs, targets in val_loader:
                inputs, targets = Variable(inputs), Variable(targets)
                
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model.forward(inputs)
                loss = self._criterion(outputs, targets)
                running_loss_val += loss.item()*inputs.size(0)
                dice_coeff += self.dice_coefficient(outputs, targets).item()*inputs.size(0)
                val_losses.append(loss.data.cpu().numpy())
                
                _, preds = torch.max(outputs, 1)

                # Only allow images/pixels with target >= 0 e.g. for segmentation
                targets_mask = targets >= 0
                scores = np.mean((preds == targets)[targets_mask].data.cpu().numpy())
                val_scores.append(scores)
            epoch_loss_val.append(running_loss_val/len(val_loader.sampler))
            
            dice_coeff = dice_coeff/len(val_loader.sampler)
            
            
            epoch_dice_coeff.append(dice_coeff)
            model.train()
            val_acc, val_loss = np.mean(val_scores), np.mean(val_losses)
            self.val_acc_history.append(val_acc)
            self.val_loss_history.append(val_loss)
            if log_nth:
                print('[Epoch %d/%d] VAL   acc/loss: %.3f/%.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   val_acc,
                                                                   val_loss))
        
        print('Dice Coefficient ', dice_coeff)
        plt.figure(0)
        plt.plot(np.array(epoch_loss_train), label ='Train loss')
        plt.plot(np.array(epoch_loss_val), label ='Val loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Train and validation loss curves')
        plt.grid(True)
        
        plt.figure(1)
        plt.plot(np.array(epoch_dice_coeff),'b')
        plt.xlabel('Epochs')
        plt.ylabel('Dice coefficient')
        plt.title('Dice coefficnet on validation data')
        plt.grid(True)
        pass

        print('FINISH.')
