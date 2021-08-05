from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import warnings
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR

class Solver(object):

    def __init__(self, class_weights_size=None, class_weights_shape=None, 
                 class_weights_hemo_dist=None, class_weights_inc_ones=None, 
                 class_weights_inc_zeros=None):
        
        self.optim_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.001}
        self.optim = torch.optim.Adam
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        
        self.cross_entropy_loss_size = nn.CrossEntropyLoss(weight = torch.FloatTensor(np.array(list(class_weights_size.values()))).to(self.device))
        
        self.cross_entropy_loss_shape = nn.CrossEntropyLoss(weight = torch.FloatTensor(np.array(list(class_weights_shape.values()))).to(self.device))
        
        self.cross_entropy_loss_hemo_dist = nn.CrossEntropyLoss(weight = torch.FloatTensor(np.array(list(class_weights_hemo_dist.values()))).to(self.device))
        
        self.bce_loss_inclusion = nn.BCELoss(reduction='none')
        self.class_weights_inc_ones = np.array(list(class_weights_inc_ones.values()))
        self.class_weights_inc_zeros = np.array(list(class_weights_inc_zeros.values()))
        
        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []
        
    def calc_inclusion_loss(self, net_output, ground_truth):
        
        weights = torch.zeros(ground_truth['inclusion_label'].size())
        for i in range(len(self.class_weights_inc_ones)):
            weight = torch.tensor([self.class_weights_inc_zeros[i],self.class_weights_inc_ones[i]])
            y = ground_truth['inclusion_label'][:,i]
            weights[:,i] = weight[y.data.long()].view_as(y)
        
        inclusion_loss = self.bce_loss_inclusion(net_output['inclusion'], ground_truth['inclusion_label'].type(torch.float))
        inclusion_loss_weighted = inclusion_loss * weights.to(self.device)
        inclusion_loss_weighted = inclusion_loss_weighted.mean()
        return inclusion_loss_weighted
    
    #Our loss criterion is linear combination of losses from  cell properties
    def _criterion(self, net_output, ground_truth):  
        
        size_loss = self.cross_entropy_loss_size(net_output['size'], ground_truth['size_label'])
        shape_loss = self.cross_entropy_loss_shape(net_output['shape'], ground_truth['shape_label'])
        hemo_dist_loss = self.cross_entropy_loss_hemo_dist(net_output['hemo_dist'], ground_truth['hemo_dist_label'])
        
        inclusion_loss = self.calc_inclusion_loss(net_output, ground_truth)
        
        loss = size_loss + shape_loss + hemo_dist_loss + 5*inclusion_loss
        
        return loss, {'size': size_loss, 'shape': shape_loss, 'hemo_dist': hemo_dist_loss, 
                      'inclusion': inclusion_loss}
   
       
        
    
    def train(self, model, train_dataloader, test_dataloader, num_epochs=10):
      
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        scheduler = StepLR(optim, step_size=31, gamma=0.1)        # decreasing LR by 0.1 after 30 epochs
        model.to(self.device)

        print('START TRAIN.')
        epoch_loss_train = []
        epoch_loss_val = []
       
        for epoch in range(num_epochs):
            # TRAINING
            total_loss = 0.0
            total_loss_val = 0.0
            
            for batch in train_dataloader:
                
                inputs = batch['img']
                target_labels = batch['labels']
              
                inputs = inputs.to(self.device)
                target_labels = {t: target_labels[t].to(self.device) for t in target_labels}

                optim.zero_grad()
                outputs = model(inputs)
                loss_train, losses_train = self._criterion(outputs, target_labels)
               
                total_loss += loss_train.item()*inputs.size(0)
                
                loss_train.backward()
                optim.step()
             
            scheduler.step()
         
            epoch_loss_train.append(total_loss/len(train_dataloader.sampler))
            
            model.eval()
            with torch.no_grad():
                for batch in test_dataloader:
                    inputs = batch['img']
                    target_labels = batch['labels']
               
                    inputs = inputs.to(self.device)
                    target_labels = {t: target_labels[t].to(self.device) for t in target_labels}
                    outputs = model(inputs)
                    loss_val, losses_val = self._criterion(outputs, target_labels)
                
                    total_loss_val += loss_val.item()*inputs.size(0)
            epoch_loss_val.append(total_loss_val/len(test_dataloader.sampler))
            
            print('Epoch ',epoch,' Train Loss ',total_loss/len(train_dataloader.sampler), 'LR ',optim.param_groups[0]['lr'],   ' Val Loss ',total_loss_val/len(test_dataloader.sampler)) 
              
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
    
    def calculate_metrics(self, output, target):
        _, predicted_size = output['size'].cpu().max(1)
        gt_size = target['size_label'].cpu()

        _, predicted_shape = output['shape'].cpu().max(1)
        gt_shape = target['shape_label'].cpu()

        _, predicted_hemo_dist = output['hemo_dist'].cpu().max(1)
        gt_hemo_dist = target['hemo_dist_label'].cpu()
    
        predicted_inclusion = np.array(output['inclusion'].cpu() > 0.5, dtype=float)
            
        with warnings.catch_warnings():  # sklearn may produce a warning when processing zero row in confusion matrix
            warnings.simplefilter("ignore")
            accuracy_size = accuracy_score(y_true=gt_size.numpy(), y_pred=predicted_size.numpy())
            accuracy_shape = accuracy_score(y_true=gt_shape.numpy(), y_pred=predicted_shape.numpy())
            accuracy_hemo_dist = accuracy_score(y_true=gt_hemo_dist.numpy(), y_pred=predicted_hemo_dist.numpy())
            accuracy_inclusion = accuracy_score(y_true=target['inclusion_label'].cpu(), y_pred=predicted_inclusion)
            
        return accuracy_size, accuracy_shape, accuracy_hemo_dist, accuracy_inclusion 

    def net_output_to_predictions(self, output):
        _, predicted_size = output['size'].cpu().max(1)
        _, predicted_shape = output['shape'].cpu().max(1)
        _, predicted_hemo_dist = output['hemo_dist'].cpu().max(1)
        predicted_inclusion = np.array(output['inclusion'].cpu() > 0.5, dtype=float)

        return predicted_size.detach().numpy().tolist(), predicted_shape.detach().numpy().tolist(),\
               predicted_hemo_dist.detach().numpy().tolist(), predicted_inclusion

    def validate(self, model, dataloader):

        model.eval()
        size_predictions = []
        shape_predictions = []
        hemo_dist_predictions = []
        inclusion_predictions = []
    
        with torch.no_grad():
          
            accuracy_size = 0
            accuracy_shape = 0
            accuracy_hemo_dist = 0
            accuracy_inclusion = 0
          

            for batch in dataloader:
                img = batch['img']
                target_labels = batch['labels']
                target_labels = {t: target_labels[t].to(self.device) for t in target_labels}
                output = model(img.to(self.device))

                batch_accuracy_size, batch_accuracy_shape, batch_accuracy_hemo_dist,\
                batch_accuracy_inclusion = self.calculate_metrics(output, target_labels)

                accuracy_size += batch_accuracy_size
                accuracy_shape += batch_accuracy_shape
                accuracy_hemo_dist += batch_accuracy_hemo_dist
                accuracy_inclusion += batch_accuracy_inclusion
                
            
                (batch_size_predictions,
                 batch_shape_predictions,
                 batch_hemo_dist_predictions,
                 batch_inclusion_predictions) = self.net_output_to_predictions(output)

                size_predictions.extend(batch_size_predictions)
                shape_predictions.extend(batch_shape_predictions)
                hemo_dist_predictions.extend(batch_hemo_dist_predictions)
                inclusion_predictions.extend(batch_inclusion_predictions)
            
        num_test_batches = len(dataloader)
        accuracy_size /= num_test_batches
        accuracy_shape /= num_test_batches
        accuracy_hemo_dist /= num_test_batches
        accuracy_inclusion /= num_test_batches
        print('-' * 72)
        print("Validation  accuracy size: {:.4f}, shape: {:.4f}, hemo_dist: {:.4f}, inclusion: {:.4f}\n".format(accuracy_size, accuracy_shape, accuracy_hemo_dist, accuracy_inclusion))

        model.train()

        return size_predictions, shape_predictions, hemo_dist_predictions, inclusion_predictions
        