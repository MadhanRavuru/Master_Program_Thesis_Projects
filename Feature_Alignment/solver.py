from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import warnings
from sklearn.metrics import accuracy_score, f1_score
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
        self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        
        #class losses
        self.cross_entropy_loss_size = nn.CrossEntropyLoss(weight = torch.FloatTensor(np.array(list(class_weights_size.values()))).to(self.device))
        
        self.cross_entropy_loss_shape = nn.CrossEntropyLoss(weight = torch.FloatTensor(np.array(list(class_weights_shape.values()))).to(self.device))
        
        self.cross_entropy_loss_hemo_dist = nn.CrossEntropyLoss(weight = torch.FloatTensor(np.array(list(class_weights_hemo_dist.values()))).to(self.device))
        
       
        self.class_weights_inc_ones = np.array(list(class_weights_inc_ones.values()))
        self.class_weights_inc_zeros = np.array(list(class_weights_inc_zeros.values()))
         
        self.bce_loss_inclusion = nn.BCELoss(reduction='none')
        
        
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
            
    def class_criterion(self, net_output, ground_truth):  
        
        size_loss = self.cross_entropy_loss_size(net_output['size'], ground_truth['size_label'])
        shape_loss = self.cross_entropy_loss_shape(net_output['shape'], ground_truth['shape_label'])
        hemo_dist_loss = self.cross_entropy_loss_hemo_dist(net_output['hemo_dist'], ground_truth['hemo_dist_label'])
        
        inclusion_loss = self.calc_inclusion_loss(net_output, ground_truth)
        
        loss = size_loss + shape_loss + hemo_dist_loss + 5*inclusion_loss
        
        return loss, {'size': size_loss, 'shape': shape_loss, 'hemo_dist': hemo_dist_loss, 
                      'inclusion': inclusion_loss}
    
        
    def mmd_linear(self,f_of_X, f_of_Y):
        delta = f_of_X - f_of_Y
        loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
        return loss
    
    def train(self, feature_extractor, class_classifier, train_dataloader, val_dataloader, num_epochs=35):
       
        optim = self.optim([{'params': feature_extractor.parameters()},
                            {'params': class_classifier.parameters()}], **self.optim_args)
                          

        print('START TRAIN.')
        epoch_loss_train_src_class = []
        epoch_loss_train_tgt_class = []
        epoch_loss_train_mmd = []
        epoch_val_tgt_acc = []
        best_acc = 0.0
        
        scheduler = StepLR(optim, step_size=16, gamma=0.1) 
        
        feature_extractor.train()
        class_classifier.train()
        
        for epoch in range(num_epochs):
            # TRAINING
            
         
            # steps
            start_steps = epoch * len(train_dataloader)
            total_steps = num_epochs * len(train_dataloader)
            
            total_src_inc_loss = 0.0
            total_tgt_inc_loss = 0.0
            
            total_mmd_loss = 0.0
            
            for batch_idx, data in enumerate(train_dataloader): 
                
      
                # prepare the data
           
                bf_dic_input = data['img']
                bf_dic_input = torch.split(bf_dic_input, 3, dim =1) 
                bf_input = bf_dic_input[0]
                dic_input = bf_dic_input[1]
               
                label = data['labels']
               
                
                bf_input, dic_input = bf_input.to(self.device), dic_input.to(self.device)
                label = {t: label[t].to(self.device) for t in label}
                
                optim.zero_grad()
                
                # compute the features of source and target domain
                src_feature = feature_extractor(bf_input)
                tgt_feature = feature_extractor(dic_input)
                
                # compute the class loss of src_feature
                src_class_preds = class_classifier(src_feature)
                src_class_loss, src_class_losses = self.class_criterion(src_class_preds, label)
                
                # compute the class loss of tgt_feature
                tgt_class_preds = class_classifier(tgt_feature)
                tgt_class_loss, tgt_class_losses = self.class_criterion(tgt_class_preds, label)
              
                
                mmd_loss = self.mmd_linear(src_feature, tgt_feature)
                loss = src_class_losses['inclusion'] + tgt_class_losses['inclusion'] + mmd_loss # Only inclusion losses 
                loss.backward()
               
                optim.step()
                
                total_src_inc_loss += src_class_losses['inclusion'].item()
                total_tgt_inc_loss += tgt_class_losses['inclusion'].item()
                total_mmd_loss += mmd_loss.item()
            
            scheduler.step()
            total_src_inc_loss = total_src_inc_loss/len(train_dataloader)  
            total_tgt_inc_loss = total_tgt_inc_loss/len(train_dataloader) 
            total_mmd_loss = total_mmd_loss/len(train_dataloader)
            
            #Validation
            feature_extractor.eval()
            class_classifier.eval()
            
            total_acc_score_inc = 0.0
            with torch.no_grad():
            
                for batch in val_dataloader:
                
                    bf_dic_input = batch['img']
                    bf_dic_input = torch.split(bf_dic_input, 3, dim =1) 
              
                    img = bf_dic_input[1]          # DIC input
                    
                    target_labels = batch['labels']
                
                    img = img.to(self.device)
           
                    output = class_classifier(feature_extractor(img))
                    predicted_inclusion = np.array(output['inclusion'].cpu() > 0.5, dtype=float)
               
                    batch_acc_score_inc = accuracy_score(target_labels['inclusion_label'], predicted_inclusion)


                    total_acc_score_inc += batch_acc_score_inc
                    
            total_acc = total_acc_score_inc/len(val_dataloader)
            
            if(total_acc > best_acc):
                best_acc = total_acc
                torch.save(feature_extractor.state_dict(), 'models/center_Inc_feature_extractor_model_effi_20.pt')  
                torch.save(class_classifier.state_dict(), 'models/center_Inc_label_classifier_model_effi_20.pt')
                
            feature_extractor.train()
            class_classifier.train()
            
            epoch_loss_train_src_class.append(total_src_inc_loss)
            epoch_loss_train_tgt_class.append(total_tgt_inc_loss)
            epoch_loss_train_mmd.append(total_mmd_loss)
            epoch_val_tgt_acc.append(total_acc) 
         
            
            print('Epoch ',epoch,' Train src inc loss ',total_src_inc_loss, ' Train tgt inc loss ',total_tgt_inc_loss, ' Train mmd loss ',total_mmd_loss, ' Val tgt acc hemo',total_acc) 
            
           
       
        plt.figure(0)
        plt.plot(np.array(epoch_loss_train_src_class), label ='Train src inc loss')
        plt.plot(np.array(epoch_loss_train_tgt_class), label ='Train tgt inc loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Train loss curves')
        plt.grid(True)
        
        plt.figure(2)
        plt.plot(np.array(epoch_loss_train_mmd), label ='Train mmd loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Train mmd loss curve')
        plt.grid(True)
        
        plt.figure(3)
        plt.plot(np.array(epoch_val_tgt_acc), label ='Val target accuracy - inc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Validation Accuracy curve')
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

    def validate(self, feature_extractor, class_classifier, dataloader):

        feature_extractor.eval()
        class_classifier.eval()
     
        
        
        size_predictions = []
        shape_predictions = []
        hemo_dist_predictions = []
        inclusion_predictions = []
    
        with torch.no_grad():
           
            accuracy_size = 0
            accuracy_shape = 0
            accuracy_hemo_dist = 0
            accuracy_inclusion = 0
            
            for batch_idx, batch in enumerate(dataloader):
                
                bf_dic_input = batch['img']
                bf_dic_input = torch.split(bf_dic_input, 3, dim =1) 
              
                img = bf_dic_input[1]          # DIC input- index 1,   BF input - index 0
                
                target_labels = batch['labels']
                
                img = img.to(self.device)
                target_labels = {t: target_labels[t].to(self.device) for t in target_labels}
                output = class_classifier(feature_extractor(img))

               
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
        print("Validation  accuracy inclusion: {:.4f} \n".format(accuracy_inclusion))

        feature_extractor.train()
        class_classifier.train()

        return size_predictions, shape_predictions, hemo_dist_predictions, inclusion_predictions
        
