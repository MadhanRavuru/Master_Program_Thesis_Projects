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

class Solver(object):

    def __init__(self, src_class_weights_size=None, src_class_weights_shape=None, 
                 src_class_weights_hemo_dist=None, src_class_weights_inc_ones=None, 
                 src_class_weights_inc_zeros=None):
        
        
        self.optim = torch.optim.SGD
        self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        
        #source class losses
        self.src_cross_entropy_loss_size = nn.CrossEntropyLoss(weight = torch.FloatTensor(np.array(list(src_class_weights_size.values()))).to(self.device))
        
        self.src_cross_entropy_loss_shape = nn.CrossEntropyLoss(weight = torch.FloatTensor(np.array(list(src_class_weights_shape.values()))).to(self.device))
        
        self.src_cross_entropy_loss_hemo_dist = nn.CrossEntropyLoss(weight = torch.FloatTensor(np.array(list(src_class_weights_hemo_dist.values()))).to(self.device))
        
       
        self.src_class_weights_inc_ones = np.array(list(src_class_weights_inc_ones.values()))
        self.src_class_weights_inc_zeros = np.array(list(src_class_weights_inc_zeros.values()))
         
        self.bce_loss_inclusion = nn.BCELoss(reduction='none')
        
        self.domain_criterion = nn.CrossEntropyLoss()
        
    def src_calc_inclusion_loss(self, net_output, ground_truth):
        
        weights = torch.zeros(ground_truth['inclusion_label'].size())
        for i in range(len(self.src_class_weights_inc_ones)):
            weight = torch.tensor([self.src_class_weights_inc_zeros[i],self.src_class_weights_inc_ones[i]])
            y = ground_truth['inclusion_label'][:,i]
            weights[:,i] = weight[y.data.long()].view_as(y)
        
        inclusion_loss = self.bce_loss_inclusion(net_output['inclusion'], ground_truth['inclusion_label'].type(torch.float))
        inclusion_loss_weighted = inclusion_loss * weights.to(self.device)
        inclusion_loss_weighted = inclusion_loss_weighted.mean()
        return inclusion_loss_weighted
            
    def src_class_criterion(self, net_output, ground_truth):  
        
        size_loss = self.src_cross_entropy_loss_size(net_output['size'], ground_truth['size_label'])
        shape_loss = self.src_cross_entropy_loss_shape(net_output['shape'], ground_truth['shape_label'])
        hemo_dist_loss = self.src_cross_entropy_loss_hemo_dist(net_output['hemo_dist'], ground_truth['hemo_dist_label'])
        
        inclusion_loss = self.src_calc_inclusion_loss(net_output, ground_truth)
        
        loss = size_loss + shape_loss + hemo_dist_loss + 5*inclusion_loss
        
        return loss, {'size': size_loss, 'shape': shape_loss, 'hemo_dist': hemo_dist_loss, 
                      'inclusion': inclusion_loss}
    
    
    def optimizer_scheduler(self, optimizer, p):
        """
        Adjust the learning rate of optimizer
        :param optimizer: optimizer for updating parameters
        :param p: a variable for adjusting learning rate
        :return: optimizer
        """
        optimizer.param_groups[0]['lr'] = 0.001 / (1. + 10 * p) ** 0.75
        optimizer.param_groups[1]['lr'] = 0.01 / (1. + 10 * p) ** 0.75
        optimizer.param_groups[2]['lr'] = 0.01 / (1. + 10 * p) ** 0.75
        
        return optimizer
    
    def plot_grad_flow(self,named_parameters):
        ave_grads = []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
    
    def train(self, feature_extractor, class_classifier, domain_classifier, source_train_dataloader, source_val_dataloader, \
             target_train_dataloader, target_val_dataloader, num_epochs=35):
    
        optim = self.optim([{'params': feature_extractor.parameters()},
                            {'params': class_classifier.parameters(), 'lr': 0.01},
                            {'params': domain_classifier.parameters(), 'lr': 0.01}], lr=1e-3, momentum=0.9, weight_decay=5e-4)

        print('START TRAIN.')
        epoch_loss_train_src_class = []
        epoch_loss_train_domain = []
        epoch_val_tgt_acc = []
        best_acc = 0.0
        
        feature_extractor.train()
        class_classifier.train()
        domain_classifier.train()
        
        for epoch in range(num_epochs):
            # TRAINING
            
         
            # steps
            start_steps = epoch * len(target_train_dataloader)
            total_steps = num_epochs * len(target_train_dataloader)
            
            total_src_hemo_loss = 0.0
            
            total_domain_loss = 0.0
            
            for batch_idx, (sdata, tdata) in enumerate(zip(cycle(source_train_dataloader),target_train_dataloader)): 
                
                # fix adaptation factor
                p = float(batch_idx + start_steps) / total_steps
                constant = (2. / (1. + np.exp(-10* p)) - 1)
      
                # prepare the data
           
                input1 = sdata['img_full_crop']
                label1 = sdata['labels']
               
                input2 = tdata['img_full_crop']
                label2 = tdata['labels']
                
                size = min((input1.shape[0], input2.shape[0]))
                
                input1, label1['size_label'], label1['shape_label'], label1['hemo_dist_label'], label1['inclusion_label'] = input1[0:size, :, :, :], label1['size_label'][0:size], label1['shape_label'][0:size], label1['hemo_dist_label'][0:size], label1['inclusion_label'][0:size]
                
                input2, label2['size_label'], label2['shape_label'], label2['hemo_dist_label'], label2['inclusion_label'] = input2[0:size, :, :, :], label2['size_label'][0:size], label2['shape_label'][0:size], label2['hemo_dist_label'][0:size], label2['inclusion_label'][0:size]
               
                
                input1, input2 = input1.to(self.device), input2.to(self.device)
                label1 = {t: label1[t].to(self.device) for t in label1}
                label2 = {t: label2[t].to(self.device) for t in label2}
                
                # prepare domain labels
                source_labels = torch.zeros((input1.size()[0])).type(torch.LongTensor).to(self.device)
                target_labels = torch.ones((input2.size()[0])).type(torch.LongTensor).to(self.device)
                
                optim = self.optimizer_scheduler(optim, p)
                optim.zero_grad()
                
                # compute the features of source and target domain
                src_feature = feature_extractor(input1)
                tgt_feature = feature_extractor(input2)
                
                # compute the class loss of src_feature
                src_class_preds = class_classifier(src_feature)
                src_class_loss, src_class_losses = self.src_class_criterion(src_class_preds, label1)
                
                # compute the domain loss of src_feature and target_feature
                tgt_preds = domain_classifier(tgt_feature, constant)
                src_preds = domain_classifier(src_feature, constant) 
                tgt_loss = self.domain_criterion(tgt_preds, target_labels)
                src_loss = self.domain_criterion(src_preds, source_labels)
                
                domain_loss = tgt_loss + src_loss
              
                loss = src_class_losses['hemo_dist'] + domain_loss #Only Source hemo loss for label classification is considered
                
                loss.backward()
                plt.figure(1)
                self.plot_grad_flow(domain_classifier.named_parameters())
                optim.step()
                
                total_src_hemo_loss += src_class_losses['hemo_dist'].item()
                total_domain_loss += domain_loss.item()
                
            total_src_hemo_loss = total_src_hemo_loss/len(target_train_dataloader)  
            total_domain_loss = total_domain_loss/len(target_train_dataloader)
            
            #Validation
            feature_extractor.eval()
            class_classifier.eval()
            domain_classifier.eval()
            
            total_acc_score_hemo = 0.0
            with torch.no_grad():
            
                for batch in target_val_dataloader:
                
                
                    img = batch['img_full_crop']
                    target_labels = batch['labels']
                
                    img = img.to(self.device)
           
                    output = class_classifier(feature_extractor(img))
                    _, predicted_shape = output['hemo_dist'].cpu().max(1)
               
                    batch_acc_score_hemo = accuracy_score(target_labels['hemo_dist_label'], predicted_shape.numpy())

                    total_acc_score_hemo += batch_acc_score_hemo
                    
            total_acc = total_acc_score_hemo/len(target_val_dataloader)
            
            if(total_acc > best_acc):
                best_acc = total_acc
                torch.save(feature_extractor.state_dict(), 'models/DANN_hemo_feature_extractor_model_effi_30.pt')  
                torch.save(class_classifier.state_dict(), 'models/DANN_hemo_label_classifier_model_effi_30.pt')
                torch.save(domain_classifier.state_dict(), 'models/DANN_hemo_domain_classifier_model_effi_30.pt')
                
            feature_extractor.train()
            class_classifier.train()
            domain_classifier.train() 
            
            epoch_loss_train_src_class.append(total_src_hemo_loss)
            epoch_loss_train_domain.append(total_domain_loss)
            epoch_val_tgt_acc.append(total_acc) 
         
            
            print('Epoch ',epoch,' Train src hemo loss ',total_src_hemo_loss, ' Train domain loss ',total_domain_loss, ' Val tgt acc hemo',total_acc) 
            
           
       
        plt.figure(0)
        plt.plot(np.array(epoch_loss_train_src_class), label ='Train src hemo loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Train loss curves')
        plt.grid(True)
        
        plt.figure(2)
        plt.plot(np.array(epoch_loss_train_domain), label ='Train domain loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Train Domain loss curve')
        plt.grid(True)
        
        plt.figure(3)
        plt.plot(np.array(epoch_val_tgt_acc), label ='Val target accuracy - hemo')
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

    def validate(self, feature_extractor, class_classifier, domain_classifier, dataloader):

        feature_extractor.eval()
        class_classifier.eval()
        domain_classifier.eval()
        
        
        size_predictions = []
        shape_predictions = []
        hemo_dist_predictions = []
        inclusion_predictions = []
    
        with torch.no_grad():
           
            accuracy_size = 0
            accuracy_shape = 0
            accuracy_hemo_dist = 0
            accuracy_inclusion = 0
          
            accuracy_domain = 0
            
            for batch_idx, batch in enumerate(dataloader):
                
                # setup adaptation factor  (though it has no use in forward pass)
                p = float(batch_idx) / len(dataloader)
                constant = 2. / (1. + np.exp(-10 * p)) - 1
                
                img = batch['img_full_crop']
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
                
                gt_domain_labels = np.ones((img.size()[0]))             # target dataloader - ones,  source dataloader - zeros
                output_domain = domain_classifier(feature_extractor(img), constant)
             
                _, predicted_domain = output_domain.cpu().max(1)
                
                accuracy_domain += accuracy_score(y_true=gt_domain_labels, y_pred=predicted_domain.numpy())
               
        num_test_batches = len(dataloader)
        accuracy_size /= num_test_batches
        accuracy_shape /= num_test_batches
        accuracy_hemo_dist /= num_test_batches
        accuracy_inclusion /= num_test_batches
        accuracy_domain /= num_test_batches
        print('-' * 72)
        print("Validation hemo_dist: {:.4f}, domain : {:.4f} \n".format( accuracy_hemo_dist, accuracy_domain))

        feature_extractor.train()
        class_classifier.train()
        domain_classifier.train()

        return size_predictions, shape_predictions, hemo_dist_predictions, inclusion_predictions
        
