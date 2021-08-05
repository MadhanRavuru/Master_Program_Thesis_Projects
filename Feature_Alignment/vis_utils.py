import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix, precision_score,\
recall_score, f1_score
import matplotlib.pyplot as plt

def visualize_grid(dataset, attributes, size_predictions, shape_predictions, hemo_dist_predictions, inclusion_predictions):
    imgs = []
    labels = []
    predicted_size_all = []
    predicted_shape_all = []
    predicted_hemo_dist_all = []
    predicted_inclusion_all =[]
    
    gt_labels = []
    gt_size_all = []
    gt_shape_all = []
    gt_hemo_dist_all = []
    gt_inclusion_all =[]
    
    gt_inclusions_ids = []
    
    for (sample, 
         predicted_size, 
         predicted_shape, 
         predicted_hemo_dist,
         predicted_inclusion) in zip(
             dataset, size_predictions, shape_predictions, hemo_dist_predictions, inclusion_predictions):
        predicted_size = attributes.size_id_to_name[predicted_size]
        predicted_shape = attributes.shape_id_to_name[predicted_shape]
        predicted_hemo_dist = attributes.hemo_dist_id_to_name[predicted_hemo_dist]
        predicted_inclusion = np.array(attributes.inclusion_labels)[np.argwhere(predicted_inclusion > 0)[:, 0]]

        gt_size = attributes.size_id_to_name[sample['labels']['size_label']]
        gt_shape = attributes.shape_id_to_name[sample['labels']['shape_label']]
        gt_hemo_dist = attributes.hemo_dist_id_to_name[sample['labels']['hemo_dist_label']]
        gt_inclusion = np.array(attributes.inclusion_labels)[np.argwhere(sample['labels']['inclusion_label'] > 0)[:, 0]]
        
        gt_inclusions_ids.append(sample['labels']['inclusion_label'])
        
        predicted_size_all.append(predicted_size)
        predicted_shape_all.append(predicted_shape)
        predicted_hemo_dist_all.append(predicted_hemo_dist)
        predicted_inclusion_all.append(predicted_inclusion)
        
        gt_size_all.append(gt_size)
        gt_shape_all.append(gt_shape)
        gt_hemo_dist_all.append(gt_hemo_dist)
        gt_inclusion_all.append(gt_inclusion)
        

        imgs.append(sample['img'])
        
        labels.append("{}\n{}\n{}\n{}".format(predicted_size, predicted_shape, predicted_hemo_dist, \
                                                  ', '.join(predicted_inclusion)))
        gt_labels.append("{}\n{}\n{}\n{}".format(gt_size, gt_shape, gt_hemo_dist, ', '.join(gt_inclusion)))

   
    ml_cn = multilabel_confusion_matrix(gt_inclusions_ids, inclusion_predictions)
    print(ml_cn)
    plt.rcParams.update({'font.size': 20})
    for i, label in zip(range(attributes.num_inclusion), attributes.inclusion_labels):
        #ml_cn[i] = ml_cn[i]/ml_cn[i].sum(axis=1)
        ConfusionMatrixDisplay(confusion_matrix=ml_cn[i], display_labels=[0,1]).plot(xticks_rotation='horizontal')
        plt.title(label)
        plt.show()
        
    print('Metrics for Inclusions ')
    print('precision ', precision_score(y_true=gt_inclusions_ids, y_pred=inclusion_predictions, average=None))
    print('recall ', recall_score(y_true=gt_inclusions_ids, y_pred=inclusion_predictions, average=None))
    print('f1_score ',f1_score(y_true=gt_inclusions_ids, y_pred=inclusion_predictions, average=None))
    
