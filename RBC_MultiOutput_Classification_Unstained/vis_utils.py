import numpy as np
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
        
        imgs.append(sample['img_center_cell'])
        labels.append("{}\n{}\n{}\n{}".format(predicted_size, predicted_shape, predicted_hemo_dist, \
                                                  ', '.join(predicted_inclusion)))
        gt_labels.append("{}\n{}\n{}\n{}".format(gt_size, gt_shape, gt_hemo_dist, ', '.join(gt_inclusion)))

    # Draw confusion matrices
    # size
    cn_matrix = confusion_matrix(
        y_true=gt_size_all,
        y_pred=predicted_size_all,
        labels=attributes.size_labels,
        normalize='true')                      # diagonal elements are recall values
    
    cn_matrix = (cn_matrix.astype('float') / cn_matrix.astype('float').sum(axis=1)[:, np.newaxis])*100

    fmt = '.2f'
    
    plt.rcParams.update({'font.size': 15})
    plt.rcParams.update({'figure.dpi': 80})
    disp = ConfusionMatrixDisplay(confusion_matrix=cn_matrix, display_labels=attributes.size_labels)
    disp.plot(values_format=fmt,xticks_rotation='horizontal', cmap=plt.cm.Blues)
    plt.title("Sizes")
    plt.tight_layout()
    #plt.savefig('DIC_CM_size.png')
    plt.show()

    # shape
    cn_matrix = confusion_matrix(
        y_true=gt_shape_all,
        y_pred=predicted_shape_all,
        labels=attributes.shape_labels,
        normalize='true')
    cn_matrix = (cn_matrix.astype('float') / cn_matrix.astype('float').sum(axis=1)[:, np.newaxis])*100
    plt.rcParams.update({'font.size': 10})
    plt.rcParams.update({'figure.dpi': 100})
    disp = ConfusionMatrixDisplay(confusion_matrix=cn_matrix, display_labels=attributes.shape_labels)
    disp.plot(values_format=fmt,xticks_rotation='horizontal', cmap=plt.cm.Blues)
    plt.title("Shapes")
    plt.tight_layout()
    #plt.savefig('DIC_CM_shape.png')
    plt.show()

    
    cn_matrix = confusion_matrix(
        y_true=gt_hemo_dist_all,
        y_pred=predicted_hemo_dist_all,
        labels=attributes.hemo_dist_labels,
        normalize='true')
    cn_matrix = (cn_matrix.astype('float') / cn_matrix.astype('float').sum(axis=1)[:, np.newaxis])*100
    plt.rcParams.update({'font.size': 12})
    plt.rcParams.update({'figure.dpi': 100})
    disp = ConfusionMatrixDisplay(confusion_matrix=cn_matrix, display_labels=attributes.hemo_dist_labels)
    disp.plot(values_format=fmt,xticks_rotation='horizontal', cmap=plt.cm.Blues)
    plt.title("Hemoglobin distributions")
    #plt.savefig('DIC_CM_hemo.png')
    plt.show()
   
    
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
    
    
    plt.rcParams.update({'font.size': 15})
    plt.rcParams.update({'figure.dpi': 150})
    title = "GT and Predicted labels"
    n_cols = 4
    n_rows = 4
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    axs = axs.flatten()
    for img, ax, gt_label, label in zip(reversed(imgs),reversed(axs), reversed(gt_labels), reversed(labels)):
        ax.set_xlabel(label, fontsize = 10)
        ax.set_ylabel(gt_label, fontsize = 10, rotation ='horizontal', ha ='right', va='center')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
       
        ax.imshow(img.numpy().transpose(1,2,0))     # Center cell is shown for display irrespective of input type 
                                                    # (But prediction results come from respective input types)
    plt.suptitle(title, fontsize =15)
    plt.tight_layout()
    #plt.savefig('DIC_Output.png')
    plt.show()
