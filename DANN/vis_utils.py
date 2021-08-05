import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix, precision_score,\
recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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
        

        imgs.append(sample['img_full_crop'])
        labels.append("{}\n{}\n{}\n{}".format(predicted_size, predicted_shape, predicted_hemo_dist, \
                                                  ', '.join(predicted_inclusion)))
        gt_labels.append("{}\n{}\n{}\n{}".format(gt_size, gt_shape, gt_hemo_dist, ', '.join(gt_inclusion)))

    # Draw confusion matrices

    # Hemoglobin distribution
    cn_matrix = confusion_matrix(
        y_true=gt_hemo_dist_all,
        y_pred=predicted_hemo_dist_all,
        labels=attributes.hemo_dist_labels,
        normalize='true')
    cn_matrix = (cn_matrix.astype('float') / cn_matrix.astype('float').sum(axis=1)[:, np.newaxis])*100
    fmt = '.2f'
    plt.rcParams.update({'font.size': 12})
    plt.rcParams.update({'figure.dpi': 100})
    disp = ConfusionMatrixDisplay(confusion_matrix=cn_matrix, display_labels=attributes.hemo_dist_labels)
    disp.plot(values_format=fmt,xticks_rotation='horizontal', cmap=plt.cm.Blues)
    plt.title("Hemoglobin distributions")
    plt.show()
    
# Visualization of classification logit space from source and target domains 
def visualizePerformance(feature_extractor, class_classifier, domain_classifier, src_test_dataloader,
                         tgt_test_dataloader):
   

    # Setup the network
    feature_extractor.eval()
    class_classifier.eval()
    domain_classifier.eval()

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        # Collect source data.
        s_feats, s_tags = [], []
        for batch in src_test_dataloader:
            images = batch['img_full_crop']

            feats = class_classifier(feature_extractor(images.to(device)))
            
            target_labels = batch['labels']
          
                
            s_feats.append(feats['hemo_dist'])
            s_tags.append(target_labels['hemo_dist_label'].type(torch.LongTensor))


        s_feats, s_tags = torch.cat(s_feats)[:len(src_test_dataloader.sampler)], \
                                 torch.cat(s_tags)[:len(src_test_dataloader.sampler)]

        print(s_feats.size())
        print(s_tags.size())
    
        # Collect test data.
        t_feats, t_tags = [], []
        for batch in tgt_test_dataloader:
            images = batch['img_full_crop']
            feats = class_classifier(feature_extractor(images.to(device)))
            target_labels = batch['labels']
            target_labels['hemo_dist_label']+=5
            t_feats.append(feats['hemo_dist'])
        
            t_tags.append(target_labels['hemo_dist_label'].type(torch.LongTensor))


        t_feats, t_tags = torch.cat(t_feats)[:len(tgt_test_dataloader.sampler)], \
                                 torch.cat(t_tags)[:len(tgt_test_dataloader.sampler)]

   
        print(t_feats.size())
        print(t_tags.size())
    
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)

    pca = PCA(n_components=2)
    dann_tsne = tsne.fit_transform(np.concatenate((s_feats.cpu().detach().numpy(),
                                                       t_feats.cpu().detach().numpy())))
    

    print(dann_tsne.shape)
    
    feature_extractor.train()
    class_classifier.train()
    domain_classifier.train()
    
    return dann_tsne, np.concatenate((s_tags, t_tags))
 

# Visualization of latent space feature from source and target domains    
def visualizePerformanceFeats(feature_extractor, class_classifier, domain_classifier, src_test_dataloader,
                         tgt_test_dataloader):
   

    # Setup the network
    feature_extractor.eval()
    class_classifier.eval()
    domain_classifier.eval()

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        # Collect source data.
        s_feats, s_tags = [], []
        for batch in src_test_dataloader:
            images = batch['img_full_crop']

            feats = feature_extractor(images.to(device))
                
            s_feats.append(feats)
            s_tags.append(torch.zeros((images.size()[0])).type(torch.LongTensor))


        s_feats, s_tags = torch.cat(s_feats)[:len(src_test_dataloader.sampler)], \
                                 torch.cat(s_tags)[:len(src_test_dataloader.sampler)]

        print(s_feats.size())
        print(s_tags.size())
    
        # Collect test data.
        t_feats, t_tags = [], []
        for batch in tgt_test_dataloader:
            images = batch['img_full_crop']
            feats = feature_extractor(images.to(device))
           
            t_feats.append(feats)
        
            t_tags.append(torch.ones((images.size()[0])).type(torch.LongTensor))


        t_feats, t_tags = torch.cat(t_feats)[:len(tgt_test_dataloader.sampler)], \
                                 torch.cat(t_tags)[:len(tgt_test_dataloader.sampler)]

   
        print(t_feats.size())
        print(t_tags.size())
    
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)

    pca = PCA(n_components=2)
    dann_tsne = tsne.fit_transform(np.concatenate((s_feats.cpu().detach().numpy(),
                                                       t_feats.cpu().detach().numpy())))
    

    print(dann_tsne.shape)
    
    feature_extractor.train()
    class_classifier.train()
    domain_classifier.train()
    
    return dann_tsne, np.concatenate((s_tags, t_tags))

    