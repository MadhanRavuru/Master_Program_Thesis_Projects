import numpy as np

# for image stuff and visualization
import matplotlib.pyplot as plt 

# OpenCV for image processing
import cv2

# for directory/file stuff
import os


###############################################################################
## Function: create_border_mask
###############################################################################

def create_border_mask(files):
    
    for file in files:
        labels = cv2.imread(file)
        # opencv reads the color channels in reverse order :(
        labels=cv2.cvtColor(labels, cv2.COLOR_BGR2RGB)
        labels_R = labels[:,:,0]  
        h = labels_R.shape[0]
        w = labels_R.shape[1]
        m=4     #some margin to avoid the first/last row/col dilation  
        sz=2
        #tmp = labels_R
        tmp = cv2.dilate(labels_R, np.ones((5,5),np.uint8)) 
        msk1 = np.zeros_like(labels_R, dtype='uint8')
        
        # loop over the labels, but skip the first/last row/col
        for y0 in range(sz+m, h-sz-m-1):
            for x0 in range(sz+m, w-sz-m-1):
                uniq = np.unique(tmp[max(0, y0-sz):min(h, y0+sz+1), 
                                   max(0, x0-sz):min(w, x0+sz+1)])
                if uniq.shape[0]==1:
                    continue
              
                # cases where cell 1 hemo and cell 2 hemo touch
                if 220 in uniq and 165 in uniq:
                    msk1[y0,x0]=255
    
              
                # cases where cell 1 hemo and cell 3 hemo touch
                if 220 in uniq and 238 in uniq:
                    msk1[y0,x0]=255
              
                # cases where cell 2 hemo and cell 3 hemo touch
                if 165 in uniq and 238 in uniq:
                    msk1[y0,x0]=255
                
                # cases where cell 1 white and cell 2 hemo touch
                if 255 in uniq and 165 in uniq:
                    msk1[y0,x0]=255
               
              
                # cases where cell 1 white and cell 3 hemo touch
                if 255 in uniq and 238 in uniq:
                    msk1[y0,x0]=255
                
                # cases where cell 1 hemo and cell 2 white touch
                if 220 in uniq and 128 in uniq:
                    msk1[y0,x0]=255
               
              
                # cases where cell 1 hemo and cell 3 white touch
                if 220 in uniq and 225 in uniq:
                    msk1[y0,x0]=255
              
                # cases where cell 1 white and cell 2 white touch
                if 255 in uniq and 128 in uniq:
                    msk1[y0,x0]=255
              
                # cases where cell 1 white and cell 3 white touch
                if 255 in uniq and 225 in uniq:
                    msk1[y0,x0]=255
              
                # cases where cell 2 white and cell 3 white touch
                if 128 in uniq and 225 in uniq:
                    msk1[y0,x0]=255
                
                # cases where cell 2 hemo and cell 3 white touch
                if 165 in uniq and 225 in uniq:
                    msk1[y0,x0]=255  
              
                # cases where cell 2 white and cell 3 hemo touch
                if 128 in uniq and 238 in uniq:
                    msk1[y0,x0]=255
                    
                # cases where reticulocyte and cell 1 hemo touch    
                if 100 in uniq and 220 in uniq:  
                    msk1[y0,x0]=255
                    
                # cases where reticulocyte and cell 2 hemo touch    
                if 100 in uniq and 165 in uniq:  
                    msk1[y0,x0]=255
                    
                # cases where reticulocyte and cell 3 hemo touch    
                if 100 in uniq and 238 in uniq:  
                    msk1[y0,x0]=255  
                    
                # cases where WBC and cell 1 hemo touch    
                if 150 in uniq and 220 in uniq:  
                    msk1[y0,x0]=255
                    
                # cases where WBC and cell 2 hemo touch    
                if 150 in uniq and 165 in uniq:  
                    msk1[y0,x0]=255
                    
                # cases where WBC and cell 3 hemo touch    
                if 150 in uniq and 238 in uniq:  
                    msk1[y0,x0]=255  
                    
                # cases where WBC and reticulocyte touch    
                if 150 in uniq and 100 in uniq:  
                    msk1[y0,x0]=255  

                # cases where platelet and cell 1 hemo touch    
                if 240 in uniq and 220 in uniq:  
                    msk1[y0,x0]=255
                    
                # cases where platelet and cell 2 hemo touch    
                if 240 in uniq and 165 in uniq:  
                    msk1[y0,x0]=255
                    
                # cases where platelet and cell 3 hemo touch    
                if 240 in uniq and 238 in uniq:  
                    msk1[y0,x0]=255  
                    
                # cases where platelet and reticulocyte touch    
                if 240 in uniq and 100 in uniq:  
                    msk1[y0,x0]=255  

                # cases where platelet and WBC touch    
                if 240 in uniq and 150 in uniq:  
                    msk1[y0,x0]=255
                    
        msk1 = cv2.dilate(msk1, np.ones((5,5),np.uint8))
        msk1 = cv2.erode(msk1, np.ones((5,5),np.uint8))
        msk1 = cv2.dilate(msk1, np.ones((5,5),np.uint8))
        
        #msk0 = labels.astype('uint8')
        #msk0[msk0>0]=255
        #msk2=  np.zeros_like(labels, dtype='uint8')
        #tf_utils.plot_images([labels, msk1, np.stack([msk0, msk1, msk2],-1)],1,3)
        file = file.replace('targets','borders')
        out_dir, img = os.path.split(file)
        
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        name = img[0:-4]
        
        cv2.imwrite(os.path.join(out_dir, name+"_border.png"), msk1);
    

