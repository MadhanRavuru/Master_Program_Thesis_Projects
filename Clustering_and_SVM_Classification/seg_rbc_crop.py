import os
import cv2
import numpy as np
import math
import json
import glob
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from numpy import asarray
from numpy import savetxt
import pandas as pd
# ML related libraries
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from scipy.signal import argrelextrema
from scipy import interpolate, optimize

class Regions:
  def __init__(self, image):
      self.image = image
      self.num_regions = 0
      
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      # Find connected components with stats
      [num_labels, labels, stats, centroids] = cv2.connectedComponentsWithStats(gray, 8, cv2.CV_32S)
      self.num_regions = num_labels
      #print('Number of regions: ', self.num_regions)
      self.regions = labels
      self.regions_stats = stats
      self.regions_centroids = centroids
      self.region_img = np.array([])
      self.center_reg_index = -1

      self.contours = np.array([])
      self.cont_hierarchy = np.array([])
    
      self.ellipses = list()
      self.circles = list()  
      self.rotated_rectangles = list()
        
      self.matched_indx_reg_to_cnt = list()

  def set_central_region_index(self):

      if len(self.regions_centroids) > 2:
          # Height, width, number of channels in image
          height = self.image.shape[0]
          width = self.image.shape[1]
          channels = self.image.shape[2]
          img_center = [height / 2, width / 2]

          # Extract and draw bbox of the central segmented region
          dist_to_centroid = []
          for i in range(0, self.num_regions-1):
              [cx, cy] = self.regions_centroids[i]
              r2 = (img_center[0] - cx) ** 2 + (img_center[1] - cy) ** 2
              dist_to_centroid.append(r2)

          self.center_reg_index = np.argmin(dist_to_centroid)
          if self.center_reg_index == len(self.regions_centroids)-1:
              print('The central region is BACKGROUND')
              #os.sys.exit(-1)
      elif len(self.regions_centroids) == 2:
          self.center_reg_index = 0
      else:
          print('Less then one regions detected!!!')
          os.sys.exit(-1)


  def color_regions(self):
    # Map component labels to hue val
    label_hue = np.uint8(179*self.regions/np.max(self.regions))
    #print('label_hue = ', label_hue)
    blank_ch = 255*np.ones_like(label_hue)
    self.region_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    self.region_img = cv2.cvtColor(self.region_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    self.region_img[label_hue==0] = 0

    # get unique color values
    unique_reg_colors = np.unique(label_hue)
    #print(unique_reg_colors)

  def show_regions(self):
    cv2.imshow('Regions', self.region_img)
    cv2.waitKey()


  def get_contours(self):
      regions_gray = cv2.cvtColor(self.region_img, cv2.COLOR_BGR2GRAY)
      threshod = np.amax(self.regions)
      #print(threshod)
      th, th_image = cv2.threshold(regions_gray, threshod - 1, 255, cv2.THRESH_BINARY)
      #cv2.imshow('Thresholded', th_image)
      #cv2.waitKeyEx()

      cont_img = th_image.copy()
      self.contours, self.hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  def get_ellipses(self):
      valid_cnt = list()
      for ind, cnt in enumerate(self.contours):
          area = cv2.contourArea(cnt)
          '''
          print(area)
          if area < 2000 or area > 4000:
              continue
              
          '''
          if len(cnt) < 5 or area < 5:
              continue

          ellipse = cv2.fitEllipse(cnt)
          self.ellipses.append(ellipse)
          valid_cnt.append(cnt)

      self.contours = valid_cnt
      
  def draw_ellipses(self):
      for ellipse in self.ellipses:
          img_aux = self.image.copy()
          cv2.ellipse(img_aux, ellipse, (0, 255, 0), 1)

      cv2.imshow('Ellipses', img_aux)
      cv2.waitKey()
        
  def get_circles(self):
    
      def calc_R(xc, yc):         # calculate the distance of each 2D points from the center (xc, yc) 
          return np.sqrt((x-xc)**2 + (y-yc)**2)
   

      def f_2(c):                 # calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) 
          Ri = calc_R(*c)
          return Ri - Ri.mean()
    
      for ind, cnt in enumerate(self.contours):
          x =  cnt[:,0,0]
          y =  cnt[:,0,1] 
           
          center_estimate = x.mean(), y.mean()
          center_2, ier = optimize.leastsq(f_2, center_estimate)
   
          xc_2, yc_2 = center_2
          Ri_2       = calc_R(xc_2, yc_2)
          R_2        = Ri_2.mean()
    
          center = (int(xc_2), int(yc_2))
          radius = int((R_2))
          circle = center, radius  
          
          self.circles.append(circle)
          
  def draw_circles(self):
      for circle in self.circles:
          (x,y), radius = circle  
          center = (x,y)
          img_aux = self.image.copy()
          cv2.circle(img_aux, center, radius, (0,255,255), 1) 
        
      cv2.imshow('Circles', img_aux)
      cv2.waitKey()  
            
  def get_rotated_rectangles(self):
      for ind, cnt in enumerate(self.contours):
          rect = cv2.minAreaRect(cnt)
          self.rotated_rectangles.append(rect)
        
  def draw_rotated_rectangles(self):  
      for rect in self.rotated_rectangles:
          box = cv2.boxPoints(rect)
          box = np.int0(box)
          img_aux = self.image.copy() 
          cv2.drawContours(img_aux,[box], 0, (0,0,255), 1)
        
      cv2.imshow('Rotated rectangles', img_aux)
      cv2.waitKey()  
        
  def draw_bbox_center_region(self):
      if self.center_reg_index == -1:
          self.set_central_region_index(self)

      i = self.center_reg_index
      # Draw a rectangle with blue line borders of thickness of 2 px
      top_left_point = np.array([self.regions_stats[i,cv2.CC_STAT_LEFT],
                                 self.regions_stats[i,cv2.CC_STAT_TOP]])
      end_right_point = top_left_point + np.array([self.regions_stats[i,cv2.CC_STAT_WIDTH],
                                                   self.regions_stats[i,cv2.CC_STAT_HEIGHT]])
      img_aux = self.image.copy()
      img = cv2.rectangle(img_aux, tuple(top_left_point), tuple(end_right_point), (255, 0, 0), 1)
      #cv2.imshow('Detected central', img)
      #cv2.waitKey()

  def remove_non_central_regions(self):
      if self.center_reg_index == -1:
          self.set_central_region_index(self)

      self.regions[self.regions != self.center_reg_index] = 0
      print('Before removal :', self.regions_stats)
      self.regions_stats = self.regions_stats[self.center_reg_index]
      print('After removal :', self.regions_stats)
      self.num_regions = 1

  def match_region_to_contour_indices(self, cnt_centroids):
      for ind_r, reg_C in enumerate(self.regions_centroids):
          if ind_r == 0:
              self.matched_indx_reg_to_cnt.append(len(self.regions_centroids)-1)
              continue
          reg_cX = reg_C[0]
          reg_cY = reg_C[1]
          minDist = math.inf
          minIndex = -1
          for ind_c, cnt_C in enumerate(cnt_centroids):
              cnt_cX = cnt_C[0]
              cnt_cY = cnt_C[1]
              dist = math.sqrt((reg_cX-cnt_cX)**2 + (reg_cY-cnt_cY)**2)
              if dist < minDist:
                  minDist = dist
                  minIndex = ind_c

          if minIndex not in self.matched_indx_reg_to_cnt:
            self.matched_indx_reg_to_cnt.append(minIndex)
          else:
            print('Warrning: Mathched region to contour index repeates!')
              #os.sys.exit(-1)
          #ind_c = ind_c + 1

      #print('Matched region to contour indices: ', self.matched_indx_reg_to_cnt)


  def update_region_labels(self):
      aux_regions = 1000*np.ones_like(self.regions)
      for reg_indx, cnt_indx in enumerate(self.matched_indx_reg_to_cnt):
        aux_regions[self.regions == reg_indx] = cnt_indx

      self.regions = aux_regions

  def update_region_stats(self):
      aux_regions_stats = np.ones_like(self.regions_stats)
      #print('Before :', self.regions_stats)
      for reg_indx, cnt_indx in enumerate(self.matched_indx_reg_to_cnt):
        aux_regions_stats[cnt_indx] = self.regions_stats[reg_indx]

      self.regions_stats = aux_regions_stats
      #print('After :', self.regions_stats)

  def update_region_centroids(self):
      aux_regions_centroids =  np.ones_like(self.regions_centroids)
      #print('Before :', self.regions_centroids)
      for reg_indx, cnt_indx in enumerate(self.matched_indx_reg_to_cnt):
        aux_regions_centroids[cnt_indx] = self.regions_centroids[reg_indx]

      self.regions_centroids = aux_regions_centroids
      #print('After :', self.regions_centroids)

  def update_regions(self):
      self.update_region_labels()
      self.update_region_stats()
      self.update_region_centroids()


class ContourAttributes:
    def __init__(self, regions):
        self.cntarea = list()
        self.regarea = list()
        self.cntcenter = list()
        self.circularity = list()
        self.completeness = list()
        self.anisometry = list()
        self.bulkiness = list()
        self.struct_factor = list()
        self.roundness = list()
        
        self.extent = list() 
        self.circularity_ = list()
        self.solidity = list()
        self.convexity = list()
        self.elongation = list()
        self.compactness_ = list()
        
        self.cnt_huMoments_gray = list()        
        self.cnt_huMoments_binary = list()        
        self.cnt_huMoments_inv_gray = list()        
        
        self.geometric_features_ellipse = list()
        self.geometric_features_circle = list()
        self.geometric_features_rotated_rect = list()
        
        self.boundary_features = list()
        
        self.regions = regions
        self.reg_means_gray = list()
        self.reg_stddevs_gray  = list()
        self.reg_means_RGB = list()
        self.reg_stddevs_RGB = list()
        self.reg_means_HSV = list()
        self.reg_stddevs_HSV = list()

    def compute_contour_area(self):
        for cnt in self.regions.contours:
            cnt_area = cv2.contourArea(cnt)
            if cnt_area >= 5:
                self.cntarea.append(cnt_area)

    def compute_region_area(self):
        if self.regions.num_regions != 1:
            for i in range(0,self.regions.num_regions):
                area = self.regions.regions_stats[i,cv2.CC_STAT_AREA]
                self.regarea.append(area)
        else:
            area = self.regions.regions_stats[cv2.CC_STAT_AREA]
            self.regarea.append(area)
   
    def compute_contour_centroids(self):
        for cnt in self.regions.contours:
            # compute the center of the contour
            M = cv2.moments(cnt)
            if M["m00"] <= 10e-10:
                bbox = cv2.boundingRect(cnt)
                bcX = bbox[0] + bbox[2] / 2;
                bcY = bbox[1] + bbox[3] / 2;
                #print('Bbox(cX,cY) = ', bcX, bcY)
                cX = bcX
                cY = bcY
            else:
                ccX = int(M["m10"] / M["m00"])
                ccY = int(M["m01"] / M["m00"])
                cX = ccX
                cY = ccY
                #print('Cnt(cX,cY) = ', ccX, ccY)

            self.cntcenter.append([cX,cY])

    def compute_circularity(self):
        for i, contour in enumerate(self.regions.contours):
            F = self.cntarea[i]
            C = self.cntcenter[i]
            max_dist = 0
            for j in range(0,len(contour)):
                dist_cnt_point = math.sqrt((C[0] - contour[j,0,0])**2 + (C[1] - contour[j,0,1])**2)
                if  dist_cnt_point > max_dist:
                    max_dist = dist_cnt_point
            C1 = F/(math.pi*max_dist**2)
            circularity = min(1,C1)
            self.circularity.append(circularity)

    def compute_compactness(self):
        for i, contour in enumerate(self.regions.contours):
            F = self.cntarea[i]
            L = cv2.arcLength(contour, closed=True)
            C = L ** 2 / (4*math.pi*max(F, 10e-7))                   
            compactness = max(1, C)
            self.completeness.append(compactness)

    def compute_eccentricity(self):
        for i, ellipse in enumerate(self.regions.ellipses):
            A = self.cntarea[i]
            (x, y), (Ra, Rb), angle = ellipse                    
            anisometry = max(Ra,Rb)/min(Ra,Rb)                   
            self.anisometry.append(anisometry)
            bulkiness = (math.pi*Ra*Rb)/(4*A)                
            self.bulkiness.append(bulkiness)
            struct_fact=max(0,anisometry*bulkiness-1)
            self.struct_factor.append(struct_fact)

    def compute_roundness(self):
        for i, contour in enumerate(self.regions.contours):
            F = len(contour)
            C = self.cntcenter[i]
            dist_cnt_point = 0.0
            for j in range(0,len(contour)):
                dst = math.sqrt((C[0] - contour[j,0,0])**2 + (C[1] - contour[j,0,1])**2)
                dist_cnt_point = dist_cnt_point + dst
            Distance = dist_cnt_point / F
            var_cnt_point = 0.0
            for j in range(0, len(contour)):
                dst = math.sqrt((C[0] - contour[j, 0, 0]) ** 2 + (C[1] - contour[j, 0, 1]) ** 2)
                var_cnt_point = var_cnt_point + (dst-Distance)**2

            Sigma2 = var_cnt_point/F
            roundness = 1 - math.sqrt(Sigma2)/Distance
            self.roundness.append(roundness)
            
    def compute_shape_factors(self):                                           # newly added shape factors
        for i, contour in enumerate(self.regions.contours):
            F = self.cntarea[i]
            x,y,w,h = cv2.boundingRect(contour)
            rect_area = w*h
            extent = float(F) / rect_area
            self.extent.append(extent)
            
            
            L = cv2.arcLength(contour, closed=True)
            circularity = math.sqrt((4*math.pi*max(F, 10e-7)) / L ** 2)      # sqrt of form factor
            self.circularity_.append(circularity)
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(F) / hull_area
            self.solidity.append(solidity)
            
            L_hull = cv2.arcLength(hull, closed=True)
            convexity = L_hull / L
            self.convexity.append(convexity)        
            
        for i, rect in enumerate(self.regions.rotated_rectangles): 
            (x, y), (w, h), angle = rect
            Max_Feret_Diameter = max(w,h)
            Min_Feret_Diameter = min(w,h)
            elongation = 1 - (Min_Feret_Diameter / Max_Feret_Diameter)
            self.elongation.append(elongation)
            
            F = self.cntarea[i]
            compactness = math.sqrt((4*F)/math.pi) / Max_Feret_Diameter
            self.compactness_.append(compactness)
            
    def compute_contour_huMoments(self):                                    # Absolute log transformed hu-moments considered

            image = self.regions.image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            

            # Calculate Hu Moments grayscale image
            moments = cv2.moments(gray)
            huMoments_gray = cv2.HuMoments(moments)
            for i in range(0,7):
                huMoments_gray[i] = -1* math.copysign(1.0, huMoments_gray[i]) * math.log10(abs(huMoments_gray[i]))
            self.cnt_huMoments_gray.append(np.abs(huMoments_gray))
            
            # Calculate Hu Moments binary image
            _,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
            moments = cv2.moments(thresh, True)
            huMoments_binary = cv2.HuMoments(moments)
            for i in range(0,7):
                huMoments_binary[i] = -1* math.copysign(1.0, huMoments_binary[i]) * math.log10(abs(huMoments_binary[i]))
            self.cnt_huMoments_binary.append(np.abs(huMoments_binary))
            
            # Calculate Hu Moments inverted gray image, excluding background
            inv = 192 - gray
            inv[inv==192] = 0 
            moments = cv2.moments(inv)
            huMoments_inv = cv2.HuMoments(moments)
            for i in range(0,7):
                huMoments_inv[i] = -1* math.copysign(1.0, huMoments_inv[i]) * math.log10(abs(huMoments_inv[i]))
            self.cnt_huMoments_inv_gray.append(np.abs(huMoments_inv)) 
               
    def compute_goodness_of_fit(self, shape, contour):
        if (len(shape) == 3):
            (x, y), (Ra, Rb), angle = shape
     
            angle = (angle/180)* math.pi;
            sum_GOF = 0.0
            for j in range(0,len(contour)):
                posx = (contour[j,0,0] - x)* np.cos(-angle) - (contour[j,0,1] - y)* np.sin(-angle)
                posy = (contour[j,0,0] - x)* np.sin(-angle) + (contour[j,0,1] - y)* np.cos(-angle)   
                sum_GOF += np.abs( (posx/max(Ra,Rb))**2 + (posy/min(Ra,Rb))**2 - 0.25 )
            mean_GOF = sum_GOF/len(contour)                     
            return mean_GOF
    
        if (len(shape) == 2):
            (x,y), radius = shape
     
            sum_GOF = 0.0
            for j in range(0,len(contour)):
                posx = contour[j,0,0] - x
                posy = contour[j,0,1] - y 
                sum_GOF += np.abs( (posx/radius)**2 + (posy/radius)**2 - 1.0)
            mean_GOF = sum_GOF/len(contour)    
            return mean_GOF
 
    def compute_geometric_features(self):
        for i, ellipse in enumerate(self.regions.ellipses):
            (x, y), (Ra, Rb), angle = ellipse
            area = math.pi*Ra*Rb / 4
            eccentricity = math.sqrt(1- (min(Ra,Rb)/max(Ra,Rb))**2)
            perimeter = 2* math.pi* math.sqrt((Ra**2 + Rb**2)/8)
            gof = self.compute_goodness_of_fit(ellipse, self.regions.contours[i])
            #print(gof)
            ellipse_features = [ eccentricity, max(Ra,Rb), min(Ra,Rb), area, perimeter, gof]
            self.geometric_features_ellipse.append(ellipse_features)
            
        for i, circle in enumerate(self.regions.circles):     
            (x,y), radius = circle
            area = math.pi * (radius**2)
            gof = self.compute_goodness_of_fit(circle, self.regions.contours[i])
            circle_features = [radius, area, gof]
            #print(gof)
            self.geometric_features_circle.append(circle_features)
            
        for i, rect in enumerate(self.regions.rotated_rectangles): 
            (x, y), (w, h), angle = rect
            Max_Feret_Diameter = max(w,h)
            Min_Feret_Diameter = min(w,h)
            rectangle_features = [Max_Feret_Diameter, Min_Feret_Diameter]
            self.geometric_features_rotated_rect.append(rectangle_features)
    
    def compute_boundary_features(self):
        for i, contour in enumerate(self.regions.contours):
            x = contour[:,0,0].tolist()
            y = (-contour[:,0,1]).tolist()
            x.append(contour[0,0,0])                                      # closing the contour
            y.append(-contour[0,0,1])
            tck, _ = interpolate.splprep([x, y], u=None, per=1)        #default degree(k) =3, per = 1 indicates shape is closed
                                                                  # default: good smoothing (s =0 overfits to all points)
            u = np.linspace(0,1,500)                  # 500 uniformly spaced points
            out = interpolate.splev(u,tck)
            
            dx_dt = np.gradient(out[0])      # centered difference
            dy_dt = np.gradient(out[1])
            d2x_dt2 = np.gradient(dx_dt)
            d2y_dt2 = np.gradient(dy_dt)
            curvature = (dx_dt * d2y_dt2 - d2x_dt2 * dy_dt) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
            
            curv = curvature.copy()
            curv[(-0.2 < curv) & (curv < 0.2)] = 0
            num_protrusions =  len(argrelextrema(curv, np.greater, order=5, mode='wrap')[0])    # 10 pixels neighbourhood
            num_indentations = len(argrelextrema(curv, np.less, order=5, mode='wrap')[0])
            
            cubic_spline_features = [curvature.mean(), curvature.std(), max(curvature), min(curvature), num_protrusions, num_indentations]
            self.boundary_features.append(cubic_spline_features)
            
    def compute_contour_regions_mean_stddev_gray(self):
        image = self.regions.image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for i, contour in enumerate(self.regions.contours):
            # construct a mask for the contour, then compute the
            # average gray value for the masked region
            mask = np.zeros(gray_image.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mask = cv2.erode(mask, None, iterations=2)
            [mean, stddev] = cv2.meanStdDev(gray_image, mask=mask)
            self.reg_means_gray.append(mean)
            self.reg_stddevs_gray.append(stddev)
            #print('Mean grayscale value of region ', i, ' is ', mean, ' and std dev is : ', stddev)

    def compute_contour_regions_mean_stddev_rgb(self):
        image = self.regions.image
        for i, contour in enumerate(self.regions.contours):
            # construct a mask for the contour, then compute the
            # average gray value for the masked region
            mask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mask = cv2.erode(mask, None, iterations=2)
            [mean, stddev] = cv2.meanStdDev(image, mask=mask)
            self.reg_means_RGB.append(mean)
            self.reg_stddevs_RGB.append(stddev)
            #print('Mean RGB value of region ', i, ' is ', mean, ' and std dev is : ', stddev)

    def compute_contour_regions_mean_stddev_hsv(self):
        image = self.regions.image
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for i, contour in enumerate(self.regions.contours):
            # construct a mask for the contour, then compute the
            # average gray value for the masked region
            mask = np.zeros(hsv_image.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mask = cv2.erode(mask, None, iterations=2)
            [mean, stddev] = cv2.meanStdDev(hsv_image, mask=mask)
            self.reg_means_HSV.append(mean)
            self.reg_stddevs_HSV.append(stddev)
            #print('Mean HSV value of region ', i, ' is ', mean, ' and std dev is : ', stddev)



def extract_region_atributs_from_json(in_json_file, crop_margin):
    with open(in_json_file, 'r', encoding='utf-8-sig') as read_json:
        content_json = json.load(read_json)

    images_in_json = content_json['_via_img_metadata']
    shape_attributes = []
    color_attribues = []
    for i, image in enumerate(images_in_json.keys()):
        print(i, '. Image: ', image)
        if i >= 15:
            continue
        list_of_regions = list()
        image_path = os.path.dirname(in_json_file) + '/' + content_json['_via_img_metadata'][image]['filename']
        img = cv2.imread(image_path)
        height = img.shape[0]
        width = img.shape[1]
        
        #cv2.imshow('Image', img)
        #print('Opened : ', image_path)
        #cv2.waitKey(-1)
        img2 = img.copy()
        regions = content_json['_via_img_metadata'][image]['regions']
        for region in regions:
            if region['region_attributes']['Class'] == 'RBC':
                reg_shape = region['shape_attributes']
                x1 = reg_shape['x']
                y1 = reg_shape['y']
                x2 = reg_shape['x']+reg_shape['width']
                y2 = reg_shape['y']+reg_shape['height']
                xc = x1 + int((x2-x1)/2)
                yc = y1 + int((y2-y1)/2)
                xc1 = int(xc - crop_margin)
                yc1 = int(yc - crop_margin)
                xc2 = int(xc + crop_margin)
                yc2 = int(yc + crop_margin)
                #cv2.waitKey(-1)
                if (xc1>50 and xc1 < width-50 and yc1>50 and yc1<height-50) and (xc2>50 and xc2 < width-50 and yc2>50 and yc2<height-50):
                    cv2.rectangle(img2, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.imshow('Image', img2)
                    img_crop = img[yc1:yc2, xc1:xc2]
                    matched = extract_region_attributes(img_crop)
                    if matched == [-1, -1]:
                        continue
                    #print(matched)
                    shape_attributes.append(matched[0])
                    color_attribues.append(matched[1])
                    #print('______________________________________________________________________________')
                    #cv2.imshow('Crop', img_crop)
                    #cv2.waitKey(-1)

    return shape_attributes, color_attribues

def make_cropped_region_coords(x_ur, y_ur, x_ll, y_ll, img_width, img_height, crop_margin):
    xc = x_ur + int((x_ll - x_ur) / 2)
    yc = y_ur + int((y_ll - y_ur) / 2)
    xn_ur = int(xc - crop_margin)
    yn_ur = int(yc - crop_margin)
    xn_ll = int(xc + crop_margin)
    yn_ll = int(yc + crop_margin)

    if xn_ur < 0:
        xn_ur = 0

    if yn_ur < 0:
        yn_ur = 0

    if xn_ll > img_width:
        xn_ll = img_width

    if yn_ll > img_height:
        yn_ll = img_height

    return xn_ur, yn_ur, xn_ll, yn_ll

def extract_and_classify_region_atributs_from_json(in_json_file, crop_margin, scaling, SVM_model):
    with open(in_json_file, 'r', encoding='utf-8-sig') as read_json:
        content_json = json.load(read_json)

    images_in_json = content_json['_via_img_metadata']
    shape_attributes = []
    color_attribues = []
    for i, image in enumerate(images_in_json.keys()):
        print(i, '. Image: ', image)
        # Print check
        #if i >= 15:
        #    continue
        list_of_regions = list()
        image_path = os.path.dirname(in_json_file) + '/' + content_json['_via_img_metadata'][image]['filename']
        img = cv2.imread(image_path)
        height = img.shape[0]
        width = img.shape[1]

        #cv2.imshow('Image', img)
        #print('Opened : ', image_path)
        #cv2.waitKey(-1)
        img2 = img.copy()
        regions = content_json['_via_img_metadata'][image]['regions']
        for region in regions:
            if region['region_attributes']['Class'] == 'RBC':
                reg_shape = region['shape_attributes']
                x1 = reg_shape['x']
                y1 = reg_shape['y']
                x2 = reg_shape['x']+reg_shape['width']
                y2 = reg_shape['y']+reg_shape['height']
                xc1, yc1, xc2, yc2 = make_cropped_region_coords(x1, y1, x2, y2, width, height, crop_margin)

                #cv2.waitKey(-1)
                #if (xc1>50 and xc1 < width-50 and yc1>50 and yc1<height-50) and (xc2>50 and xc2 < width-50 and yc2>50 and yc2<height-50):
                cv2.rectangle(img2, (int(xc1), int(yc1)), (int(xc2), int(yc2)), (255, 0, 0), 2)
                #cv2.imshow('Image', img2)
                img_crop = img[yc1:yc2, xc1:xc2]
                matched = extract_region_attributes(img_crop)
                if matched != [-1, -1]:
                    # Classify
                    X_test = np.concatenate([matched[0], matched[1]])
                    X_test_reshape = X_test.reshape(1,-1)
                    X_test = scaling.transform(X_test_reshape)
                    y_pred = SVM_model.predict(X_test)
                    if y_pred == 0.0:
                        region['region_attributes']['Subclass RBC'] = 'NOR'

                #cv2.imshow('Crop', img_crop)
                #cv2.waitKey(-1)
    out_filepath = in_json_file[:in_json_file.find('_GT.json')]+'_SubClass_GT.json'
    with open(out_filepath, 'w') as outfile:
        json.dump(content_json, outfile, indent=4)

    return shape_attributes, color_attribues

def load_crops_and_compute_features(base_dir):
    normal_rbc_crops = glob.glob(base_dir + '/normal_RBCs/*/*processed.png')
    abnormal_rbc_crops = glob.glob(base_dir + '/abnormal_RBCs/*/*processed.png')

    # Load and extract attributes of positive RBC samples
    image_paths = []
    shape_attributes = []
    color_attribues = []
    shape_color_attributes = []
    skipped_data_samples = 0
    for i, img_path in enumerate(normal_rbc_crops):
        img = cv2.imread(img_path)
        image_paths.append(img_path)
        #print(i, '.', img_path)
        matched = extract_region_attributes(img)
        if matched == [-1, -1]:
            #print('Skipping data sample!')
            skipped_data_samples = skipped_data_samples + 1
            continue
        shape_attributes.append(matched[0])
        color_attribues.append(matched[1])
        all_matched  = np.concatenate([matched[0], matched[1]])
        shape_color_attributes.append(all_matched)

    print('Total number of skipped positive data samples : ', skipped_data_samples, ' out of (',len(normal_rbc_crops),')')

    # Prepare and save shape attributes for positive samples (normal RBCs)
    X1_shape = np.array(shape_attributes)
    X_shape_normal = np.c_[ X1_shape, np.zeros(X1_shape.shape[0]) ]
    shape_data = asarray(X_shape_normal)
    savetxt('shape_attrib_normal_rbc.csv', shape_data, delimiter=',')

    # Prepare and save color attributes for positive samples (normal RBCs)
    X1_color = np.array(color_attribues)
    X_color_normal = np.c_[ X1_color, np.zeros(X1_color.shape[0]) ]
    color_data = asarray(X_color_normal)
    savetxt('color_attrib_normal_rbc.csv', color_data, delimiter=',')

    # Prepare and save shape and color attributes for positive samples (normal RBCs)
    X1_shape_color = np.array(shape_color_attributes)
    X_shape_color_normal = np.c_[X1_shape_color, np.zeros(X1_shape_color.shape[0])]
    shape_color_data = asarray(X_shape_color_normal)
    savetxt('shape_color_attrib_normal_rbc.csv', shape_color_data, delimiter=',')

    # Load and extract attributes of negative RBC samples
    shape_attributes = []
    color_attribues = []
    shape_color_attributes = []
    skipped_data_samples = 0
    for i, img_path in enumerate(abnormal_rbc_crops):
        img = cv2.imread(img_path)
        image_paths.append(img_path)
        matched = extract_region_attributes(img)
        if matched == [-1, -1]:
            #print('Skipping data sample!')
            skipped_data_samples = skipped_data_samples + 1
            continue
        shape_attributes.append(matched[0])
        color_attribues.append(matched[1])
        all_matched = np.concatenate([matched[0], matched[1]])
        shape_color_attributes.append(all_matched)

    print('Total number of skipped negative data samples : ', skipped_data_samples, ' out of (',len(abnormal_rbc_crops),')')

    # Prepare and save shape attributes for negative samples (abnormal RBCs)
    X2_shape = np.array(shape_attributes)
    X_shape_abnormal = np.c_[X2_shape, np.ones(X2_shape.shape[0])]
    shape_data = asarray(X_shape_abnormal)
    savetxt('shape_attrib_abnormal_rbc.csv', shape_data, delimiter=',')

    # Prepare and save color attributes for negative samples (abnormal RBCs)
    X2_color = np.array(color_attribues)
    X_color_abnormal = np.c_[X2_color, np.ones(X2_color.shape[0])]
    color_data = asarray(X_color_abnormal)
    savetxt('color_attrib_abnormal_rbc.csv', color_data, delimiter=',')

    # Prepare and save shape and color attributes for negative samples (abnormal RBCs)
    X2_shape_color = np.array(shape_color_attributes)
    X_shape_color_abnormal = np.c_[X2_shape_color, np.ones(X2_shape_color.shape[0])]
    shape_color_data = asarray(X_shape_color_abnormal)
    savetxt('shape_color_attrib_abnormal_rbc.csv', shape_color_data, delimiter=',')

    # Prepare and save complete labeled dataset
    X_all_shape = np.concatenate((X_shape_normal, X_shape_abnormal))
    data = asarray(X_all_shape)
    savetxt('shape_rbc.csv', data, delimiter=',')

    X_all_color = np.concatenate((X_color_normal, X_color_abnormal))
    data = asarray(X_all_color)
    savetxt('color_rbc.csv', data, delimiter=',')

    X_all = np.concatenate((X_shape_color_normal, X_shape_color_abnormal))
    data = asarray(X_all)
    savetxt('shape_color_rbc.csv', data, delimiter=',')
    
    return image_paths


def extract_region_attributes(img):
   
    image_regions_c = Regions(img)
    

    # Convert region labels to colors and show regions
    image_regions_c.color_regions()
    #image_regions_c.show_regions()
    # print('Region centroids: ', image_regions_c.regions_centroids)

    # Extract contours in colored image, get ellipses and show them
    image_regions_c.get_contours()
    image_regions_c.get_ellipses()
    image_regions_c.get_circles()
    image_regions_c.get_rotated_rectangles()
    
    # Compute contour attributes
    cnt_attribs = ContourAttributes(image_regions_c)
    cnt_attribs.compute_contour_area()
    cnt_attribs.compute_contour_centroids()
    # print('Contour centroids : ', cnt_attribs.cntcenter)
    # print('Contour areas : ', cnt_attribs.cntarea)


    # Update region labels and properties so to match indices of contours
    image_regions_c.match_region_to_contour_indices(cnt_attribs.cntcenter)
    image_regions_c.update_regions()
    #print('Region centroids: ', image_regions_c.regions_centroids)
    cnt_attribs.compute_region_area()
    #print('Region area = ', cnt_attribs.regarea)

    # Extract central region/contour index
    image_regions_c.set_central_region_index()
    # image_regions_c.remove_non_central_regions()

    # Show regions and central's bbox
    #image_regions_c.show_regions()
    image_regions_c.draw_bbox_center_region()

    # Compute contour properties
    cnt_attribs.compute_circularity()
    cnt_attribs.compute_compactness()
    cnt_attribs.compute_eccentricity()
    cnt_attribs.compute_roundness()
    
    cnt_attribs.compute_shape_factors()
    
    cnt_attribs.compute_contour_huMoments()
    
    cnt_attribs.compute_geometric_features()
    cnt_attribs.compute_boundary_features()
    
    cnt_attribs.compute_contour_regions_mean_stddev_gray()
    cnt_attribs.compute_contour_regions_mean_stddev_rgb()
    cnt_attribs.compute_contour_regions_mean_stddev_hsv()

    # Print contour properties
    c_index = image_regions_c.center_reg_index
    # print('c_index = ', c_index)
    # if c_index == 1:
    #     c_index = 0
    # elif c_index == len(cnt_attribs.cntarea):
    #     c_index = len(cnt_attribs.cntarea) - 1

    # print('Contour area = ', cnt_attribs.cntarea)
    # print('Region  area = ', cnt_attribs.regarea)
    # print('Circularity = ', cnt_attribs.circularity)
    # print('Compactness = ', cnt_attribs.completeness)
    # print('Anisometry = ', cnt_attribs.anisometry)
    # print('Bulkiness = ', cnt_attribs.bulkiness)
    # print('Structure factor = ', cnt_attribs.stuct_factor)
    # print('Roundness = ', cnt_attribs.roundness)

    region_shape_attributes = np.empty(39)
    shape_attributes = {
        'CONTOUR_AREA'     : cnt_attribs.cntarea[c_index],
        'REGION_AREA'      : cnt_attribs.regarea[c_index],
        #'CIRCULARITY'      : cnt_attribs.circularity[c_index],
        #'COMPACTNESS'      : cnt_attribs.completeness[c_index],
        #'ANISOMETRY'       : cnt_attribs.anisometry[c_index],
        #'BULKINESS'        : cnt_attribs.bulkiness[c_index],
        #'STRUCTURE_FACTOR' : cnt_attribs.struct_factor[c_index],
        #'ROUNDNESS'        : cnt_attribs.roundness[c_index],        
        'EXTENT'           : cnt_attribs.extent[c_index],                    # newly added shape factors
        'CIRCULARITY_'     : cnt_attribs.circularity_[c_index],
        'SOLIDITY'         : cnt_attribs.solidity[c_index],
        'CONVEXITY'        : cnt_attribs.convexity[c_index],
        'ELONGATION'       : cnt_attribs.elongation[c_index],
        'COMPACTNESS_'     : cnt_attribs.compactness_[c_index]
    }
    
    hu_moments = {
        'HU_MOMENTS_GRAY'       : cnt_attribs.cnt_huMoments_gray[c_index],
        'HU_MOMENTS_BINARY'     : cnt_attribs.cnt_huMoments_binary[c_index],
       # 'HU_MOMENTS_INV_GRAY'   : cnt_attribs.cnt_huMoments_inv_gray[c_index]
    }
    
    geometric_and_boundary_atributes = {
        'ELLIPSE_FIT'      : cnt_attribs.geometric_features_ellipse[c_index],
        'CIRCLE_FIT'       : cnt_attribs.geometric_features_circle[c_index],
        'RECTANGLE_FIT'    : cnt_attribs.geometric_features_rotated_rect[c_index],
        'CUBIC_SPLINE_FIT' : cnt_attribs.boundary_features[c_index]
    }
    
    color_attributes = {
        'MEAN_GRAY'        : cnt_attribs.reg_means_gray[c_index],
        'STD_GRAY'         : cnt_attribs.reg_stddevs_gray[c_index],
        'MEAN_RGB'         : cnt_attribs.reg_means_RGB[c_index],
        'STD_RGB'          : cnt_attribs.reg_stddevs_RGB[c_index],
        'MEAN_HSV'         : cnt_attribs.reg_means_HSV[c_index],
        'STD_HSV'          : cnt_attribs.reg_stddevs_HSV[c_index]
    }

    for ind, key in enumerate(shape_attributes.keys()):
        region_shape_attributes[ind] = shape_attributes[key]
    
    ind = 8
    for key in hu_moments.keys():
        for elem in hu_moments[key]:
            region_shape_attributes[ind] = elem
            ind = ind + 1
            
    ind = 22        
    for key in geometric_and_boundary_atributes.keys():
        for elem in geometric_and_boundary_atributes[key]:
            region_shape_attributes[ind] = elem
            ind = ind + 1
            
    region_color_attributes = np.empty(14)
    ind = 0
    for key in color_attributes.keys():
        for elem in color_attributes[key]:
            region_color_attributes[ind] = elem
            ind = ind + 1

    return region_shape_attributes, region_color_attributes
  

def do_kmeans(X1, attrib_type='shape', attrib_indx=[0,1,2]):
    # KMeans
    shape_attrib_labels = ['CONTOUR_AREA', 'REGION_AREA', 'CIRCULARITY', 'COMPACTNESS', 'ANISOMETRY', 'BULKINESS', 'STRUCTURE_FACTOR', 'ROUNDNESS']
    color_attrib_labels = ['MEAN_GRAY', 'STD_GRAY', 'MEAN_R', 'MEAN_G', 'MEAN_B', 'STD_R', 'STD_G', 'STD_B', 'MEAN_H', 'MEAN_S', 'MEAN_V', 'STD_H', 'STD_S', 'STD_V']

    if attrib_type == 'shape':
        attrib_labels = [shape_attrib_labels[i] for i, lab in enumerate(shape_attrib_labels) if i in attrib_indx]

    if attrib_type == 'color':
        attrib_labels = [color_attrib_labels[i] for i, lab in enumerate(color_attrib_labels) if i in attrib_indx]

    # standardize the data attributes
    X = preprocessing.scale(X1)
    km = KMeans(n_clusters=2)
    km.fit(X)
    km.predict(X)
    labels = km.labels_
    # Plotting
    fig = plt.figure(1, figsize=(7, 7))
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
    ax.scatter(X[:, attrib_indx[0]], X[:, attrib_indx[1]], X[:, attrib_indx[2]],
               c=labels.astype(np.float), edgecolor="k", s=50)
    ax.set_xlabel(attrib_labels[0])
    ax.set_ylabel(attrib_labels[1])
    ax.set_zlabel(attrib_labels[2])
    plt.title("K Means", fontsize=14)
    plt.show()

all_regions = False
center_region = False
unsupervised = False
supervised = True
retrain = False
use_train_test_split = False

# Unsupervised clustering of RBCs to normal and other cell subclasses
if unsupervised:
    in_json = 'D:/data/ISR/ISR_Leica_Scans/RBC_ClassDB/train/EH5_Siemens003/LE_40X_BF_CEL_DS_EH5_Siemens003_Slide2/JAI/LE_40X_BF_CEL_DS_EH5_Siemens003_Slide2_RBC_YOLO_RBC_HAL_GT.json'
    #in_json = '/mnt/Data/isr/ISR_Leica_Scans/RBC_ClassDB/train/EH5_Siemens003/LE_40X_BF_CEL_DS_EH5_Siemens003_Slide2/JAI/LE_40X_BF_CEL_DS_EH5_Siemens003_Slide2_RBC_GT_HAL_YOLO.json'
    [reg_attr, col_attr] = extract_region_atributs_from_json(in_json, crop_margin=50)
    print('Region Attributes \n')
    X1 = np.array(col_attr)
    do_kmeans(X1, attrib_type='color', attrib_indx=[2, 3, 4])
elif supervised:
    if retrain:
        #RBC_DB_path = 'D:/data/ISR/ISR_Leica_Scans/RBC_ClassDB/' #'/mnt/Data/isr/ISR_Leica_Scans/RBC_ClassDB/'
        #load_crops_and_compute_features(RBC_DB_path)
        data = pd.read_csv('shape_color_rbc.csv')
        # Pandas ".iloc" expects row_indexer, column_indexer
        X = data.iloc[:, :-1].values
        print('Before scaling: ', X)
        scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X)
        X = scaling.transform(X)
        print('After scaling: ', X)
        #X = data.iloc[:, [0,1,2,3,4,5,6,7]].values
        # Now let's tell the dataframe which column we want for the target/labels.
        y = data.iloc[:,-1:].values
        filename = 'svm_scaling.sav'
        pickle.dump(scaling, open(filename, 'wb'))

        if use_train_test_split:
            # Test size specifies how much of the data you want to set aside for the testing set.
            # Random_state parameter is just a random seed we can use.
            # You can use it if you'd like to reproduce these specific results.
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)
            print('Training set size: ', X_train.shape)
            print('Test set size: ', X_test.shape)

            # Train SVM classifier
            svclassifier = SVC(kernel='linear')
            svclassifier.fit(X_train, y_train)

            # Save the model to disk
            filename = 'svm_train_model.sav'
            pickle.dump(svclassifier, open(filename, 'wb'))

            # Make predictions
            y_pred = svclassifier.predict(X_test)
            print(y_pred)

            # prediction results
            print('SVM Accuracy Score: ')
            print(accuracy_score(y_pred, y_test))
            print('SVM Confusion Matrix:')
            print(confusion_matrix(y_test, y_pred))
            print('SVM Classification Report:')
            print(classification_report(y_test, y_pred))

            # KNN classifier
            # KNN model requires you to specify n_neighbors,
            # the number of points the classifier will look at to determine what class a new point belongs to
            KNN_model = KNeighborsClassifier(n_neighbors=5)
            KNN_model.fit(X_train, y_train)
            print('KNN Accuracy Score: ')
            KNN_prediction = KNN_model.predict(X_test)
            print(accuracy_score(KNN_prediction, y_test))
            print('KNN Confusion Matrix:')
            print(confusion_matrix(KNN_prediction, y_test))
            print('KNN Classification Report:')
            print(classification_report(KNN_prediction, y_test))

            # Logistic Regression
            logreg_clf = LogisticRegression()
            logreg_clf.fit(X_train, y_train)
            logreg_y_pred = logreg_clf.predict(X_test)
            print('Logistic Regression Accuracy Score: ')
            print(accuracy_score(logreg_y_pred, y_test))
            print('Logistic Regression Confusion Matrix:')
            print(confusion_matrix(y_test, logreg_y_pred))
            print('Logistic Regression Classification Report:')
            print(classification_report(y_test, logreg_y_pred))
        else: # Use all data for training and test on all json files
            X_train = X
            y_train = y

            # Train SVM classifier
            svclassifier = SVC(kernel='linear')
            svclassifier.fit(X_train, y_train)

            # Save the model to disk
            filename = 'svm_all_model.sav'
            pickle.dump(svclassifier, open(filename, 'wb'))

    else:
        if use_train_test_split:
            # load the svm model from disk
            filename = 'svm_train_model.sav'
            loaded_model = pickle.load(open(filename, 'rb'))

            data = pd.read_csv('shape_color_rbc.csv')
            X = data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values
            y = data.iloc[:, -1:].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)
            scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)

            X_train = scaling.transform(X_train)
            X_test = scaling.transform(X_test)

            y_pred = loaded_model.predict(X_test)

            # prediction results
            print('SVM Accuracy Score: ')
            print(accuracy_score(y_pred, y_test))
            print('SVM Confusion Matrix:')
            print(confusion_matrix(y_test, y_pred))
            print('SVM Classification Report:')
            print(classification_report(y_test, y_pred))
        
else:
    # Read image
    rbc_img_crop_name = ''
    if all_regions:
        rbc_img_crop_name = '/mnt/Data/isr/ISR_Leica_Scans/RBC_ClassDB/all/EH5_Siemens003/304/JAI_0000304_455_760.png'
    elif center_region:
        rbc_img_crop_name = '/home/slobodan/Pictures/Selection_001.png'

    img = cv2.imread(rbc_img_crop_name, cv2.IMREAD_COLOR)
    cv2.imshow('Image', img)
    cv2.waitKey()

    # Calculations for all regions
    if all_regions:
        # Threshold regions according to the range values
        image_regions_a = Regions(img, 49, 147)
        #image_regions_a = Regions(img, 0, 2)

        # Convert region labels to colors and show regions
        image_regions_a.color_regions()
        image_regions_a.show_regions()
        #print('Region centroids: ', image_regions_a.regions_centroids)

        # Extract contours in colored image, get ellipses and show them
        image_regions_a.get_contours()
        image_regions_a.get_ellipses()
        image_regions_a.draw_ellipses()

        # Compute contour attributes
        cnt_attribs = ContourAttributes(image_regions_a)
        cnt_attribs.compute_contour_area()
        cnt_attribs.compute_contour_centroids()
        print('Contour centroids : ', cnt_attribs.cntcenter)

        # Update region labels and properties so to match indices of contours
        image_regions_a.match_region_to_contour_indices(cnt_attribs.cntcenter)
        image_regions_a.update_regions()

        # Compute remaining contour properties
        cnt_attribs.compute_region_area()
        cnt_attribs.compute_circularity()
        cnt_attribs.compute_compactness()
        cnt_attribs.compute_eccentricity()
        cnt_attribs.compute_roundness()

        # Print contour properties
        print('Contour area = ', cnt_attribs.cntarea)
        print('Region  area = ', cnt_attribs.regarea)
        print('Circularity = ', cnt_attribs.circularity)
        print('Compactness = ', cnt_attribs.completeness)
        print('Anisometry = ', cnt_attribs.anisometry)
        print('Bulkiness = ', cnt_attribs.bulkiness)
        print('Structure factor = ', cnt_attribs.stuct_factor)
        print('Roundness = ', cnt_attribs.roundness)

        cnt_attribs.compute_contour_regions_mean_stddev_gray()
        cnt_attribs.compute_contour_regions_mean_stddev_rgb()
        cnt_attribs.compute_contour_regions_mean_stddev_hsv()

    # Calculations only for central region
    if center_region:
        # Threshold regions according to the range values
        #image_regions_c = Regions(img, 49, 147)
        image_regions_c = Regions(img, 0, 2)

        # Convert region labels to colors and show regions
        image_regions_c.color_regions()
        image_regions_c.show_regions()
        print('Region centroids: ', image_regions_c.regions_centroids)

        # Extract contours in colored image, get ellipses and show them
        image_regions_c.get_contours()
        image_regions_c.get_ellipses()

        # Compute contour attributes
        cnt_attribs = ContourAttributes(image_regions_c)
        cnt_attribs.compute_contour_area()
        cnt_attribs.compute_contour_centroids()
        print('Contour centroids : ', cnt_attribs.cntcenter)

        # Update region labels and properties so to match indices of contours
        image_regions_c.match_region_to_contour_indices(cnt_attribs.cntcenter)
        image_regions_c.update_regions()
        print('Region centroids: ', image_regions_c.regions_centroids)

        # Extract central region/contour index
        image_regions_c.set_central_region_index()
        #image_regions_c.remove_non_central_regions()

        # Show regions and central's bbox
        image_regions_c.show_regions()
        image_regions_c.draw_bbox_center_region()

        # Compute remaining contour properties
        cnt_attribs.compute_region_area()
        cnt_attribs.compute_circularity()
        cnt_attribs.compute_compactness()
        cnt_attribs.compute_eccentricity()
        cnt_attribs.compute_roundness()

        cnt_attribs.compute_contour_regions_mean_stddev_gray()
        cnt_attribs.compute_contour_regions_mean_stddev_rgb()
        cnt_attribs.compute_contour_regions_mean_stddev_hsv()

        # Print contour properties
        c_index =  image_regions_c.center_reg_index
        if c_index == 1:
            c_index = 0
        print('Contour area = ', cnt_attribs.cntarea[c_index])
        print('Region  area = ', cnt_attribs.regarea[c_index])
        print('Circularity = ', cnt_attribs.circularity[c_index])
        print('Compactness = ', cnt_attribs.completeness[c_index])
        print('Anisometry = ', cnt_attribs.anisometry[c_index])
        print('Bulkiness = ', cnt_attribs.bulkiness[c_index])
        print('Structure factor = ', cnt_attribs.stuct_factor[c_index])
        print('Roundness = ', cnt_attribs.roundness[c_index])
        print('Mean Gray = ', cnt_attribs.reg_means_gray[c_index])
        print('Std. Gray = ', cnt_attribs.reg_stddevs_gray[c_index])
        print('Mean RBG = ', cnt_attribs.reg_means_RGB[c_index])
        print('Std. RGB = ', cnt_attribs.reg_stddevs_RGB[c_index])
        print('Mean HSV = ', cnt_attribs.reg_means_HSV[c_index])
        print('Std.HSV = ', cnt_attribs.reg_stddevs_HSV[c_index])

    '''
    # Height, width, number of channels in image
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    img_center = [height/2, width/2]
    
    # Decompose it to 3 channels
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    
    # Threshold image green channel
    green_filtered = cv2.inRange(g, 49, 147)
    cv2.imwrite('../output/segmented.png',green_filtered)
    cv2.imshow('Segmented', green_filtered)
    
    # Find connected components with stats
    [num_labels, labels, stats, centroids] = cv2.connectedComponentsWithStats(green_filtered, 8, cv2.CV_32S)
    
    # Extract and draw bbox of the central segmented region
    dist_to_centroid = []
    for i in range(0, num_labels):
        [cx, cy] = centroids[i]
        r2 = (img_center[0]-cx)**2 + (img_center[1]-cy)**2
        dist_to_centroid.append(r2)
    
    index_min = np.argmin(dist_to_centroid)
    
    print('The segment in the center has coordinates = ', centroids[index_min], ' and area = ', stats[index_min,cv2.CC_STAT_AREA])
    
    # Draw a rectangle with blue line borders of thickness of 2 px
    top_left_point = np.array([stats[index_min, cv2.CC_STAT_LEFT], stats[index_min, cv2.CC_STAT_TOP]])
    end_right_point = top_left_point + np.array([stats[index_min, cv2.CC_STAT_WIDTH], stats[index_min, cv2.CC_STAT_HEIGHT]])
    print('top_left_point ', top_left_point)
    print('end_right_point', end_right_point)
    img = cv2.rectangle(img, tuple(top_left_point), tuple(end_right_point), (255,0,0), 1)
    cv2.imshow('Detected central', img)
    cv2.waitKey()
    
    # Fit ellipse to the segment
    labels[labels != index_min] = 0
    labeled_img = imshow_components(labels)
    labeled_gray = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
    threshod = np.amax(labeled_gray)
    th, th_image = cv2.threshold(labeled_gray, threshod-1, 255, cv2.THRESH_BINARY);
    cont_img = th_image.copy()
    contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000 or area > 4000:
            continue
        if len(cnt) < 5:
            continue
        ellipse = cv2.fitEllipse(cnt)
    
    cv2.ellipse(img, ellipse, (0,255,0), 1)
    cv2.imshow('Contours', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    '''
