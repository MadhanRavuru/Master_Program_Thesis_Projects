import numpy as np
import random

import scipy.stats
from skimage import img_as_bool, io, color, morphology


# for image stuff and visualization
import matplotlib.pyplot as plt 

# OpenCV for image processing
import cv2

# for labelling
from skimage.morphology import label

from sklearn.decomposition import PCA

import pandas as pd

import os

from data_utils import label_img_to_rgb
from file_loader import get_all_files

###############################################################################
## Class: CellSeparator
## Description:
##   Class encapsulating line segments separating two cells
###############################################################################

class CellSeparator(object):
  """Container class for a dataset."""
  def __init__(self, cnt1, cnt2, pt1, pt2, delta):
    self._cnt1  = cnt1
    self._cnt2  = cnt2
    self._pt1   = pt1
    self._pt2   = pt2
    self._delta = delta

  def optimize_all_pts(self, nr_steps=10):
    nr_steps  = max(3, nr_steps)
    delta     = self._pt2-self._pt1
    len       = np.linalg.norm(delta) 
    step      = step/(nr_steps-1)

    result = np.zeros((1, nr_steps, 2))
    for i in range(0, nr_steps):
      result[0][i]=optimize_single_pt(self._pt1+i*step)
    return result


  def optimize_single_pt(self, p, max_step=20):
    # 3) Optimize the mid point such that is outside of both contours
    dist1 = cv2.pointPolygonTest(self._cnt1, (p[0], p[1]), True)
    dist2 = cv2.pointPolygonTest(self._cnt2, (p[0], p[1]), True)

    # dist1 positive, i.e inside, shift towards dist2
    i=0
    while dist1>0 and dist2<0 and i<max_step:
      p+=self._delta
      dist1 = cv2.pointPolygonTest(self._cnt1, (p[0], p[1]), True)
      dist2 = cv2.pointPolygonTest(self._cnt2, (p[0], p[1]), True)
      i=i+1

    # dist1 positive, i.e inside, shift towards dist2
    i=0
    while dist1<0 and dist2>0 and i<max_step:
      p-=self._delta
      dist1 = cv2.pointPolygonTest(self._cnt1, (p[0], p[1]), True)
      dist2 = cv2.pointPolygonTest(self._cnt2, (p[0], p[1]), True)
      i=i+1
    return p


###############################################################################
## Function: _find_contours
## FunctionType: 
## Description: 
##   Convenience function for contour finding
###############################################################################

def _find_contours(img):
  version = int(cv2.__version__.split('.')[0])
  if version<4:
    _, contours, _ = cv2.findContours(img,cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
  else:
    contours, _ = cv2.findContours(img,cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
  return contours


###############################################################################
## Function: is_pt_within_img
## FunctionType: 
## Description: 
##   checks if a point is within the image borders
###############################################################################

def is_pt_within_img(pt, img):
  if pt[0]<0 or pt[1]<0:
    return False
  if pt[0]>img.shape[1]-1 or pt[1]>img.shape[0]-1:
    return False
  return True


###############################################################################
## Function: is_left_of
## FunctionType: 
## Description: 
##   computes a binary mask for each class that is 1 where the segmentation has
##   the current class, 0 otherwise- needed for iou-calculation
###############################################################################

def is_left_of(a,b,c):
  """checks if p3 is left of the line formed by p1 and p2"""
  return ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])) > 0;



###############################################################################
## Function: _get_contour_pts_as_mat
## FunctionType: 
## Description: 
##   ..
###############################################################################

def _get_contour_pts_as_mat(border_img):
  separators = _find_contours(border_img)
  if separators is None or separators is []:
    return None

  filtered_pts = None
  for c in separators:
    # kick out the small fish
    if cv2.contourArea(c)<100:
      continue
    if filtered_pts is not None:
      filtered_pts=np.vstack((filtered_pts, c.squeeze()))
    else:
      filtered_pts=c.squeeze()

  # put all points (from all contours) into a single matrix
  poly = np.polyfit(filtered_pts[:,0], filtered_pts[:,1], 2)
  print(poly)



###############################################################################
## Function: _fit_poly_to_cont
## FunctionType: 
## Description: 
##   ..
###############################################################################

def _calc_max_rms(pts, poly):
  # compute the maximal rms fit
  max_error=0.
  for pt in pts:
    y=np.polyval(poly,pt[0]) 
    error = np.fabs(y-pt[1])
    if error>max_error:
      max_error=error
  return max_error

def _poly_to_array(pts, poly, offset=20):
  # transform the points (x,y) and the poly p into an array of points 
  # (x, p(x)), thereby extrapolating by a distance of 'offset'
  x_min = int(np.amin(pts[:,0]))
  x_max = int(np.amax(pts[:,0]))
  
  counter=0
  new_pts = np.zeros((x_max-x_min+2*offset, 2), dtype='float64')
  for x in range(x_min-offset, x_max+offset):
    new_pts[counter]=(x, np.polyval(poly, x))
    counter=counter+1
  return new_pts

def _set_pts(pts, img, value, size):
  # sets the pixels of img to value for the coords specified by points
  # is size>0, use a rectangular pen of the given size
  h = img.shape[0]
  w = img.shape[1]
  s = int(size/2)

  # loop over points
  for pt in pts:
    y=int(pt[1])
    x=int(pt[0])
    y=np.clip(y,0,h-1)
    x=np.clip(x,0,w-1)
    img[y,x]=value
    for dy in range(-s,s):
      for dx in range(-s,s):
        y2=np.clip(y+dy,0,h-1)
        x2=np.clip(x+dx,0,w-1)
        img[y2,x2]=value


def _fit_ransac(pts, pca, img, seg_img):
  # currently unused
  best_rms = 1000.
  nr = 5
  max_error=20
  cur_pts=np.empty((nr, 2), dtype=pts.dtype)
  pts_a=np.empty_like(pts)
  pts_b=np.empty_like(pts)
  best_poly_a = None
  best_poly_b = None
  best_pts_a  = None
  best_pts_b  = None
  
  for it in range(0, 100):
    for i in range(0, nr):  
      cur_pts[i] = random.choice(pts)
    poly = np.polyfit(cur_pts[:,0], cur_pts[:,1], 2)

    # loop over all points and select the rest
    counter_a=0
    counter_b=0
    for pt in pts:
      y=np.polyval(poly,pt[0]) 
      error = np.fabs(y-pt[1])
      if error<max_error:
        pts_a[counter_a]=pt
        counter_a+=1
      else:
        pts_b[counter_b]=pt
        counter_b+=1

    # fit poly to rest
    poly_a = np.polyfit(pts_a[0:counter_a,0], pts_a[0:counter_a,1], 2)
    poly_b = np.polyfit(pts_b[0:counter_b,0], pts_b[0:counter_b,1], 2)

    # if max_rms<10 return good!
    rms2 = _calc_max_rms(pts_b[0:counter_b], poly_b)
    if rms2<best_rms:
      best_rms=rms2
      best_poly_a = np.copy(poly_a)
      best_poly_b = np.copy(poly_b)
      best_pts_a  = np.copy(pts_a[0:counter_a])
      best_pts_b  = np.copy(pts_a[0:counter_b])
  print(best_rms)

  # deal with a 
  pts1 = _poly_to_array(best_pts_a, best_poly_a)
  new_pts = pca.inverse_transform(pts1)
  _set_pts(new_pts, img, 128, 0)
  _set_pts(new_pts, seg_img, 0, 2)

  # deal with b
  pts1 = _poly_to_array(best_pts_b, best_poly_b)
  new_pts = pca.inverse_transform(pts1)
  _set_pts(new_pts, img, 128, 0)
  _set_pts(new_pts, seg_img, 0, 2)


def _fit_poly_to_cont(c, degree, img, seg_img):
  # replace c with a polyomial of degree 2
  h = img.shape[0]
  w = img.shape[1]
  filtered_pts=c.squeeze()

  pca = PCA(n_components=2)
  pca.fit(filtered_pts)

  # Use the new basis and for a poly
  new_pts = pca.transform(filtered_pts)
  poly = np.polyfit(new_pts[:,0], new_pts[:,1], degree)
  max_error = _calc_max_rms(new_pts, poly)

  # the eigenvalues should give a hint if we have a line-like appearance or not
  eigenvalues = pca.explained_variance_

  # ok, what do we do if the second eigenvalue e2 is >0.5*e1
  # in that case we often have a peace-sign appearance (the borders are jointed). Can't fit poly for contour.
  if eigenvalues[1]>50 and max_error>15:
    #_fit_ransac(new_pts, pca, img, seg_img)
    new_img=np.zeros_like(seg_img)
    cv2.drawContours(new_img, [c], 0, 255, -1)
    out = morphology.skeletonize(img_as_bool(new_img))
    #out = morphology.thin(img_as_bool(new_img))
  
    # Create a small kernel
    kernel = np.ones((2,2), np.uint8)     
    output = cv2.dilate(out.astype('uint8'), kernel)
    
    # apply it as inverse mask
    seg_img[output>0]=0
    img[output>0]=128
    #plot_images([seg_img, new_img, out, out2], 2, 2)    

  else:
    
    pts1 = _poly_to_array(new_pts, poly)   # Offset(extrapolation) is 20 
    new_pts = pca.inverse_transform(pts1)
    _set_pts(new_pts, img, 128, 0)
    _set_pts(new_pts, seg_img, 0, 2)
  
 
   




###############################################################################
## Function: wipe_out_second_cell
## FunctionType: 
## Description: 
##   Estimates a separator line (to do:polyline) between two contours and 
##   sets the area of the second contour to 0
###############################################################################

def wipe_out_second_cell(img, path, c1, c2):

  # 0) load the split image, if available
  folder_name, file_name = os.path.split(path)
  file_name, _ = os.path.splitext(file_name)
  b_path = folder_name+"_border.png"
  if os.path.exists(b_path):
    border_img = cv2.imread(b_path, cv2.IMREAD_GRAYSCALE)

    separators = _find_contours(border_img)
    if separators is not None and separators is not []:

      # put all points (from all contours) into a single matrix
      # this is not a good idea with three or more cells
      all = np.vstack(separators).squeeze()
      poly = np.polyfit(all[:,0], all[:,1], 2)
      print(poly)
      mat = np.zeros((img.shape[1], 2))

      for x in range(0, img.shape[1]):
        y=int(np.polyval(poly,x))
        y=y if y>0 else 0
        y=y if y<img.shape[0]-1 else img.shape[0]-1
        img[y,x]=2
        mat[x]=(x,y)


      # now we have to add two corners, but which ones?

      plot_images([img, border_img], 1, 2)




    
    # what to do about it?
    # * fit polynomial (2nd degree)
    # * straight line from beginning to image border and end to image border
    #   (fit thru first n points and last n points)
    # * decide contour
    

  # 1) Get the contour centers
  mom1    = cv2.moments(c1)
  mom2    = cv2.moments(c2)
  (x1,y1) = (mom1['m10'] / mom1['m00'], mom1['m01'] / mom1['m00'])
  (x2,y2) = (mom2['m10'] / mom2['m00'], mom2['m01'] / mom2['m00'])

  # 2) Derive several vectors from the two center points
  # delta shows in direction of p2 from p1
  delta     = normalize(np.array([x2-x1, y2-y1]))
  mid_pt    = np.array([(x1+x2)/2,(y1+y2)/2])
  ortho     = np.array([-delta[1], delta[0]])

  # 3) Optimize the mid point such that is outside of both contours
  dist1 = cv2.pointPolygonTest(c1, (mid_pt[0], mid_pt[1]), True)
  dist2 = cv2.pointPolygonTest(c2, (mid_pt[0], mid_pt[1]), True)

  # dist1 positive, i.e inside, shift towards dist2
  while dist1>0 and dist2<0:
    mid_pt+=delta
    dist1 = cv2.pointPolygonTest(c1, (mid_pt[0], mid_pt[1]), True)
    dist2 = cv2.pointPolygonTest(c2, (mid_pt[0], mid_pt[1]), True)

  # dist1 positive, i.e inside, shift towards dist2
  while dist1<0 and dist2>0:
    mid_pt-=delta
    dist1 = cv2.pointPolygonTest(c1, (mid_pt[0], mid_pt[1]), True)
    dist2 = cv2.pointPolygonTest(c2, (mid_pt[0], mid_pt[1]), True)

  # 4) Derive 2 image border points that define the basis of the separator
  #    The approach is simple - go from the mid point in the direction of the
  #    ortho vector until an image border is reached
  k=0.
  while is_pt_within_img(mid_pt+k*ortho, img):
    k=k+1.
  pt1=mid_pt+(k-1)*ortho;

  k=0.
  while is_pt_within_img(mid_pt+k*ortho, img):
    k=k-1.
  pt2=mid_pt+(k+1)*ortho;
 

  # Form a polyline from there...
  # this is rather complicated as we also need a suitable order
  w = img.shape[1]
  h = img.shape[0]
  dist1 = cv2.pointPolygonTest(c1, (w/2, h/2), True)
  dist2 = cv2.pointPolygonTest(c2, (w/2, h/2), True)

  result = np.zeros((1, 4, 2), dtype='int32')
  result[0][0] = pt1
  result[0][1] = pt2
  counter=2

  # These are the four corners
  pts_to_check=[[0,0], [w-1, 0], [w-1, h-1], [0, h-1]]
  
  is_left = is_left_of(pt1, pt2, np.array([x1, y1]));
  if dist1<dist2:
    is_left = is_left_of(pt1, pt2, np.array([x2, y2]));

  dists = []
  for pt in pts_to_check:
    if is_left!=is_left_of(pt1, pt2, np.array(pt)):
      result[0][counter]=pt
      counter=counter+1
      dists.append((pt2[0]-pt[0])*(pt2[0]-pt[0])+(pt2[1]-pt[1])*(pt2[1]-pt[1]))

  to_draw = np.array([])
  if counter==4:
    if dists[1]<dists[0]:
      tmp=result[0][2].copy()
      result[0][2]=result[0][3]
      result[0][3]=tmp
    to_draw=result
  else:
    to_draw = np.array([[result[0][0], result[0][1],result[0][2]]])
  cv2.fillPoly(img, to_draw, (0,0,0))


###############################################################################
## Function: split_cells
## FunctionType: 
## Description: 
##   Splits ... if there are several nuclei
###############################################################################

def split_cells(color_img, seg_img, contour_cell, contour_nuc, areas_nuc, 
                path, do_show):
  if len(contour_nuc)<=1:
    return False

  # we have several nuclei!!
  cell_area = cv2.contourArea(contour_cell)
  min_nuc_area = min(areas_nuc)
  max_nuc_area = max(areas_nuc)
  print("Cell area: "+str(cell_area)+", n_min: "+str(min_nuc_area)+
        ", max: "+str(max_nuc_area))

  # we have some hardcoded logic here, that is no good
  if cell_area<15000 or max_nuc_area<5000:
    # with segmented neutrophils, there can be several small nuceli
    return False

  # Separate the largest cell from the rest
  if do_show:
    debug_img = color_img.copy()
    for i,c in enumerate(contour_nuc):
      cur_area   = float(cv2.contourArea(c))
      (xc, yc), radius = cv2.minEnclosingCircle(c)
      cv2.putText(debug_img, str(cur_area), (int(xc), int(yc)), 
                  cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255),2)
    plot_images([seg_img,debug_img], 1, 2, title="Multiple Nuclei", 
                window_title="Debug")
  wipe_out_second_cell(seg_img, path, contour_nuc[0], contour_nuc[1])
  if do_show:
    plot_images([seg_img], 1, 1, title="Selected Nucleus")
  return True


###############################################################################
## Function: extract_per_class_masks
## FunctionType: 
## Description: 
##   computes a binary mask for each class that is 1 where the segmentation has
##   the current class, 0 otherwise- needed e.g. for iou-calculation
###############################################################################

def extract_per_class_masks(segm, all_classes):
  s = segm.shape
  h = s[0]
  w = s[1]
  nr_classes = all_classes.shape[0]
  masks = np.zeros((nr_classes, h, w))

  # compute a mask that is 1 where the segmentation has the current class, 
  # 0 otherwise
  for i, c in enumerate(all_classes):
    masks[i, :, :] = segm == c
    
  return masks


###############################################################################
## Function: set_margin_to_zero
## FunctionType: 
## Description: 
##   Sets the margin to 0
###############################################################################
 
def set_margin_to_zero(mat, m):
  '''Set the margin of the array/mat to 0'''
  h=mat.shape[0];w=mat.shape[1]
  mat[0:m]          =0
  mat[h-m-1:h]      =0
  mat[:,0:m]        =0
  mat[:,w-m-1:w-1]  =0


###############################################################################
## Function: calc_mean_IOU_per_image
## FunctionType: 
## Description: 
##   computes mean intersection over union
###############################################################################
 
def calc_mean_IOU_per_image(segm_is, segm_gt, margin=3):
  '''
  (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
  takes output and ground truth as parameters  
  '''
  set_margin_to_zero(segm_is, margin)
  set_margin_to_zero(segm_gt, margin)  
  
  classes_is  = np.unique(segm_is)
  classes_gt  = np.unique(segm_gt)
  all_classes = np.union1d(classes_is, classes_gt)
  nr_classes  = all_classes.shape[0]
  masks_is    = extract_per_class_masks(segm_is, all_classes)
  masks_gt    = extract_per_class_masks(segm_gt, all_classes)

  IU = list([0]) * nr_classes

  for i, c in enumerate(all_classes):
    cur_eval_mask  = masks_is[i, :, :]
    cur_gt_mask    = masks_gt[i, :, :]
 
    # check for empty masks
    if (np.sum(cur_eval_mask) == 0) or (np.sum(cur_gt_mask) == 0):
      continue

    n_ii = np.sum(np.logical_and(cur_eval_mask, cur_gt_mask))
    t_i  = np.sum(cur_gt_mask)
    n_ij = np.sum(cur_eval_mask)

    IU[i] = n_ii / (t_i + n_ij - n_ii)
 
  # calc the mean
  mean_IU_ = np.sum(IU) / classes_gt.shape[0]
  return mean_IU_, IU


###############################################################################
## Function: calc_mean_IOU_per_dataset
## FunctionType: 
## Description: 
##  Evaluate a data set with the current net
###############################################################################

def calc_mean_IOU_per_dataset(data_set, sess, img, label, pred,width,height):            

  results = np.zeros((data_set.nr_images))
  counter = 0

  # show results for
  for i in range(data_set.nr_images):

    # no augmentation in the case of evaluation
    x_batch, _, y_batch = data_set.next_batch_in_place(1)

    # run the graph
    s = x_batch[0].shape
    # sanit check - we get stuck if the images get too large (out of memory...)
    if s[0]*s[1]>800*800:
      print("Warning: Skipping image of size "+str(s[0])+" x "+str(s[1])+
            " as we might run out of memory")
      continue
    feed_dict = {img: x_batch,label: y_batch,width:s[1],height:s[0],
                 tf.keras.backend.learning_phase(): 0}
    pred_logits = sess.run([pred], feed_dict=feed_dict)
    
    # reduce to one-channel gray-level
    pred_map = np.argmax(pred_logits[0], axis=-1)
    score, _ = calc_mean_IOU_per_image(pred_map[0], y_batch[0])
    results[counter]=score
    counter+=1
  return np.mean(results[0:counter+1])


###############################################################################
## Function: normalize
## FunctionType: 
## Description: 
##  Vector normalization such that its sum is 1
###############################################################################

def normalize(v):
  norm=np.linalg.norm(v, ord=1)
  return v if norm==0 else v/norm


###############################################################################
## Function: softmax
## FunctionType: 
## Description: 
##  Applies softmax
###############################################################################

def softmax(x, axis=None):
  x = x - x.max(axis=axis, keepdims=True)
  y = np.exp(x)
  return y / y.sum(axis=axis, keepdims=True)


###############################################################################
## Function: calc_histogram
## FunctionType: 
## Description: 
##  Calculates the histogram of an image
###############################################################################

def calc_histogram(img, mask, nr_bins=64, do_show=True, do_reduce=False,
                   prefix=None, channel_list=None):
  names = []
  if do_reduce:
    mean, std_dev =   cv2.meanStdDev(img, mask=mask)
    values_as_list = mean.flatten().tolist()+std_dev.flatten().tolist()
    if prefix is not None and channel_list is not None:
      for c in channel_list:
        names.append(prefix+"_mean_"+c)
      for c in channel_list:
        names.append(prefix+"_stddev_"+c)
    return values_as_list, names

  color = ('b','g','r')
  result = []
  for i,col in enumerate(color):
    hist = normalize(cv2.calcHist([img],[i],mask,[nr_bins],[0,256]))
    # warning - this is an histogram, we would be more interested in the 
    # peak index as in the peak value...

    if do_reduce:
      #result.append(scipy.stats.describe(hist))
      median = np.median(hist)
      #mean = np.mean(hist)
      std  = np.std(hist)
      kurtosis = scipy.stats.kurtosis(hist)[0]
      skewness = scipy.stats.skew(hist)[0]
      result.append([median, std, kurtosis, skewness])
    else:
      result = result + ([x[0] for x in hist])
    if not do_show:
      continue
    plt.plot(hist,color = col)
    plt.xlim([0,nr_bins])
  
  if do_show:
    plt.show()
  return result


###############################################################################
## Function: calc_hull_deficit
## FunctionType: 
## Description: 
##  Calculates deficits of the convex hull
###############################################################################

def calc_hull_deficit(cnt, do_show, img):
  if len(cnt)==0:
    return 0, 0, 0, 0

  # fit circle to nucleus
  (x,y),radius = cv2.minEnclosingCircle(cnt)
  if do_show:
    cv2.circle(img,(int(x),int(y)),int(radius),(255,0,0),2)  

  # try out the contour deficits
  hull = cv2.convexHull(cnt,returnPoints = False)
  defects = cv2.convexityDefects(cnt,hull)
  if defects is None:
    return 0, 0, 0, 0
  rel_max_defect = 0.

  for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    d=d/256./radius
    if d>rel_max_defect:
      rel_max_defect=d

    if not do_show:
      continue
    if d<0.3:
      continue
       
    #print (str(i)+": "+str(d)+", "+str(d/radius))
    start   = tuple(cnt[s][0])
    end     = tuple(cnt[e][0])
    middle  = tuple(((cnt[s][0]+cnt[e][0])/2).astype('int32'))
    far = tuple(cnt[f][0])
    cv2.line(img,start,end,[0,255,0],2)
    cv2.line(img,middle,far,[0,255,0],2)
    cv2.circle(img,far,5,[255,0,0],-1)
  return x, y, radius, rel_max_defect     


###############################################################################
## Function: calc_coarseness_coeff
## FunctionType: 
## Description: 
##  Calculates the delta distribution
##  To do: should we do this per channel?
###############################################################################

def calc_coarseness_coeff(image, _mask, step, cap):
  # key idea: look at the four neighbors
  # calc sample distribution: -20 to +20
  # calc stats on that?
  s = image.shape
  offsets = [(-step, 0), (step, 0), (0,-step), (0,step)]

  # create an eroded version of the mask to avoid border regions
  kernel = np.ones((7,7),np.uint8)
  mask = cv2.erode(_mask, kernel)
  
  #tmp_mask = mask.copy()
  #tmp_mask = tmp_mask/2
  #tmp_mask = np.stack([tmp_mask, tmp_mask, tmp_mask], 2)
  #plot_images([image, ((image*tmp_mask)).astype('uint8'), _mask, mask], 2, 2)
  
  
  # loop over pixels
  result = []
  for row in range(step, s[0]-step):
    for col in range(step, s[1]-step):
      # are we within the masked region?
      m   = mask[row, col]
      if not m:
        continue

      # get the rgb values
      rgb = image[row, col].astype('float')
      for o in offsets:
        row2   = row+o[0]
        col2   = col+o[1]

        # Need to consider the borders...
        if row2<0 or col2<0:
          continue
        if row2>=s[0] or col2>=s[1]:
          continue

        if not mask[row2][col2]:
          continue

        rgb2  = image[row2, col2].astype('float')
        d     = rgb - rgb2
        d     = np.clip(d, -cap, cap)
        
        result.append(np.fabs(d[0]))
        result.append(np.fabs(d[1]))
        result.append(np.fabs(d[2]))
        

  return np.histogram(np.array(result), cap, density=True)[0]
  #return np.array(result)


###############################################################################
## Function: calc_min_dist_to_point
## FunctionType: 
## Description: 
##   
###############################################################################

def calc_min_dist_to_point(cnt, pt):
  # what data type do we have? How to get std::limits -like stuff?
  min_dist    = 10000*10000
  closest_pt  = None

  for _p in cnt:
    p  = _p[0]
    dx = p[0]-pt[0]
    dy = p[1]-pt[1]
    cur_dist = dx*dx+dy*dy
    if cur_dist<min_dist:
      min_dist = cur_dist
      closest_pt = p
  return np.sqrt(min_dist), closest_pt


###############################################################################
## Function: select_cell_contour
## FunctionType: 
## Description: 
##   Tries to select the cell contour, that is the contour of the cell we are
##   interested in; there might be several contours in an image
##   Currently, we simply use the one closest to the image center
##   (after filtering out small segments first)
###############################################################################

def select_cell_contour(cnts, img_center, min_size):

  min_dist  = -1000
  best_c   = None

  # Loop over the contours
  for c in cnts:
    if cv2.contourArea(c)<min_size:
      continue
    # Checks distance of contour to image center
    dist  = cv2.pointPolygonTest(c, img_center, True)
    dist = min(0, dist)
    if dist>min_dist:
      best_c=c
      min_dist=dist

  return best_c


###############################################################################
## Function: select_nuc_contours
## FunctionType: 
## Description: 
##   
###############################################################################

def select_nuc_contours(cnts, area_threshold):
  result = []
  areas  = []
  max_a  = 0

  for c in cnts:
    a = cv2.contourArea(c)
    if a>max_a:
      max_a=a
    if a>area_threshold:
      result.append(c)
      areas.append(a)

  if result==[]:
    print("Largetst nuc candidate found had area "+str(max_a))

  return result, areas


###############################################################################
## Function: _calc_params_internal
## FunctionType: 
## Description: 
##  Calculates a number of parameters from a segmented color image
##  This is the internal version that collects the parameter names and values
###############################################################################

def _calc_params_internal(color_img, seg_img, seg_nuc_img, border_img, 
                          contour_cell, contour_nuc, path, label):
  CYTOPLASM = 1
  font      = cv2.FONT_HERSHEY_PLAIN
  w         = (255, 255, 255)

  # issues we might have:
  # - contour_cell is None or []
  # - contour_nuc is None or []


  folder_name, file_name = os.path.split(path)
  file_name, _ = os.path.splitext(file_name)
  
  nuc_img   = color_img.copy()
  nuc_img[seg_img==0]=0
  cell_img  = nuc_img.copy()

  # if condense_histogram is set, we do not output the entire histogram as 
  # parameter, but a condensed version (mean, stddev, etc., see below)
  condense_histogram = True

  # we use HSV for some color stats, so let's create a HSV image
  hsv_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

  # fit circle to nucleus
  xn=0; yn=0; radius_n=0; rel_max_defect_n=0
  for c in contour_nuc:
    xn, yn, radius_n, rel_max_defect_n = calc_hull_deficit(c, do_draw, nuc_img)
  xc, yc, radius_c, rel_max_defect_c = calc_hull_deficit(contour_cell, do_draw, 
                                                         cell_img)

  # Calculate the eccentricity of the cell nucleus (relative to entire cell)
  ecc = np.sqrt((xc-xn)*(xc-xn)+(yc-yn)*(yc-yn))/(max(1,radius_c))

  # 1-2)  area cell, area nucleus
  # 3)    n-c ratio
  # 4)    radius minimal enclosing circle
  # 5)    relative size of the maximal convexcity defect
  a_cell  = 0 if len(contour_cell)==0 else float(cv2.contourArea(contour_cell))
  cv2.putText(cell_img, "Cell area: "+str(int(a_cell)), (10, 350-60), font, 1.2, w,2)
  area_nuc=0;
  
  for i,c in enumerate(contour_nuc):
    cur_area   = float(cv2.contourArea(c))
    (xc, yc), radius = cv2.minEnclosingCircle(c)
    cv2.putText(nuc_img, "Nuc area: "+str(int(cur_area)), (10,350-60), font, 1.2, w,2)
    area_nuc+=cur_area
  _a_cyto  = a_cell-area_nuc
  nc_ratio = min(5, area_nuc/(max(1., _a_cyto)))
  cv2.putText(cell_img, "NC ratio: "+str(int(nc_ratio*10)/10), (10, 350-40), font, 1.2, w,2)
  cv2.putText(nuc_img, "Eccentr.: "+str(int(ecc*10)/10), (10, 350-40), font, 1.2, w,2)


  # 6-7) perimeter
  perimeter_cell = 0 if len(contour_cell)==0 else (cv2.arcLength(contour_cell,True))
  perimeter_nuc = 0
  for c in contour_nuc:
    perimeter_nuc+=cv2.arcLength(c,True)

  # Roundness: 4*area/pi*max_diameter
  # Elongation: fiber length/fiber width
  # Curl: length/fiber length
  # Convexity: convex perimeter/perimeter
  #
  
  # 1-8) area cell, area nucleus, nc-ration, radius circle nuc, 
  if not label:
    label='Unknown'

  features = [file_name, label, a_cell, len(contour_nuc), area_nuc, nc_ratio, radius_n, 
              rel_max_defect_n, radius_c, rel_max_defect_c, ecc,
              perimeter_cell, perimeter_nuc]
  feature_names = ['filename', 'cell_type', 'cell_area', 'nr_segments_nuc', 'nuc_area', 
                   'nc_ratio', 'nuc_radius',
                   'nuc_max_defect',  'cell_radius', 'cell_max_defect', 
                   'eccentricity', 'cell_perimeter', 'nuc_perimeter']

  # Calc the two histogram for the cytoplasma
  seg_cyto_img = seg_img.copy()
  seg_cyto_img[seg_cyto_img!=CYTOPLASM]=0
  nr_bins=64
  for i in range(0,4):
    vals, names = calc_histogram(color_img if i%2==0 else hsv_img, 
      seg_cyto_img if i<2 else seg_nuc_img, nr_bins, False, condense_histogram, 
      "cyto" if i<2 else "nuc", ['r','g','b'] if i%2==0 else ['h','s','v'])
    features      = features + vals
    feature_names = feature_names+names

  # to do: check the definition...
  if len(contour_cell)==0:
    dummy = [0,0,0,0,0,0, 0]
  else:
    dummy = [x[0] for x in cv2.HuMoments(cv2.moments(contour_cell))]
  features = features + dummy
  if len(contour_nuc)==0: 
    dummy = [0,0,0,0,0,0, 0]
  else:
    dummy = [x[0] for x in cv2.HuMoments(cv2.moments(contour_nuc[0]))]
  features = features + dummy
  feature_names= feature_names+['cell_hu_mom1', 'cell_hu_mom2', 'cell_hu_mom3', 
                   'cell_hu_mom4','cell_hu_mom5','cell_hu_mom6','cell_hu_mom7']
  feature_names= feature_names+['nuc_hu_mom1', 'nuc_hu_mom2', 'nuc_hu_mom3', 
                   'nuc_hu_mom4','nuc_hu_mom5','nuc_hu_mom6','nuc_hu_mom7']
  
  if len(contour_cell)>0:
    rect = cv2.minAreaRect(contour_cell)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    if do_draw:
      cv2.drawContours(cell_img,[box],0,(0,255,0),2)
    #features = features + ([rect.width, rect.height])

  if len(contour_nuc)>0:
    rect = cv2.minAreaRect(contour_nuc[0])
    (x, y), (width, height), angle = rect
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    if do_draw:
      cv2.drawContours(nuc_img,[box],0,(0,255,0),2)
    #features = features + ([rect.width, rect.height])

  #ellipse = cv2.fitEllipse(cnt)
  #cv2.ellipse(color_img,ellipse,(0,255,0),2)


  
  #for x, y in zip(feature_names, features):
  #  print(x+": "+str(y))

  #c_coeff_nuc = calc_coarseness_coeff(color_img, seg_img, 2, 30)
  #for i in range(0, c_coeff_nuc.shape[0]):
  #  feature_names.append('nuc_median_coarseness'+str(i))
  #  features.append(c_coeff_nuc[i])

  #plt.hist(c_coeff_nuc, 100, range=(0, +20.))
  #plt.show()

  #c_coeff_cell = calc_coarseness_coeff(color_img, seg_img, 2, 30)
  #for i in range(0, c_coeff_cell.shape[0]):
  #  feature_names.append('cell_median_coarseness'+str(i))
  #  features.append(c_coeff_cell[i])

  #plt.hist(c_coeff_cell, 100, range=(0, +30.))
  #plt.show()

  # draw crosshairs for debugging only
  cv2.line(nuc_img, (int(color_img.shape[1]/2), 0),
           (int(color_img.shape[1]/2), color_img.shape[0]-1), (255,0,0)) 
  cv2.line(nuc_img, (0, int(color_img.shape[0]/2)),
           (color_img.shape[1]-1, int(color_img.shape[0]/2)), (255,0,0)) 


  return features, feature_names


###############################################################################
## Function: calc_params
## FunctionType: 
## Description: 
##  Calculates a number of parameters from a segmented color image
##  Returns features and feature names
##  To do: 
##    * should we use a peripheral area around the cell??
##    * should we count the percentage of the cell outline that connects to RBC
##    * how to combine segmented neutrophils whose nucleus is split up?
###############################################################################

def calc_params(color_img, seg_img, seg_raw, label, path):

  # some hardcoded parameters and constants
  CYTOPLASM     = 1
  nuc_min_area  = 750
  cell_min_area = 2000

  # try to load the matching border image to split the cell if necessary
  border_img=None
  full_path=path[0:-4]+"_border.png"
  if os.path.exists(full_path):
    border_img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    debug_img = seg_img.copy()
    # Apply the separator line as mask; the problem here is that the segment
    # is often too thick
    # idea would be to fit a poly per (large) segment and use the resulting
    # line (only if the residual is not too big). This would solve the problem
    # in 80-90% of the cases
    #seg_img[border_img!=0]=0
    #plot_images([color_img, debug_img, seg_img, border_img], 2, 2)

    cs = _find_contours(border_img)
    for c in cs:
      # kick out the small fish
      if cv2.contourArea(c)<100:
        continue
      _fit_poly_to_cont(c, 2, border_img, seg_img, False)


  # important: we check for non-zero here, i.e. we use both cytoplasm and
  # nucleus, i.e. we should find the contour of the entire cell
  contours_cell = _find_contours(seg_img)

  # now select the relevant cell contour - there could be several... 
  img_center = (color_img.shape[0]/2,color_img.shape[1]/2)
  contour_cell =select_cell_contour(contours_cell, img_center, cell_min_area)  

  # Wipe out everything outside of our cell...
  cell_of_interest_mask = np.zeros(seg_img.shape, seg_img.dtype)
  if contour_cell is not None:
    cv2.drawContours(cell_of_interest_mask, [contour_cell], -1, 255, -1)
  seg_img[cell_of_interest_mask<255]=0
  
  # This image is a mask for the nucleus (by deleting the cytoplasm)
  seg_img_nuc = seg_img.copy()
  seg_img_nuc[seg_img_nuc==CYTOPLASM]=0
  contours_nuc = _find_contours(seg_img_nuc)

  # Select the contour of the nucleus - there could be none, there could be several
  contour_nuc, areas_nuc  = select_nuc_contours(contours_nuc, nuc_min_area)

  contour_nuc = [] if contour_nuc is None else contour_nuc
  contour_cell= [] if contour_cell is None else contour_cell

  # Do we have a case of multiple cells?
  #if split_cells(color_img, seg_img, contour_cell, contour_nuc, areas_nuc,
  #               path, do_show):
  #  # Recursive call, this time without the removed cell
  #  return calc_params(color_img, seg_img, seg_raw, label, do_show, path)  
  
  return _calc_params_internal(color_img, seg_img,  seg_img_nuc, border_img,
                               contour_cell,contour_nuc, path, label)


###############################################################################
## Function: rle_encoding
## FunctionType: 
## Description: 
## Run-length encoding taken from 
## https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
###############################################################################

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

###############################################################################
## Function: prob_to_rles
## FunctionType: 
## Description: 
## Raw probability values to run-length-encoding
###############################################################################

def _prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


def _generate_rles(test_ids, predictions):
  new_test_ids = []
  rles = []
  for n, id_ in enumerate(test_ids):
    rle = list(_prob_to_rles(predictions[n]))
    if len(rle)==0:
      print("Problem")
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))
  return new_test_ids, rles

def _generate_rle(test_id, prediction):
  rle = list(_prob_to_rles(prediction))
  if len(rle)==0:
    print("Problem")
  return ([test_id] * len(rle)), rle


def create_submission_frame(test_ids, predictions):
  """Create submission DataFrame"""

  new_test_ids, rles = _generate_rles(test_ids, predictions)
  sub = pd.DataFrame()
  sub['ImageId'] = new_test_ids
  sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
  sub.to_csv('c:\\temp\\sub-dsbowl2018-1.csv', index=False)

    
###############################################################################
#################### Additional Function added   #############################
##### 1. Saving the predictions(segemented, borders)   #####
###############################################################################
def save_test_predictions_segmented(predictions):
    test_files = np.loadtxt('segmentation_data/test.txt', dtype = str)
    
    assert len(predictions) == len(test_files)
    
    for i, file in enumerate(test_files,0):
        file = file.replace('images','test_predictions/segmented_images')
        out_dir, img = os.path.split(file)
        
        
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        
        prediction = label_img_to_rgb(predictions[i])   #this is RGB image
        prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR) # conversion to BGR
        cv2.imwrite(file,prediction);


def save_test_predictions_border(predictions):
    test_files = np.loadtxt('segmentation_data/test.txt', dtype = str)
    
    assert len(predictions) == len(test_files)
    
    for i, file in enumerate(test_files,0):
        file = file.replace('images','test_predictions/border_images')
        out_dir, img = os.path.split(file)
        
        
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        
        prediction = label_img_to_rgb(predictions[i])   
        prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file,prediction);

def save_test_predictions_combined(predictions):
    
    data = 'segmentation_data/test_predictions/border_images/'
    border_images = get_all_files(data, 'png')
    
    
    for i, file in enumerate(border_images,0):
        file = file.replace('border_images','combined')
        out_dir, img = os.path.split(file)
        
        
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        
        cv2.imwrite(file,predictions[i]); #predictins are in BGR format
        
###############################################################################
##### 2. combining predicted segmented and border to get only center cell #####
###############################################################################

def combining_predicted_segmented_and_border(border_images, segmented_images):
       
    cell_min_area = 500                    
    center_cell_images = []
    for i in range(len(border_images)):
        border_img = cv2.imread(border_images[i], cv2.IMREAD_GRAYSCALE)
        seg_img = cv2.imread(segmented_images[i], cv2.IMREAD_GRAYSCALE)
        seg_color_img = cv2.imread(segmented_images[i])
        
        cs = _find_contours(border_img)
        
        for c in cs:
             # kick out the small fish
             if cv2.contourArea(c)<40:             
                 continue
                    
             # high degree polynomial approximation for big contours       
             if cv2.contourArea(c)>500:    
                 _fit_poly_to_cont(c, 7, border_img, seg_img)  
             else:
                 _fit_poly_to_cont(c, 2, border_img, seg_img)
             
        
        contours_cell = _find_contours(seg_img)
        img_center = (seg_color_img.shape[0]/2,seg_color_img.shape[1]/2)
        contour_cell = select_cell_contour(contours_cell, img_center, cell_min_area)
        
        cell_of_interest = np.zeros(seg_img.shape, seg_img.dtype)
        if contour_cell is not None:
            cv2.drawContours(cell_of_interest, [contour_cell], -1, 255, -1)
            seg_img[cell_of_interest<255]=0
            msk = cv2.dilate(seg_img, np.ones((3,3),np.uint8))   #to cover cell completely (fill the gap introduced by polylines, if any)
            msk = cv2.erode(msk, np.ones((3,3),np.uint8))
            
        seg_color_img[msk==0] = 0 
        
        center_cell_images.append(seg_color_img)   #BGR image format
    
    center_cell_images = np.array(center_cell_images)   
    return center_cell_images   

