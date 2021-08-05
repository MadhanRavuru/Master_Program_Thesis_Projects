## Dimensionality Reduction, Clustering and Visualization

We analyze only data, where center cell is extracted properly. Else, it does not make sense to extract hand-picked features for the cell.

We are interested in clustering and classifying size, shape and hemoglobin distribution variations of extracted center cell from crop.

**Size classes:** 
-  MACR
-  MICR
-  NORM

**Shape classes:**
-  ACAN 
-  BITE 
-  ECHI 
-  ELLI 
-  HELM 
-  NONE 
-  OVAL 
-  SCHI 
-  SICK 
-  SPHE
-  TEAR

**Hemoglobin distribution classes:** 
 -  HYPO
 -  HYPR
 -  NONE
 -  STOM
 -  TARG

### Hand-picked features for unsupervised learning:

**Size (10 features):**
-  contour area
-  region area 
-  ellipse (major axis, minor axis, area ,perimeter)
-  circle(area, perimeter)
-  rect(max, min diameter)

**Shape (22 features):**
-  Shape factors (extent, circularity, solidity, convexity, elongation, compactness)
-  Absolute Hu moments binary (Hu 1, Hu 2,..., Hu 7)
-  Geometric features (ellipse eccentricity and Goodness of fit, circle goodness of fit)
-  Boundary features (Spline Curvature mean and std dev, max and min curvature, No. of protrusions and indentations)

**Hemoglobin distribution (21 features):**
-  Colour features (mean and std deviation of gray, RGB, HSV image)
-  Absolute Hu moments gray (Hu 1, Hu 2,..., Hu 7)


### Dimensionality reduction:

From hand-picked features of each variation, we take 2 components of t-SNE which performs better than PCA. We change perplexity value of t-SNE algorithm, untill we get good clusters for the classes. PCA didn't form clusters although good amount of variance was retained.

### Clustering: 

Hierarchical Density Based Spatial Clustering Algorithm (HDBSCAN) is used on 2 component t-SNE to visualize clusters. It is better than partition based clustering algorithms, as it can detect outliers within clusters.

### Example
Size Clustering is performed as shown in notebook _clustering_and_classification.ipynb_. Clustering is done on the low dimensional features from t-SNE algorithm. Visualization of low dimensional features vary each time, on using t-SNE. As a result, the clusters formed using HDBSCAN also vary.

## SVM classification

We train 3 SVM classifiers (Size, Shape, Hemoglobin distribution) from the hand-picked features with polynomial kernel.
There is imbalance between the classes. So, we train for by balancing using class weights.
The obtained results are good, which suggests the hand-picked features for the respective classifiers are relevant.

## Reference:
The features for analyis were considered from [A methodology for morphological feature extraction and unsupervised cell classification](https://www.biorxiv.org/content/10.1101/623793v1.full)
