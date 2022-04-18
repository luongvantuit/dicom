import numpy as np
from pydicom import dcmread
 
DICOM_file_path = '/Users/user/Desktop/DICOM/PAT034/D0001.dcm'
 
DICOM_file = dcmread(DICOM_file_path)
print(DICOM_file)
 
## DICOM_file is FileDataSet object with attributes that match the meta data 
print("class object from pydicom ->", type(DICOM_file))
 
 
# plot the image using matplotlib
import matplotlib.pyplot as plt
 
plt.imshow(DICOM_file.pixel_array, cmap=plt.cm.gray)
plt.show()
 
## Lets see pixel values 
print(DICOM_file.pixel_array)
# from DataScienceDeck.preprocessing.preprocessing import load_scan
 
# pathway = 'D:\DICOM files\PAT001'
 
# x = load_scan(pathway)
# print(x)
 
# from DataScienceDeck.preprocessing.preprocessing import load_scan
import numpy as np
import os
from pydicom import dcmread
 
def load_scan(path):
    print("Loading scan", path)
    slices = [dcmread(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
 
    if slices[0].ImagePositionPatient[2] == slices[1].ImagePositionPatient[2]:
        sec_num = 2;
        while slices[0].ImagePositionPatient[2] == slices[sec_num].ImagePositionPatient[2]:
            sec_num = sec_num+1;
        slice_num = int(len(slices) / sec_num)
        slices.sort(key = lambda x:float(x.InstanceNumber))
        slices = slices[0:slice_num]
        slices.sort(key = lambda x:float(x.ImagePositionPatient[2]))
 
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices # list of DICOM 
 
pathway = 'D:\DICOM files\PAT001'
 
All_DICOM_FILES = load_scan(pathway)
#print(All_DICOM_FILES)
 
# Now here we can for loop over all the DICOM files
pixel_ds = []
 
for files in All_DICOM_FILES:
    pixel_numpy = files.pixel_array
    pixel_ds.append(pixel_numpy)
 
print(pixel_ds)
 
def get_pixels_hu(slices):
    print("get pixels converted to HU")
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):        
        intercept = slices[slice_number].RescaleIntercept ## DICOM metadata attribute call
        slope = slices[slice_number].RescaleSlope         ## DICOM metadata attribute call
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
 
    case_pixels = np.array(image, dtype=np.int16)
    pixel_spacing = np.array([slices[0].SliceThickness, slices[0].PixelSpacing[0], slices[0].PixelSpacing[1]], dtype=np.float32)
    return case_pixels, pixel_spacing
 
 
####  get_pixels_hus converting pixel values to HU values 
patient_pixels, pixel_spacing = get_pixels_hu(All_DICOM_FILES)
 
#### Code plots the slice 150 
import matplotlib.pyplot as plt
plt.hist(patient_pixels[150].flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()
 
plt.imshow(patient_pixels[150], cmap=plt.cm.gray)
plt.show()
 
### cide runs for slice 1
plt.hist(patient_pixels[1].flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()
# # Show some slice in the middle
plt.imshow(patient_pixels[1], cmap=plt.cm.gray)
plt.show()
 
## mount google drive, be careful limited space on my google drive 
from google.colab import drive
drive.mount('/content/drive/')
 
# !pip3 install pydicom
#https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
#https://www.kaggle.com/sentdex/first-pass-through-data-w-3d-convnet
#https://numpy.org/doc/stable/reference/index.html#module-numpy
#https://docs.scipy.org/doc/scipy/reference/ndimage.html#module-scipy.ndimage
#https://docs.scipy.org/doc/scipy/reference/tutorial/general.html
 
import numpy as np
import pydicom
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io 
 
from pydicom import dcmread
 
 
## skimage algorithms for image processing
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
 
## plotly
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *
import plotly.graph_objects as go
init_notebook_mode(connected=True) 
 
from sklearn.cluster import KMeans
 
 
def make_lungmask(img, display=False):
    row_size= img.shape[0]
    col_size = img.shape[1]
    mean = np.mean(img)
    std = np.std(img) 
    img = (img-mean)/std # Subtracts mean and divide standard deviation from each element ## Standardises the whole data
 
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img
    # middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)]
    print(middle)
    print(middle.shape)
    
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
 
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean # Goes through the list and if anything equals max, it changes its value to mean.
    img[img==min]=mean # Goes through the list and if anything equals min, it changes its value to mean.
 
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #sensitive to outliers and noise Kmeans could be really bad 
    # https://www.youtube.com/watch?v=_aWzGGNrcic
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
 
 
    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # We don't want to accidentally clip the lung.
 
    eroded = morphology.erosion(thresh_img,np.ones([3,3]))
    dilation = morphology.dilation(eroded,np.ones([8,8]))
 
  ## START here
    labels = measure.label(dilation) 
    # Different labels are displayed in different colors
 
    label_vals = np.unique(labels)
    ## ok region props what is this 
    regions = measure.regionprops(labels)
    good_labels = []
    
# what is this SECTION OF CODE?? 
# why is it ommitting half the lung 
    for prop in regions:
        B = prop.bbox
        good_labels.append(prop.label)
 
        # if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
        #     good_labels.append(prop.label)
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0
 
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
 
    if (display):
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ## Labels so after labels some parts disppear at the 1-30
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
 
        ## cuts off body cavity at 50 - 140 WHY!!?!?!
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask*img, cmap='gray')
        ax[2, 1].axis('off')
        
        plt.show()
    return mask*img
 
#data_dir = "/content/drive/My Drive/Vessel Outline storage/Input_data/Stats data/data_filePAT001.npy"
data_dir = "/content/drive/My Drive/Vessel Outline storage/Input data/Stats data/data_filePAT001.npy"
 
## PAT001
img_after_samples = np.load(data_dir)
 
#print(img_after_samples)
## size of array also 
len_of_img = len(img_after_samples)
print("Number of slices" ,len_of_img)
#print(img_after_samples) ## 3D dimsions 
## step though by changing this number in img_after_samples
## could implement slice range with for loop for number in img_after_samples 
slice_num = 140
slice_single = img_after_samples[slice_num]
print(slice_single)
print(slice_single.shape)
#print("Slice number: ", slice_num + 1)
slice_one = make_lungmask(slice_single, display=True)
 
#for img in img_after_samples:
 #make_lungmask(img, display=True)
#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html