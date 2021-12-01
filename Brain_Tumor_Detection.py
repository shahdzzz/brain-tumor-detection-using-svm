#!/usr/bin/env python
# coding: utf-8

# In[34]:


#importing libraries
import numpy as np
import pandas as pd
import os
from os import listdir
from sklearn.utils import shuffle
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import pathlib
import seaborn as sns
from keras.applications.vgg16 import VGG16
import math
from  keras.preprocessing.image import *
import pixellib
from pixellib.custom_train import instance_custom_training
from random import choices
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Conv2D,MaxPooling2D,BatchNormalization,Flatten,Dropout
import splitfolders
from glob import glob
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm


# In[35]:


#Uploading data
dataDir = r"F:\Materials\AI\brain_tumor_dataset"


# In[36]:


#make sure this is the intended data
for file in os.listdir(dataDir):
    print(file)


# In[37]:


#getting a look into our images
Categories = ["yes", "no"]
for category in Categories:
    path = os.path.join(dataDir, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap="gray")
        plt.show()
        break
    break    


# In[38]:


no_dir = r"F:\Materials\AI\brain_tumor_dataset\no"
yes_dir = r"F:\Materials\AI\brain_tumor_dataset\yes"


# In[39]:


img_to_array(load_img(os.path.join(no_dir , os.listdir(no_dir)[5]))).shape


# In[40]:


print(img_array)


# In[41]:


print(img_array.shape)


# In[42]:


#visualisng data
type1=len(os.listdir(dataDir+'/no'))
type2=len(os.listdir(dataDir+'/yes'))

count=[type1,type2]
label=['Tumor','Normal']

sns.barplot(label,count)


# In[43]:


def show_image(folder):
    path=os.path.join(dataDir,folder)
    
    images=choices(os.listdir(path),k=4)
    images=[os.path.join(path,file) for file in images]
    
    return images


# In[44]:


img1=show_image('no')
img2=show_image('yes')
label1=['no']*4
label2=['yes']*4

images=img1+img2
labels=label1+label2

plt.figure(figsize=(16,15))

for i,path_name in enumerate(images):
    plt.subplot(4,2,i+1)
    image=cv2.imread(path_name)
    plt.imshow(image)
    plt.title(labels[i])
    plt.axis('off')


# In[45]:


#Image Augmentation
datagen = ImageDataGenerator(rescale=1/255,
                             rotation_range=20,
                             horizontal_flip=True,
                             height_shift_range=0.1,
                             width_shift_range=0.1,
                             shear_range=0.1,
                             brightness_range=[0.3, 1.5],
                             validation_split=0.2
                            )

train_gen= datagen.flow_from_directory(dataDir,
                                       target_size=(224,224),
                                       class_mode='binary',
                                       subset='training'
                                      )
val_gen = datagen.flow_from_directory( dataDir,
                                       target_size=(224,224),
                                       class_mode='binary',
                                       subset='validation'
                                      )


# In[46]:


#Splitting data
def load_data(dataDir):

    X = []
    y = []
    
    for directory in dataDir:
        for filename in listdir(directory):
            image = cv2.imread(directory+'/'+filename)
            X.append(image)
           
            if directory[-3:] == 'yes':
                y.append([1])
            else:
                y.append([0])
                
    X = np.array(X)
    y = np.array(y)
    X, y = shuffle(X, y)
    
    print(f'Number of examples is: {len(X)}')
    print(f'X shape is: {X.shape}')
    print(f'y shape is: {y.shape}')
    
    return X, y


# In[47]:


def plot_samples(X, y, labels_dict, n=50):
    for index in range(len(labels_dict)):
        imgs = X[np.argwhere(y == index)][:n]
        j = 10
        i = int(n/j)

        plt.figure(figsize=(15,6))
        c = 1
        for img in imgs:
            plt.subplot(i,j,c)
            plt.imshow(img[0])

            plt.xticks([])
            plt.yticks([])
            c += 1
        plt.suptitle('Tumor: {}'.format(labels_dict[index]))
        plt.show()


# In[48]:


X_train, y_train = load_data([no_dir , yes_dir])


# In[50]:


plot_samples(X_train, y_train, ['yes','no'], 30)


# In[51]:


#Converting images to grayscale
def crop_brain_contour(image, plot=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        new_image = image[y:y+h, x:x+w]
        break        

    if plot:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.tick_params(axis='both', which='both', top=False, bottom=False, left=False, right=False,labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(new_image)
        plt.tick_params(axis='both', which='both',top=False, bottom=False, left=False, right=False,labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.title('Cropped Image')
        plt.show()
    
    return new_image


# In[52]:


#putting the images' edited dataset in X,Y
def Croping_Data(train):
    X = []
    y = []
    
    for img in train:
        image = crop_brain_contour(img, plot=False)
        X.append(image)
                
    X = np.array(X)
    
    return X


# In[53]:


X = Croping_Data(X_train)


# In[54]:


plot_samples(X, y_train, ['yes','no'], 30)


# In[55]:


def Resize_Data(train):
    X = []
    y = []
    
    IMG_WIDTH, IMG_HEIGHT = (240, 240)
    
    for img in train:
        image = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        image = image / 255. #standarization
        X.append(image)
                
    X = np.array(X)
    
    return X


# In[56]:


yes_dir =dataDir+'yes'
no_dir = dataDir+'no'

IMG_WIDTH, IMG_HEIGHT = (240, 240)

X = Resize_Data(X)
y = y_train


# In[57]:


plot_samples(X, y_train, ['yes','no'],30)


# In[58]:


def split_data(X, y, test_size=0.2):
       
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# In[59]:


X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)


# In[60]:


print ("number of training examples = " + str(X_train.shape[0]))
print ("number of validation examples = " + str(X_val.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print('Training data and target sizes: \n{}, {}'.format(X_train.shape,y_train.shape))
print('Test data and target sizes: \n{}, {}'.format(X_test.shape,y_test.shape))


# In[61]:


#Visualization splitted_data
y = dict()
y[0] = []
y[1] = []
for set_name in (y_train, y_val, y_test):
    y[0].append(np.sum(set_name == 0))
    y[1].append(np.sum(set_name == 1))

trace0 = go.Bar(
    x=['Train Set', 'Validation Set', 'Test Set'],
    y=y[0],
    name='No',
    marker=dict(color='#33cc33'),
    opacity=0.7
)
trace1 = go.Bar(
    x=['Train Set', 'Validation Set', 'Test Set'],
    y=y[1],
    name='Yes',
    marker=dict(color='#ff3300'),
    opacity=0.7
)
data = [trace0, trace1]
layout = go.Layout(
    title='Count of classes in each set',
    xaxis={'title': 'Set'},
    yaxis={'title': 'Count'}
)
fig = go.Figure(data, layout)
iplot(fig)


# In[62]:


X_train = X_train.reshape(X_train.shape[0], 172800)
y_train = y_train.reshape(y_train.shape[0], )
print('Training data and target sizes: \n{}, {}'.format(X_train.shape,y_train.shape))
X_val = X_val.reshape(X_val.shape[0], 172800)
y_val = y_val.reshape(y_val.shape[0], )
X_test = X_test.reshape(X_test.shape[0], 172800)
y_test = y_test.reshape(y_test.shape[0], )
print('Test data and target sizes: \n{}, {}'.format(X_test.shape,y_test.shape))


# In[63]:


param_grid = [
  {'C': [1, 10], 'kernel': ['linear']},
 ]
svc = svm.SVC()
classifier = GridSearchCV(svc, param_grid, verbose = 3)


# In[64]:


#fitting the data
classifier.fit(X_train, y_train)


# In[65]:


y_predicted = classifier.predict(X_test)


# In[68]:


print("Classification report %s:\n%s\n" #convertting arguments string formatting
      % (classifier, metrics.classification_report(y_test, y_predicted)))

