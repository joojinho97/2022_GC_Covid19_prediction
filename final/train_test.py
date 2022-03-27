import medpy
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import pandas as pd
import numpy as np
import cv2
import SimpleITK as sitk
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense,Flatten,Conv2D,Concatenate,MaxPooling2D,BatchNormalization,Activation,Add,Dropout,Conv3D
from tensorflow.python.keras.models import Model

from tensorflow.keras.utils import get_custom_objects

from focal_loss import BinaryFocalLoss


def precision_0(y_true,y_pred):
  y_true = tf.cast(y_true,dtype=tf.float32)
  y_pred = tf.cast(y_pred,dtype=tf.float32)
  y_true=y_true-1
  y_pred=y_pred-1
  y_true=tf.math.abs(y_true)
  y_pred=tf.math.abs(y_pred)
  right=tf.math.reduce_sum(float(y_true*y_pred)>=0.5)
  return right/tf.math.reduct_sum(y_pred)
def di_preprocess():
  
  os.chdir('/mount/HDD/JJH/grand_challenge/mha/')
  data_list=glob.glob('*.mha')
  data_list.sort(key=lambda fname: int(fname.split('.')[0]))
  for i in range(len(data_list)):
    data_list[i]='/mount/HDD/JJH/grand_challenge/mha/'+data_list[i]
  data=pd.read_csv('/mount/HDD/JJH/grand_challenge/metadata/reference.csv')
  data=data.sort_values(by='PatientID')
  tr=list(zip(data["probCOVID"].tolist(), data["probSevere"].tolist()))
  return data_list, tr

from typing import Iterable

def clip_and_normalize(np_image: np.ndarray,
                       clip_min: int = -703,
                       clip_max: int = -368
                       ) -> np.ndarray:
                       
    '''
    np_image=np.transpose(np_image,(1,2,0))  
    
    np_image_or=np_image
    
    np_image[np_image<clip_min]=-1450
    np_image[np_image>clip_max]=-1450
    
    
    
    np_image = (np_image - np_image.min()) / (np_image.max() - np_image.min())
    
    
    np_image1=cv2.rotate(np_image, cv2.ROTATE_180)
    
    np_image2=np.transpose(np_image,(1,0,2)) 
    np_image3=cv2.rotate(np_image2, cv2.ROTATE_180) 
    '''
    
    
    np_image=np.transpose(np_image,(1,2,0))
    
    np_image[np_image<clip_min]=-1450
    np_image[np_image>clip_max]=-1450
    
    
    
    np_image = (np_image - np_image.min()) / (np_image.max() - np_image.min())
    
    np_image1=np_image.sum(axis=-1)
    np_image1=np.stack((np_image1,np_image1,np_image1),axis=-1)
    
    
    
    
    
    
    
    return np_image,np_image1

def resample(itk_image: sitk.Image,
             new_spacing: Iterable[float],
             outside_val: float = 0
             ) -> sitk.Image:
             
    shape = itk_image.GetSize()
    spacing = itk_image.GetSpacing()
    output_shape = tuple(int(round(s * os / ns)) for s, os, ns in zip(shape, spacing, new_spacing))
    return sitk.Resample(
        itk_image,
        output_shape,
        sitk.Transform(),
        sitk.sitkLinear,
        itk_image.GetOrigin(),
        new_spacing,
        itk_image.GetDirection(),
        outside_val,
        sitk.sitkFloat32,
    )


def center_crop(np_image: np.ndarray,
                new_shape: Iterable[int],
                outside_val: float = 0
                ):
  
    output_image = np.full(new_shape, outside_val, np_image.dtype)

    slices = tuple()
    offsets = tuple()
    
    for it, sh in enumerate(new_shape):
        size = sh // 2
        if it == 0:
            
            center = np_image.shape[it] //2
            
        else:
            
            center = (np_image.shape[it] // 2)
        start = center - size
        stop = center + size

        # computing what area of the original image will be in the cropped output
        slce = slice(max(0, start), min(np_image.shape[it], stop))
        slices += (slce,)

        # computing offset to pad if the crop is partly outside of the scan
        offset = slice(-min(0, start), 2 * size - max(0, (start + 2 * size) - np_image.shape[it]))
        offsets += (offset,)

    output_image[offsets] = np_image[slices]

    return output_image


def preprocess(input_image: sitk.Image,
               new_spacing: Iterable[float] = (1.6, 1.6, 1.6),
               new_shape: Iterable[int] = (240, 240, 240),
               ) -> np.ndarray:

    input_image = resample(input_image, new_spacing=new_spacing)
    input_image = sitk.GetArrayFromImage(input_image)
    input_image = center_crop(input_image, new_shape=new_shape)
    input_image = clip_and_normalize(input_image)
    return input_image

def mha2jpg(mhapath):
        
    mhapath=str(mhapath.numpy())[2:-1]
    '''
    reader = sitk.ImageFileReader()
    reader.SetFileName(mhapath)   # Give it the mha file as a string
    reader.LoadPrivateTagsOn()     # Make sure it can get all the info
    reader.ReadImageInformation()
    
    try:
      gender=reader.GetMetaData('PatientSex')
      if gender=='M':
        genders=float(0)
      elif gender=='F':
        genders=float(5)
    except:
      genders=float(1)
    
    try:
      age=reader.GetMetaData('PatientAge')
      age=float(age)
      
      age=int(age)
      if age<40:
        age=0
      elif age<50:
        age=1
      elif age<60:
        age=2
      elif age<70:
        age=3
      elif age<80:
        age=4
      else:
        age=5
    except:
      if age=='035Y':
        age=0
      elif age=='045Y':
        age=1
      elif age=='055Y':
        age=2
      elif age=='065Y':
        age=3
      elif age=='075Y':
        age=4
      elif age=='085Y':
        age=5
    '''
    
    input_image=sitk.ReadImage(mhapath)
    
    input_image = resample(input_image, new_spacing=(1.6,1.6,1.6))
    input_image=sitk.GetArrayFromImage(input_image)
    
    input_image = center_crop(input_image, new_shape=(160,190,190))
     
    input_image,input_image1 = clip_and_normalize(input_image)
    
    '''
    age=np.expand_dims(age,axis=-1)
    try:
      genders=np.expand_dims(genders,axis=-1)
    except:
      genders=0
      genders=np.expand_dims(genders,axis=-1)
    
    '''
    return input_image, input_image1
    #,age,genders

def data_(dataset):
  
  dataset=dataset.map(data_preprocess,num_parallel_calls=tf.data.experimental.AUTOTUNE)  
  return  dataset

def data_preprocess(x, labels):
  
  img,img1=tf.py_function(mha2jpg,inp=[x],Tout=(tf.float64,tf.float64))
  
  return {'input_1' :img,'input_2' :img1}, {'probCOVID':labels[0], 'probSevere' :labels[1]}

def make_dataset(train_list, tr):
  
  test_flist=tf.data.Dataset.from_tensor_slices((train_list, tr))
  test_dataset=data_(test_flist)
  test_dataset=test_dataset.batch(7)
  test_dataset=test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
  
  return test_dataset


def make_data(data_list,tr):
  train_list=data_list[:1400]
  val_list=data_list[1401:1800]
  train_label=tr[:1400]
  val_label=tr[1401:1800]
  train_dataset=make_dataset(train_list, train_label)
  val_dataset=make_dataset(val_list,val_label)
  
  return train_dataset, val_dataset
  
def make_test_data(data_list,tr):
  train_list=data_list[1801:1999]

  train_label=tr[1801:1999]
 
  train_dataset=make_dataset(train_list, train_label)
  
  
  return train_dataset
  
  
def identity_block(X, f, filters, stage, block):
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'
  F1, F2, F3 = filters
  X_shortcut = X
  # first step of main path
  X = tf.keras.layers.Conv2D(filters=F1, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2a',
  kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
  X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base+'2a')(X)
  X = tf.keras.layers.Activation('relu')(X)
  # second step of main path
  X = tf.keras.layers.Conv2D(filters=F2, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b',
  kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
  X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base+'2b')(X)
  X = tf.keras.layers.Activation('relu')(X)
  # third step of main path
  X = tf.keras.layers.Conv2D(filters=F3, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2c',
  kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
  
  X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base+'2c')(X)
  # add shortcut value and pass it through a ReLU activation
  X = tf.keras.layers.Add()([X, X_shortcut])
  X = tf.keras.layers.Activation('relu')(X)
  return X


  
  
def convolutional_block(X, f, filters, stage, block, s=2):
  conv_name_base = 'res'+str(stage)+block+'_branch'
  bn_name_base = 'bn'+str(stage)+block+'_branch'
  F1, F2, F3 = filters
  X_shortcut = X
  # first step of main path
  X = tf.keras.layers.Conv2D(filters=F1, kernel_size=1, strides=s, padding='valid', name=conv_name_base+'2a',
  kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
  X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base+'2a')(X)
  X = tf.keras.layers.Activation('relu')(X)
  # second step of main path
  X = tf.keras.layers.Conv2D(filters=F2, kernel_size=f, strides=1, padding='same', name=conv_name_base+'2b',
  kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
  X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base+'2b')(X)
  X = tf.keras.layers.Activation('relu')(X)
  # third step of main path
  X = tf.keras.layers.Conv2D(filters=F3, kernel_size=1, strides=1, padding='valid', name=conv_name_base+'2c',
  kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
  X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base+'2c')(X)
  # shortcut path
  X_shortcut = tf.keras.layers.Conv2D(filters=F3, kernel_size=1, strides=s, padding='valid', name=conv_name_base+'1',
  kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X_shortcut)
  X_shortcut = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base+'1')(X_shortcut)
  # Add and pass it through a ReLU activation
  X = tf.keras.layers.Add()([X, X_shortcut])
  X = tf.keras.layers.Activation('relu')(X)
  return X


  
def ResNet50(input_shape=(190,190,10)):
  input_1 =Input(shape=(190,190,160),name='input_1')
  input_2 =Input(shape=(190,190,3),name='input_2')
  
  
  # zero padding
  X = tf.keras.layers.ZeroPadding2D((3,3))(input_1)
  # stage 1
  X = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, name='conv1',
  kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
  X = tf.keras.layers.BatchNormalization(axis=3, name='bn_conv1')(X)
  X = tf.keras.layers.Activation('relu')(X)
  X = tf.keras.layers.MaxPooling2D((3,3), strides=(2,2))(X)
  # stage 2
  X = convolutional_block(X, f=3, filters=[64,64,256], stage=2, block='a', s=1)
  X = identity_block(X, 3, [64,64,256], stage=2, block='b')
  X = identity_block(X, 3, [64,64,256], stage=2, block='c')
  # stage 3
  X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
  X = identity_block(X, 3, [128, 128, 512], stage = 3, block='b')
  X = identity_block(X, 3, [128, 128, 512], stage = 3, block='c')
  X = identity_block(X, 3, [128, 128, 512], stage = 3, block='d')
  
  # Stage 4
  X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
  X = identity_block(X, 3, [256, 256, 1024], stage = 4, block='b')
  X = identity_block(X, 3, [256, 256, 1024], stage = 4, block='c')
  X = identity_block(X, 3, [256, 256, 1024], stage = 4, block='d')
  X = identity_block(X, 3, [256, 256, 1024], stage = 4, block='e')
  X = identity_block(X, 3, [256, 256, 1024], stage = 4, block='f')
  
  # Stage 5
  X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
  X = identity_block(X, 3, [512, 512, 2048], stage = 5, block='b')
  X = identity_block(X, 3, [512, 512, 2048], stage = 5, block='c')
  # AVGPOOL
  X = tf.keras.layers.GlobalAveragePooling2D()(X)
  # output layer
  X = tf.keras.layers.Flatten()(X)
  
  
  
  
  X1 = tf.keras.layers.ZeroPadding2D((3,3))(input_2)
  # stage 1
  X1 = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, name='codsnv1',
  kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X1)
  X1 = tf.keras.layers.BatchNormalization(axis=3, name='bnas_conv1')(X1)
  X1 = tf.keras.layers.Activation('relu')(X1)
  X1 = tf.keras.layers.MaxPooling2D((3,3), strides=(2,2))(X1)
  # stage 2
  X1 = convolutional_block(X1, f=3, filters=[64,64,256], stage=2, block='auy', s=1)
  X1 = identity_block(X1, 3, [64,64,256], stage=2, block='wb')
  X1 = identity_block(X1, 3, [64,64,256], stage=2, block='qc')
  # stage 3
  X1 = convolutional_block(X1, f = 3, filters = [128, 128, 512], stage = 3, block='abb', s = 2)
  X1 = identity_block(X1, 3, [128, 128, 512], stage = 3, block='rb')
  X1 = identity_block(X1, 3, [128, 128, 512], stage = 3, block='qc')
  X1 = identity_block(X1, 3, [128, 128, 512], stage = 3, block='dd')
  
  # Stage 4
  X1 = convolutional_block(X1, f = 3, filters = [256, 256, 1024], stage = 4, block='qwera', s = 2)
  X1 = identity_block(X1, 3, [256, 256, 1024], stage = 4, block='sb')
  X1 = identity_block(X1, 3, [256, 256, 1024], stage = 4, block='cb')
  X1 = identity_block(X1, 3, [256, 256, 1024], stage = 4, block='asdd')
  X1 = identity_block(X1, 3, [256, 256, 1024], stage = 4, block='ef')
  X1 = identity_block(X1, 3, [256, 256, 1024], stage = 4, block='fasd')
  
  # Stage 5
  X1 = convolutional_block(X1, f = 3, filters = [512, 512, 2048], stage = 5, block='aasdf', s = 2)
  X1 = identity_block(X1, 3, [512, 512, 2048], stage = 5, block='bs')
  X1 = identity_block(X1, 3, [512, 512, 2048], stage = 5, block='ca')
  #1 AVGPOOL
  X1 = tf.keras.layers.GlobalAveragePooling2D()(X1)
  # output layer
  X1 = tf.keras.layers.Flatten()(X1)
 
  
  
  X= Concatenate()([X,X1])
  

  predictions_covid_ = Dense(128,activation='relu', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
  predictions_severe_ = Dense(128,activation='relu', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
  
 
  

  
  predictions_covid_ = Dense(1,activation='sigmoid', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),name='probCOVID')(predictions_covid_)
  predictions_severe_ = Dense(1,activation='sigmoid', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(predictions_severe_)
  

  predictions_severe_00 = tf.keras.layers.Add()([predictions_covid_*0.99,predictions_covid_*0.01])
  predictions_severe_S = tf.keras.backend.switch(predictions_covid_>=0.5,predictions_severe_,predictions_severe_00)
  
  predictions_severe_= tf.keras.layers.Activation('relu',name='probSevere')(predictions_severe_S)
  
  '''
  X_severe = Dense(1, activation='sigmoid',kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X) #multi-class
  X_covid = Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
  
  predictions_severe_ = tf.keras.layers.Add()([X_severe*0.8,X4_severe*0.2])
  predictions_covid_  = tf.keras.layers.Add()([X_covid*0.8,X4_covid*0.2])
  
  predictions_severe_= tf.keras.layers.Activation('relu',name='probSevere')(predictions_severe_)
  predictions_covid_ = tf.keras.layers.Activation('relu',name='probCOVID')(predictions_covid_)
  '''
  
 
  
  
  
  
  
  
  
  model = tf.keras.Model(inputs=[input_1,input_2], outputs=[predictions_severe_,predictions_covid_])   
  return model

def BCEE(y_true, y_pred):
  y_true = tf.cast(y_true,dtype=tf.float32)
  y_pred = tf.cast(y_pred,dtype=tf.float32)
  
  
  delta=1e-7
  pred_diff=1-y_pred
  label_diff=1-y_true
  
  delta=tf.cast(delta,dtype=tf.float32)
  pred_diff=tf.cast(pred_diff,dtype=tf.float32)
  label_diff=tf.cast(label_diff,dtype=tf.float32)
  
  result= -y_true*tf.math.log(y_pred+delta)-label_diff*tf.math.log(pred_diff+delta)-label_diff*tf.math.log(pred_diff+delta)-label_diff*tf.math.log(pred_diff+delta)-label_diff*tf.math.log(pred_diff+delta)-label_diff*tf.math.log(pred_diff+delta)+delta*5
  
  result=tf.cast(result,dtype=tf.float32)
  return result
  
def BCEE2(y_true,y_pred):
  y_pred = tf.cast(y_pred,dtype=tf.float32)
  y_true = tf.cast(y_true,dtype=tf.float32)

  
  pred_diff=1-y_pred
  label_diff=1-y_true
  result= tf.reduce_sum(-y_true*tf.math.log(y_pred+1e-7)*10-label_diff*tf.math.log(pred_diff+1e-7))/float(len(y_true))
  
  return result

def BCEE2_(y_true,y_pred):
  y_pred = tf.cast(y_pred,dtype=tf.float32)
  y_true = tf.cast(y_true,dtype=tf.float32)

  
  pred_diff=1-y_pred
  label_diff=1-y_true
  result= tf.reduce_sum(-y_true*tf.math.log(y_pred+1e-7)-label_diff*tf.math.log(pred_diff+1e-7))/float(len(y_true))
  
  return result
def harmony(y_true, y_pred):
  y_true = tf.cast(y_true,dtype=tf.float32)
  y_pred = tf.cast(y_pred,dtype=tf.float32)
  
  y_pred=tf.clip_by_value(y_pred,1e-7,1-1e-7)
  y_true=tf.clip_by_value(y_true,1e-7,1-1e-7)
  
  
  label_diff=1+1e-7-y_true
  
  
  
  label_diff=tf.cast(label_diff,dtype=tf.float32)
  
  a=(2*y_true*y_pred)/(y_true+y_pred)
  
  result= -(1e-7+y_true)*tf.math.log(a)-label_diff*tf.math.log(1-a)
  
  result=tf.cast(result,dtype=tf.float32)
  return result
  

def harmony2(y_true, y_pred):
  y_true = tf.cast(y_true,dtype=tf.float32)
  y_pred = tf.cast(y_pred,dtype=tf.float32)
  y_trued = y_true
  label_diff=1-y_true
  y_pred=tf.clip_by_value(y_pred,1e-3,1-1e-7)
  y_true=tf.clip_by_value(y_true,1e-3,1-1e-7)
  
  
  
  
  label_diff=tf.cast(label_diff,dtype=tf.float32)
  
  a=(2*y_true*y_pred)/(y_true+y_pred)
  
  result= -(y_trued)*tf.math.log(a)-label_diff*tf.math.log(1-a)
  
  result=tf.cast(result,dtype=tf.float32)
  return result
''' 
def harmony3(y_true, y_pred):
  y_true = tf.cast(y_true,dtype=tf.float32)
  y_pred = tf.cast(y_pred,dtype=tf.float32)
  
  minus=tf.math.abs(y_true-y_pred)
  
  result= tf.math.reduce_sum(-(y_true)*((1-y_pred)**2)*(minus+1e-4)*tf.math.log(tf.math.sin((minus-0.5)**6)+1e-12) -(1-y_true)*((y_pred)**2)*(minus+1e-4)*tf.math.log(tf.math.sin((minus-0.5)**6)+1e-12))/float(len(y_pred))
  
  
  return result
  
def harmony3_(y_true, y_pred):
  y_true = tf.cast(y_true,dtype=tf.float32)
  y_pred = tf.cast(y_pred,dtype=tf.float32)
  
  minus=tf.math.abs(y_true-y_pred)
  
  result= tf.math.reduce_sum(-(y_true)*((1-y_pred)**2)*(minus+1e-4)*tf.math.log(tf.math.sin((minus-0.5)**6)+1e-12) -(1-y_true)*((y_pred)**2)*(minus+1e-4)*tf.math.log(tf.math.sin((minus-0.5)**6)+1e-12))/float(len(y_pred))
  
  
  return result

'''

def harmony3(y_true, y_pred):
  y_true = tf.cast(y_true,dtype=tf.float32)
  y_pred = tf.cast(y_pred,dtype=tf.float32)
  
  minus=tf.math.abs(y_true-y_pred)+1e-12
  minus_p=tf.math.abs(y_true-y_pred)+1e-12
  result= tf.math.reduce_sum(-((y_true)*((1-y_pred)**2)*((y_pred)**2)*tf.math.log(tf.math.sin(((minus-0.5)*(minus-0.7))**6)+1e-12)*8)-((1-y_true)*((y_pred)**2)*((1-y_pred)**2)*tf.math.log(tf.math.sin(((minus_p-0.5)*(minus_p-0.7))**6)+1e-12)*8))/(float(len(y_pred))+1e-12)
  
  return result

  
def harmony3_(y_true, y_pred):
  y_true = tf.cast(y_true,dtype=tf.float32)
  y_pred = tf.cast(y_pred,dtype=tf.float32)
  
  minus=tf.math.abs(y_true-y_pred)+1e-12
  minus_p=tf.math.abs(y_true-y_pred)+1e-12
  result= tf.math.reduce_sum(-(y_true)*((1-y_pred)**2)*((y_pred)**2)*tf.math.log(tf.math.sin(((minus-0.5)*(minus-0.7))**6)+1e-12)*8-(1-y_true)*((y_pred)**2)*((1-y_pred)**2)*tf.math.log(tf.math.sin(((minus_p-0.5)*(minus_p-0.7))**6)+1e-12)*8)/(float(len(y_pred))+1e-12)
  
  return result
def mse_jh(y_true, y_pred):
    y_true = tf.cast(y_true,dtype=tf.float32)
    y_pred = tf.cast(y_pred,dtype=tf.float32)
    a=(2*y_true*y_pred)/(y_true+y_pred)
    
    return (a-y_pred)**2

def play(train_dataset, val_dataset):
  #input_1=Input(shape=(80,190,190),name='input_1')
  
  model=ResNet50()
  


  
  
  
  
  #'probCOVID': BCEE2_, 'probSevere':BCEE2
  
  
  model.summary()  
  #model.load_weights('/home/jhjoo/grand_challenge/best/en/another/ensemble_04.h5')
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss={'probCOVID': BCEE2_, 'probSevere':BCEE2},metrics=[tf.keras.metrics.AUC(curve='ROC', name='auc_pr'),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
  callbacks = [
    tf.keras.callbacks.ModelCheckpoint('/home/jhjoo/grand_challenge/best/en/another/ensemble_{epoch:02d}.h5', verbose=True, save_weights_only=True, mode='auto'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=200, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0),
]
  #model.load_weights("/home/jhjoo/grand_challenge/best/en/another/ensemble_02.h5")
  
#loss_weights 조절/ 가중치 줄이는거 probcovid로하자 전처리는 대회사이트에 올라와있는걸로 하고
  model.fit(train_dataset, epochs=300,validation_data=val_dataset,callbacks=callbacks)                   
  #BinaryFocalLoss(gamma=2)
  
def make_test_data(data_list,tr):
  train_list=data_list[1801:1999]

  train_label=tr[1801:1999]
 
  train_dataset=make_dataset(train_list, train_label)
  
  
  return train_dataset
  
def test(test_dataset):
  #input_1=Input(shape=(80,190,190),name='input_1')
  
  model=ResNet50()
  
  #model.summary()  
  
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss={'probCOVID': BCEE2_, 'probSevere':'binary_crossentropy'},metrics=[tf.keras.metrics.AUC(curve='ROC', name='auc_pr'),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
  
    
  model.load_weights("/home/jhjoo/grand_challenge/best/en/another/7517/ensemble_13.h5")
  model.evaluate(test_dataset)
if __name__=="__main__":
  os.environ["CUDA_VISIBLE_DEVICES"]=''

  '''
  data_list,tr=di_preprocess()
  train_dataset,val_dataset=make_data(data_list,tr)
  
  play(train_dataset,val_dataset)
  '''
  
  data_list,tr=di_preprocess()
  test_dataset=make_test_data(data_list,tr)
  test(test_dataset)
  
  
