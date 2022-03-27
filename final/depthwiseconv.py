import medpy
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import pandas as pd
import numpy as np
import cv2
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense,Flatten,Conv2D,Concatenate,MaxPooling2D,BatchNormalization,Activation,Add,Dropout
from tensorflow.python.keras.models import Model


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
    np_image=np.transpose(np_image,(2,1,0))  
    np_image[np_image<clip_min]=0
    np_image[np_image>clip_max]=0
    
    np_image = np_image/255.0
    np_image=np.abs(np_image)
    np_image=cv2.resize(np_image,(256,256),cv2.INTER_AREA)
    return np_image

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
            
            center = np_image.shape[it] - size
            
        else:
            
            center = (np_image.shape[it] // 2)
        start = center - size
        stop = center + size + (sh % 2)

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

    
    input_image=sitk.ReadImage(mhapath)
    
    input_image = resample(input_image, new_spacing=(1.6, 1.6, 1.6))
    input_image=sitk.GetArrayFromImage(input_image)
    
    input_image = center_crop(input_image, new_shape=(100,512,512))
     
    input_image = clip_and_normalize(input_image)
    
    age=np.expand_dims(age,axis=-1)
    try:
      genders=np.expand_dims(genders,axis=-1)
    except:
      genders=0
      genders=np.expand_dims(genders,axis=-1)

    return input_image,age,genders

def data_(dataset):
  
  dataset=dataset.map(data_preprocess,num_parallel_calls=tf.data.experimental.AUTOTUNE)  
  return  dataset

def data_preprocess(x, labels):
  
  img,age,gender=tf.py_function(mha2jpg,inp=[x],Tout=(tf.float32,tf.float32,tf.float32))
  return {'input_1' :img, 'age':age,'gender':gender}, {'probCOVID':labels[0], 'probSevere' :labels[1]}

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


def play(train_dataset, val_dataset):
  input_=Input(shape=(256,256,100),name='input_1')
  input_shape=(256,256,100)
  input_age=Input(shape=(1,),name='age')
  input_gender=Input(shape=(1,),name='gender')
  input_age=Input(shape=(1,),name='age')
  input_gender=Input(shape=(1,),name='gender')

  inputs = MaxPooling2D((3, 3))(input_)
  x=tf.keras.layers.DepthwiseConv2D((7,7),padding='valid',depth_multiplier=1)(input_)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((3, 3))(x)

  x=tf.keras.layers.DepthwiseConv2D((5,5),padding='valid',depth_multiplier=1)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)



  x=tf.keras.layers.DepthwiseConv2D((3,3),padding='same',depth_multiplier=3)(x)
  x = Conv2D(100, kernel_size=(3, 3),padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(x)
  x_ = Add()([x, inputs])


  

  x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x_)
  x = Flatten()(x_)
  x=Dense(150,activation='relu', kernel_initializer='he_normal')(x)
  #x=BatchNormalization()(x)
  x=Concatenate()([x,input_age,input_gender])
  x=Dense(100,activation='relu', kernel_initializer='he_normal')(x)
  predictions_covid = Dense(1, activation='sigmoid', kernel_initializer='he_normal',name='probSevere')(x) #multi-class
  predictions_severe = Dense(1, activation='sigmoid', kernel_initializer='he_normal',name='probCOVID')(x)



  model = tf.keras.Model(inputs=[input_,input_age,input_gender], outputs=[predictions_covid,predictions_severe])                                                   
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0015), loss={'probCOVID': 'binary_crossentropy', 'probSevere':'binary_crossentropy'},metrics=[tf.keras.metrics.AUC( multi_label=True,curve='ROC', name='auc_pr')],loss_weights=[20,1])
  callbacks = [
    tf.keras.callbacks.ModelCheckpoint('/home/jhjoo/grand_challenge/new_0116_{epoch:02d}.h5', verbose=True, save_weights_only=True, mode='auto'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=2, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0),
]
#loss_weights 조절/ 가중치 줄이는거 probcovid로하자 전처리는 대회사이트에 올라와있는걸로 하고
  model.fit(train_dataset, epochs=50,validation_data=val_dataset,callbacks=callbacks)

if __name__=="__main__":
  data_list,tr=di_preprocess()
  train_dataset,val_dataset=make_data(data_list,tr)
  play(train_dataset,val_dataset)