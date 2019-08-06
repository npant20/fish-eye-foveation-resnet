import tensorflow as tf
import numpy as np
import tensorflow.keras
from .resnet import ResNet
from .image_warp import warp_image
from .data_loader import get_data, get_id_dictionary

def train_preprocess0(image, label):
  image = warp_image(image, 64, dxc = 0, dyc = 0)
  return image, label
  
  
def train_preprocess1(image, label):
  image = warp_image(image, 64, dxc = 17, dyc = 17)
  return image, label
  
  
def train_preprocess2(image, label):
  image = warp_image(image, 64, dxc = 17, dyc = -17)
  return image, label
  
  
def train_preprocess3(image, label):
  image = warp_image(image, 64, dxc = -17, dyc = 17)
  return image, label
  
  
def train_preprocess4(image, label):
  image = warp_image(image, 64, dxc = -17, dyc = -17)
  return image, label
  
  
train_data, train_labels, val_data, val_labels, test_data, test_labels = get_data(get_id_dictionary())

dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
dataset = dataset.shuffle(100000)

dataset0 = dataset.map(train_preprocess0, num_parallel_calls = 4)
dataset1 = dataset.map(train_preprocess1, num_parallel_calls = 4)
dataset2 = dataset.map(train_preprocess2, num_parallel_calls = 4)
dataset3 = dataset.map(train_preprocess3, num_parallel_calls = 4)
dataset4 = dataset.map(train_preprocess4, num_parallel_calls = 4)

dataset = dataset0.concatenate(dataset1) 
dataset = dataset.concatenate(dataset2) 
dataset = dataset.concatenate(dataset3) 
dataset = dataset.concatenate(dataset4)  

dataset = dataset.batch(64)
dataset = dataset.prefetch(1)

model = ResNet(input_shape = (64, 64, 3), classes = 200)
model.compile(optimizer ='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(dataset, epochs = 100, verbose = 1, validation_data = (val_data, val_labels), shuffle=True)
