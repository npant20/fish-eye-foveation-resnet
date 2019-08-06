import import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import Input, Model
import tensorflow.keras.layers
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, Add, Flatten, Dense


def identity_block(x, f, filters, stage, block):
  
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'
  
  F1, F2, F3 = filters
  
  x_shortcut = x
  
  x = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1, 1), padding = 'valid',
            name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(x)
  x = BatchNormalization(axis = 3, name = bn_name_base + '2a')(x)
  x = Activation('relu')(x)
  
  x = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same',
            name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(x)
  x = BatchNormalization(axis = 3, name = bn_name_base + '2b')(x)
  x = Activation('relu')(x)
  
  x = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid',
            name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(x)
  x = BatchNormalization(axis = 3, name = bn_name_base + '2c')(x)
             
  x = Add()([x, x_shortcut])
  x = Activation('relu')(x)
  
  return x
  
  def convolutional_block(x, f, filters, stage, block, s=2):
  
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'
  
  F1, F2, F3 = filters
  
  x_shortcut = x
  
  x = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1, 1), padding = 'valid',
            name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(x)
  x = BatchNormalization(axis = 3, name = bn_name_base + '2a')(x)
  x = Activation('relu')(x)
  
  x = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same',
            name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(x)
  x = BatchNormalization(axis = 3, name = bn_name_base + '2b')(x)
  x = Activation('relu')(x)
  
  x = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid',
            name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(x)
  x = BatchNormalization(axis = 3, name = bn_name_base + '2c')(x)
  
  x_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s, s), padding = 'valid',
                     name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(x_shortcut)
  x_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(x_shortcut)

  x = Add()([x, x_shortcut])
  x = Activation('relu')(x)
  
  return x
  
  def ResNet(input_shape = (64, 64, 3), classes = 200):
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=1)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=1)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=1)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    
    
    # AVGPOOL
    X = AveragePooling2D(pool_size=(2,2), padding='same')(X)

    # Output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model
