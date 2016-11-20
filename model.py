from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Input, Dense, Activation
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.models import Model
from keras import backend as K
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.utils.np_utils import to_categorical
from keras.callbacks import CSVLogger

def VGG_19_GAP_functional(weights_path=None,heatmap=False):

    inputs = Input(shape=(3, None, None))

    x = Convolution2D(64, 3, 3, activation='relu',border_mode='same',name='conv1_1')(inputs)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(64, 3, 3, activation='relu',name='conv1_2')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128, 3, 3, activation='relu',name='conv2_1')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128, 3, 3, activation='relu',name='conv2_2')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256, 3, 3, activation='relu',name='conv3_1')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256, 3, 3, activation='relu',name='conv3_2')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256, 3, 3, activation='relu',name='conv3_3')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256, 3, 3, activation='relu',name='conv3_4')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv4_1')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv4_2')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv4_3')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv4_4')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv5_1')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv5_2')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv5_3')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv5_4')(x)

    x = ZeroPadding2D((1,1))(x)
    final_conv = Convolution2D(1024, 3, 3, activation='relu',name='conv6_1')(x)

    x = GlobalAveragePooling2D()(final_conv)

    main_output = Dense(2, activation = 'softmax', name="main_output")(x)
    aux_output = final_conv


    if heatmap:
        model = Model(input=inputs, output=[main_output,aux_output])
    else:
        model = Model(input=inputs, output=main_output)#[main_output,aux_output])

    if weights_path:
        model.load_weights(weights_path,by_name=True)

    return model

def VGG_19(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model