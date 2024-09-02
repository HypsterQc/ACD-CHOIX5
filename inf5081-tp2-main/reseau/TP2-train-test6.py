import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Add, \
    Activation
from keras.src.metrics import Precision, Recall
tf.config.set_visible_devices([], 'GPU')

img_size = [32]
epochs = [20]
batch_size = [16]
complexity_levels = [1]


def residual_block(x, filters, kernel_size=(3, 3), padding='same', strides=1):
    res = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    res = BatchNormalization()(res)
    res = Activation('relu')(res)
    res = Conv2D(filters, kernel_size, padding=padding, strides=strides)(res)
    res = BatchNormalization()(res)

    shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = BatchNormalization()(shortcut)

    res = Add()([res, shortcut])
    res = Activation('relu')(res)
    return res


def create_model(input_shape, num_classes, complexity_level=None):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), padding='same', strides=2, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = residual_block(x, 64)
    x = MaxPooling2D((2, 2))(x)

    x = residual_block(x, 128)
    x = MaxPooling2D((2, 2))(x)

    x = residual_block(x, 256)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])
    return model