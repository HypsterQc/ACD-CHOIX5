import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, Dropout, \
    BatchNormalization, Activation
from keras.src.metrics import Precision, Recall
tf.config.set_visible_devices([], 'GPU')

img_size = [32]
epochs = [20]
batch_size = [16]
complexity_levels = [1]


def inception_module(x, filters):
    conv1 = Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
    conv3 = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    conv5 = Conv2D(filters, (5, 5), padding='same', activation='relu')(x)

    x = Concatenate()([conv1, conv3, conv5])
    x = BatchNormalization()(x)
    return x


def create_model(input_shape, num_classes, complexity_level=None):
    inputs = Input(shape=input_shape)

    x = inception_module(inputs, 64)
    x = MaxPooling2D((2, 2))(x)

    x = inception_module(x, 128)
    x = MaxPooling2D((2, 2))(x)

    x = inception_module(x, 256)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])
    return model