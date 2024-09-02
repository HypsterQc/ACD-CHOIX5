import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.src.metrics import Precision, Recall

img_size = [64]
epochs = [5]
batch_size = [16]
complexity_levels = [1]


def create_model(input_shape, num_classes, complexity_level=None):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])
    return model
