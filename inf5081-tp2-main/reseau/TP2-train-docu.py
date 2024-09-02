import psutil
import tensorflow as tf
from keras.src.metrics import Recall, Precision
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D

tf.config.set_visible_devices([], 'GPU')


def adjust_configuration_limits():
    available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
    print(f"Available Memory: {available_memory_gb:.2f} GB")

    img_sizes = [32, 64]
    epochs_list = [10, 20]
    batch_sizes = [8, 16]
    complexity_levels = [1]

    return img_sizes, epochs_list, batch_sizes, complexity_levels


def create_model(input_shape, num_classes, complexity_level):
    inputs = Input(shape=input_shape)
    x = inputs

    for i in range(complexity_level):
        x = Conv2D(32 * (i + 1), (3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        if i % 2 == 1:
            x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])
    return model
