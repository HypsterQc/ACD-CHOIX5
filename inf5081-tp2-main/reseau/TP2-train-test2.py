import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import PIL
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from keras.src.metrics import Precision, Recall
from sklearn.model_selection import train_test_split

from TP2 import select_equal_random_images, get_all_jpeg_images_by_folder, predict_image

translate = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel"}
anmls_list = ["dog", 'cat', 'horse', 'spyder', 'butterfly', 'chicken', 'sheep', 'cow', 'squirrel', 'elephant']

def evaluate_model_with_confusion_matrix(model, X_test, y_test, class_names, is_fursuit_model=False):
    if len(y_test.shape) > 1 and not is_fursuit_model:
        true_classes = np.argmax(y_test, axis=1)
    else:
        true_classes = y_test

    predictions = model.predict(X_test)

    if is_fursuit_model:
        predicted_classes = (predictions > 0.5).astype("int32").flatten()
    else:
        predicted_classes = np.argmax(predictions, axis=1)

    cm = confusion_matrix(true_classes, predicted_classes)
    row_sums = np.sum(cm, axis=1, keepdims=True)
    cm_percent = (cm / row_sums) * 100

    percent_success = np.diag(cm_percent)
    percent_success_col = percent_success.reshape(-1, 1)
    cm_with_percent_col = np.hstack([cm_percent, percent_success_col])

    mean_percent_success = np.mean(percent_success)
    percent_row = np.append(percent_success, mean_percent_success)
    cm_with_summary = np.vstack([cm_with_percent_col, percent_row])

    if is_fursuit_model:
        class_names_with_total = class_names + ['Taux de Réussite']
        column_names_with_total = class_names + ['Pourcentage']
    else:
        class_names_with_total = class_names + ['Taux de Réussite']
        column_names_with_total = class_names + ['Pourcentage']

    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(cm_with_summary, annot=True, fmt='.2f', cmap='viridis',
                          xticklabels=column_names_with_total, yticklabels=class_names_with_total)
    plt.xlabel('Prédiction')
    plt.ylabel('Vérité')
    plt.title('Matrice de Confusion en Pourcentages avec Taux de Réussite')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

# Translating animal file titles
raw_img_path = '../TP2-images/espece/'

anmls_dict = {'Class':[], 'Count':[]}

#extracting data about classes
for img_class in os.listdir(raw_img_path):
    if img_class in translate.keys():
        anmls_dict['Class'].append(translate[img_class])
        anmls_dict['Count'].append(min(100,len(os.listdir(raw_img_path + img_class))))
    else:
        anmls_dict['Class'].append(img_class)
        anmls_dict['Count'].append(min(100,len(os.listdir(raw_img_path + img_class))))



train_labels = []
i = -1
for anml_class in anmls_dict['Class']:
    #global i
    i+=1
    train_labels += [i]*min(100, anmls_dict['Count'][i])

train_labels = np.array(train_labels)

height, width = 64, 64

train_features = np.empty((0, height, width, 3))

from IPython.display import clear_output

m = 0
max_images_per_folder = 100
for img_class in anmls_dict['Class']:
    #global train_features
    #global m
    for key, value in translate.items():
        if value == img_class:
            trgt_folder = key

    images_processed = 0  # Compteur d'images traitées pour chaque dossier
    for img_data in os.listdir(raw_img_path + trgt_folder):
        if images_processed >= max_images_per_folder:
            break  # Sortir de la boucle si le nombre max d'images est atteint

        new_path = raw_img_path + trgt_folder + "/" + img_data
        img = tf.keras.preprocessing.image.load_img(new_path,
                                                    color_mode="rgb",
                                                    target_size=(height, width))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img / 255.0
        train_features = np.append(train_features, img[np.newaxis, ...], axis=0)
        images_processed += 1  # Incrémenter le compteur d'images
        m += 1
        clear_output()  # Effacer la sortie actuelle
        print(m, '\n', train_features.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),
                                 strides=(1, 1),
                                 padding='valid',
                                 activation="relu",
                                 input_shape=(height,width,3)), # input layer (1)
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3)), # hidden layer (2)
    tf.keras.layers.Flatten(),  # hidden layer (3)
    tf.keras.layers.Dense(128, activation='relu'),  # hidden layer (4)
    tf.keras.layers.Dense(64, activation='relu'),  # hidden layer (5)
    tf.keras.layers.Dense(10, activation='softmax') # output layer (6)
])
model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy',
             #Precision(name='precision'),
             #Recall(name='recall')
             ]
)

print(f"train_features shape: {train_features.shape}")
print(f"train_labels shape: {train_labels.shape}")
#X_train_fursuit, X_test_fursuit, y_train_fursuit, y_test_fursuit
train_features, test_features,train_labels, test_labels = train_test_split(train_features, train_labels,
                                                                           test_size=0.2, random_state=42)

history = model.fit(train_features , train_labels ,
                    validation_data=(test_features, test_labels),
                    epochs = 10)

metrics = ['accuracy', 'precision', 'recall', 'loss', 'precision_2', 'recall_2']
for metric in metrics:
    if metric in history.history:
        plt.plot(history.history[metric], label=metric)
        plt.plot(history.history[f'val_{metric}'], label=f'val_{metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend(loc='lower right')
        plt.show()
evaluate_model_with_confusion_matrix(model, test_features, test_labels, anmls_list)




probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

# SÃ©lectionner alÃ©atoirement 10 images par dossier pour les tests
images_by_folder = get_all_jpeg_images_by_folder(raw_img_path)
filtered_images_by_folder = {folder_name: images for folder_name, images in images_by_folder.items() if folder_name != "Fursuit"}
test_images = select_equal_random_images(filtered_images_by_folder, num_images_per_folder=10)
bonneReponse = 0
for file in test_images:
    #anmls_dict['Class'][predicted_classes[0]]
    result2, prob2 = predict_image(model, file, 64, anmls_list)
    espece = result2

    print(f"Nom image : {file}")
    print(f"Prediction: {translate[file.replace('\\', '/').split('/')[2]]}")
    if translate[file.replace('\\', '/').split('/')[2]] == espece:
        bonneReponse += 1

    print(f"Espece : {espece}")
print(f"RÃ©sultat: {bonneReponse}/{len(test_images)}, {round((bonneReponse / len(test_images)) * 100, 2)}%")



test_img1_path = '../TP2-images/espece/cane/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg'
test_img1 = tf.keras.preprocessing.image.load_img(test_img1_path,
                                             #grayscale=False,
                                             color_mode="rgb",
                                             target_size=(height, width))
test_img1 = tf.keras.preprocessing.image.img_to_array(test_img1)
test_img1 = test_img1/255.0

test_imgs = np.empty((0, height, width, 3))
test_imgs = np.append(test_imgs, [test_img1], axis=0)
test_imgs.shape

# Predict
predicted_class1 = probability_model.predict(test_imgs)

predicted_classes = np.argmax(predicted_class1, axis=1)
print('Predicted class: ', anmls_dict['Class'][predicted_classes[0]])