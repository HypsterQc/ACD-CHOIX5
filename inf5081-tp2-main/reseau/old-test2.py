import os
import random
from collections import defaultdict

import numpy as np
import seaborn as sns
import tensorflow as tf
from keras.src.layers import Activation
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.metrics import Precision, Recall
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization

print(tf.__version__)
print(tf.keras.__version__)

tf.config.set_visible_devices([], 'GPU')

dictionnaire_traduction = {
    'cane': 'chien',
    'cavallo': 'cheval',
    'elefante': 'elephant',
    'farfalla': 'papillon',
    'Fursuit': 'fursuit',
    'gallina': 'poule',
    'gatto': 'chat',
    'mucca': 'vache',
    'pecora': 'mouton',
    'ragno': 'araignee',
    'scoiattolo': 'ecureuil'
}


def charge_donnee_detection_fursuit(data_dir, grosseur_image, nom_classe_italien, nombre_images_par_classe=100):
    X, y = [], []
    for class_name in nom_classe_italien:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        img_files = os.listdir(class_path)
        if len(img_files) > nombre_images_par_classe:
            img_files = random.sample(img_files, nombre_images_par_classe)
        for img_file in img_files:
            img_path = os.path.join(class_path, img_file)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(grosseur_image, grosseur_image))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalisation
            X.append(img_array)
            y.append(1 if class_name == 'fursuit' else 0)  # True class if fursuit, else False class
    X = np.array(X)
    y = np.array(y)
    return X, y

def charge_donnee_detection_espece(data_dir, grosseur_image, class_names_it, classes_exclues=None, nombre_images_par_classe=100):
    X, y = np.empty((0, 64, 64, 1)), []
    for class_name in class_names_it:
        if class_name == classes_exclues:
            continue
        class_path = os.path.join(data_dir+"/espece/", class_name)
        if not os.path.isdir(class_path):
            continue
        img_files = os.listdir(class_path)
        if len(img_files) > nombre_images_par_classe:
            img_files = random.sample(img_files, nombre_images_par_classe)
        for img_file in img_files:
            img_path = os.path.join(class_path, img_file)
            img = tf.keras.preprocessing.image.load_img(img_path, color_mode="grayscale", target_size=(grosseur_image, grosseur_image))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalisation
            X = np.append(X, img_array[np.newaxis, ...], axis=0)
            #X.append(img_array)
            y.append(class_name)
    X = np.array(X)
    y = np.array(y)
    return X, y

def prepare_labels_de_fursuit(X, y, fursuit_class_index):
    """
    Prepare les labels pour la classification binaire fursuit vs non-fursuit.
    """
    y_binary = np.where(np.argmax(y, axis=1) == fursuit_class_index, 1, 0)
    return X, y_binary

def obtenir_images_jpeg_par_dossier(racine):
    images_par_dossier = defaultdict(list)
    for sous_dossier, _, fichiers in os.walk(racine):
        for fichier in fichiers:
            if fichier.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                nom_dossier = os.path.basename(sous_dossier)
                images_par_dossier[nom_dossier].append(os.path.join(sous_dossier, fichier))
    return images_par_dossier

def selectionner_images_au_hasard(images_par_dossier, nb_images_par_dossier):
    toutes_images_selectionnees = []
    for dossier, images in images_par_dossier.items():
        if len(images) <= nb_images_par_dossier:
            images_selectionnees = images
        else:
            images_selectionnees = random.sample(images, nb_images_par_dossier)

        for image_selectionnee in images_selectionnees:
            img_test = tf.keras.preprocessing.image.load_img(image_selectionnee, color_mode="grayscale", target_size=(64, 64))
            img_test = tf.keras.preprocessing.image.img_to_array(img_test)
            img_test = img_test / 255.0

            images_test = np.empty((0, 64, 64, 1))
            toutes_images_selectionnees = np.append(images_test, [img_test], axis=0)

    return toutes_images_selectionnees

def predire_image(modele, chemin_image, taille_img, noms_classes, est_modele_fursuit=False):
    img = tf.keras.preprocessing.image.load_img(chemin_image, target_size=(taille_img, taille_img))
    tableau_img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    tableau_img = np.expand_dims(tableau_img, axis=0)
    predictions = modele.predict(tableau_img)

    if est_modele_fursuit:
        probabilite = predictions[0][0]
        idx_classe = 1 if probabilite > 0.5 else 0
        nom_classe = noms_classes[idx_classe]
        return nom_classe, probabilite
    else:
        noms_classes_sans_fursuit = [nom_classe for nom_classe in noms_classes if nom_classe != 'fursuit']
        prob_max = 0
        nom_classe = ""
        for i, label_classe in enumerate(noms_classes_sans_fursuit):
            probabilite = predictions[0][i]
            print(f'Classe : {label_classe}, Probabilite : {probabilite:.4f}')
            if probabilite > prob_max:
                prob_max = probabilite
                nom_classe = label_classe
        return nom_classe, prob_max

def evaluer_modele_avec_matrice_confusion(modele, X_test, y_test, noms_classes, est_modele_fursuit=False):
    if len(y_test.shape) > 1 and not est_modele_fursuit:
        classes_reelles = np.argmax(y_test, axis=1)
    else:
        classes_reelles = y_test

    predictions = modele.predict(X_test)

    if est_modele_fursuit:
        classes_predites = (predictions > 0.5).astype("int32").flatten()
    else:
        classes_predites = np.argmax(predictions, axis=1)

    mc = confusion_matrix(classes_reelles, classes_predites)
    sommes_lignes = np.sum(mc, axis=1, keepdims=True)
    mc_pourcent = (mc / sommes_lignes) * 100

    taux_reussite = np.diag(mc_pourcent)
    taux_reussite_colonne = taux_reussite.reshape(-1, 1)
    mc_avec_colonne_pourcent = np.hstack([mc_pourcent, taux_reussite_colonne])

    taux_reussite_moyen = np.mean(taux_reussite)
    pourcentage_ligne = np.append(taux_reussite, taux_reussite_moyen)
    mc_avec_resume = np.vstack([mc_avec_colonne_pourcent, pourcentage_ligne])

    if est_modele_fursuit:
        noms_classes_avec_total = noms_classes + ['Taux de Reussite']
        noms_colonnes_avec_total = noms_classes + ['Pourcentage']
    else:
        noms_classes_avec_total = noms_classes + ['Taux de Reussite']
        noms_colonnes_avec_total = noms_classes + ['Pourcentage']

    plt.figure(figsize=(12, 10))
    carte_chaleur = sns.heatmap(mc_avec_resume, annot=True, fmt='.2f', cmap='viridis',
                                xticklabels=noms_colonnes_avec_total, yticklabels=noms_classes_avec_total)
    plt.xlabel('Prediction')
    plt.ylabel('Verite')
    plt.title('Matrice de Confusion en Pourcentages avec Taux de Reussite')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

def creer_modele(forme_entree, nb_classes, augmentation_donnees=None, est_modele_fursuit=False):
    modele = Sequential()
    if augmentation_donnees:
        modele.add(augmentation_donnees)

    if est_modele_fursuit:
        modele.add(Conv2D(32, (3, 3), activation='relu', input_shape=forme_entree))
        modele.add(BatchNormalization())
        modele.add(MaxPooling2D((2, 2)))
        modele.add(Dropout(0.25))

        modele.add(Conv2D(64, (3, 3), activation='relu'))
        modele.add(BatchNormalization())
        modele.add(Activation('relu'))
        modele.add(MaxPooling2D((2, 2)))
        modele.add(Dropout(0.25))

        modele.add(Conv2D(128, (3, 3), activation='relu'))
        modele.add(BatchNormalization())
        modele.add(Activation('relu'))
        modele.add(MaxPooling2D((2, 2)))
        modele.add(Dropout(0.25))

        modele.add(Flatten())
        modele.add(Dense(256, activation='relu'))
        modele.add(Dropout(0.5))
        modele.add(Dense(1, activation='sigmoid'))  # Pour la classification binaire
        perte = 'binary_crossentropy'
        metriques = [
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall')]
        modele.compile(optimizer='rmsprop', loss=perte, metrics=metriques)
    else:
        modele.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="relu", input_shape=(64, 64, 1)))
        modele.add(Conv2D(filters=128, kernel_size=(3, 3)))
        modele.add(Flatten())
        modele.add(Dense(128, activation='relu'))
        modele.add(Dense(64, activation='relu'))
        modele.add(Dense(nb_classes, activation='softmax'))
        perte = 'sparse_categorical_crossentropy'
        metriques = [
            'accuracy',
        ]
        modele.compile(optimizer='adam', loss=perte, metrics=metriques)

    modele.summary()
    return modele

def entrainer_et_evaluer_modele(modele, X_train, y_train, X_test, y_test, est_modele_fursuit=False):
    arret_precoce = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduction_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.0001)

    if est_modele_fursuit:
        gen_augmentation = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        gen_augmentation.fit(X_train)
        historique = modele.fit(gen_augmentation.flow(
            X_train, y_train, batch_size=32),
            epochs=100,
            validation_split=0.2,
            validation_data=(X_test, y_test),
            callbacks=[arret_precoce, reduction_lr],
            class_weight={0: 0.5, 1: 1.5}
        )
    else:
        historique = modele.fit(
            X_train, y_train,
            epochs=10,
            validation_split=0.2,
            validation_data=(X_test, y_test),
        )

    metriques = ['accuracy', 'precision', 'recall', 'loss']
    for metrique in metriques:
        if metrique in historique.history:
            plt.plot(historique.history[metrique], label=metrique)
            plt.plot(historique.history['val_' + metrique], label='val_' + metrique)
            plt.xlabel('epochs')
            plt.ylabel('Valeur')
            plt.legend(loc='upper right')
            plt.show()


if __name__ == '__main__':
    data_dir = '../TP2-images'
    num_images_per_folder = 100
    img_size = 64
    dictionnaire_traduction = {
        'cane': 'chien',
        'cavallo': 'cheval',
        'elefante': 'elephant',
        'farfalla': 'papillon',
        'Fursuit': 'fursuit',
        'gallina': 'poule',
        'gatto': 'chat',
        'mucca': 'vache',
        'pecora': 'mouton',
        'ragno': 'araignee',
        'scoiattolo': 'ecureuil'
    }


    class_names_it = dictionnaire_traduction.keys()
    class_names_fr = dictionnaire_traduction.values()

    images_by_folder = obtenir_images_jpeg_par_dossier(data_dir)
    valid_images_by_folder = {folder: [img for img in images if tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(img)) is not None] for folder, images in images_by_folder.items()}
    training_images = selectionner_images_au_hasard(valid_images_by_folder, num_images_per_folder)

    # Preparation des donnees pour la detection de fursuits
    X_fursuit, y_fursuit = charge_donnee_detection_fursuit(data_dir, 64, class_names_it)
    X_train_fursuit, X_test_fursuit, y_train_fursuit, y_test_fursuit = train_test_split(X_fursuit, y_fursuit,
                                                                                        test_size=0.2, random_state=42)

    # Preparation des donnees pour la classification des especes
    class_names_it_no_fursuit = [c for c in class_names_it if c != 'Fursuit']
    X_species, y_species = charge_donnee_detection_espece(data_dir, 64, class_names_it_no_fursuit,
                                                          classes_exclues='Fursuit')
    X_train_species, X_test_species, y_train_species, y_test_species = train_test_split(X_species, y_species,
                                                                                        test_size=0.2, random_state=42)

    # Convertir les labels de classification des especes en one-hot encoding
    num_classes_species = len(class_names_it_no_fursuit)
    y_train_species = tf.keras.utils.to_categorical(
        [class_names_it_no_fursuit.index(label) for label in y_train_species], num_classes=num_classes_species)

    y_test_species = tf.keras.utils.to_categorical([class_names_it_no_fursuit.index(label) for label in y_test_species],
                                                   num_classes=num_classes_species)


    model_fursuit = creer_modele((img_size, img_size, 3), nb_classes=2,
                                 augmentation_donnees=None,#Sequential([layers.RandomRotation(0.1)]),
                                 est_modele_fursuit=True)
    history_fursuit =  entrainer_et_evaluer_modele(model_fursuit, X_train_fursuit, y_train_fursuit, X_test_fursuit,
                                               y_test_fursuit, est_modele_fursuit=True)
    evaluer_modele_avec_matrice_confusion(model_fursuit, X_test_fursuit, y_test_fursuit, ['Non-fursuit', 'Fursuit'], True)
    y_pred_fursuit = (model_fursuit.predict(X_test_fursuit) > 0.3).astype("int32")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_fursuit, y_pred_fursuit))

    print("\nClassification Report:")
    print(classification_report(y_test_fursuit, y_pred_fursuit))



    classnames_fr_no_fursuit = [c for c in class_names_fr if c != 'fursuit']

    model_species = creer_modele((img_size, img_size, 1), nb_classes=len(class_names_it_no_fursuit),
                                 augmentation_donnees=None)#Sequential([layers.RandomRotation(0.1)]))
    entrainer_et_evaluer_modele(model_species, X_train_species, y_train_species, X_test_species, y_test_species)

    evaluer_modele_avec_matrice_confusion(model_species, X_test_species, y_test_species, classnames_fr_no_fursuit,est_modele_fursuit=False)

    test_images = selectionner_images_au_hasard(images_by_folder, nb_images_par_dossier=10)
    bonneReponse = 0
    for file in test_images:
        result1, prob1 = predire_image(model_fursuit, file, 64, ['non-fursuit','fursuit'], True)
        result2, prob2 = predire_image(model_species, file, 64, class_names_fr)
        is_fursuit = 'oui' if result1 == 'fursuit' else 'non'
        espece = result2 if result1 != 'fursuit' else 'aucune'

        print(f"fursuit : {is_fursuit}")
        print(f"Nom image : {file}")
        print(f"Prediction: {dictionnaire_traduction[file.replace('\\', '/').split('/')[2]]}")
        if dictionnaire_traduction[file.replace('\\', '/').split('/')[2]] == espece:
            bonneReponse += 1

        print(f"Espece : {espece}")
    print(f"RÃ©sultat: {bonneReponse}/{len(test_images)}, {round((bonneReponse / len(test_images)) * 100, 2)}%")

