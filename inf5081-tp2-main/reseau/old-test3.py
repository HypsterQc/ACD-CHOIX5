import os
import random
from collections import defaultdict

import numpy as np
import tensorflow as tf
from keras.src.metrics import Precision, Recall
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, layers


def creer_modele_cnn_1(forme_entree, nb_classes):
    modele = Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=forme_entree),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(nb_classes, activation='softmax')
    ])
    metriques = [
        'accuracy',
        Precision(name='precision'),
        Recall(name='recall')]

    modele.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metriques)
    return modele

def creer_modele_cnn_2(forme_entree, nb_classes):
    modele = Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=forme_entree),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(nb_classes, activation='softmax')
    ])
    metriques = [
        'accuracy',
        Precision(name='precision'),
        Recall(name='recall')]

    modele.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metriques)
    return modele

def ensemble_predict(modeles, X_test):
    predictions = np.zeros((X_test.shape[0], 11))
    for modele in modeles:
        predictions += modele.predict(X_test)
    return np.argmax(predictions, axis=1)

def entrainer_et_evaluer_modele(resultat_entrainement, metrique, etiquette_metrique):
    perf_entrainement = resultat_entrainement.history[metrique]
    perf_validation = resultat_entrainement.history[f'val_{metrique}']

    print("Performance Entrainement: ", perf_entrainement)
    print("Performance Validation: ", perf_validation)

    atol = 1e-2
    idx_intersection = np.argwhere(np.isclose(perf_entrainement, perf_validation, atol=atol)).flatten()

    if idx_intersection.size > 0:
        idx_intersection = idx_intersection[0]
        valeur_intersection = perf_entrainement[idx_intersection]
        print(f'Intersection trouvee a l\'epoch {idx_intersection} avec une valeur de {valeur_intersection:.4f}')
    else:
        print('Aucune intersection trouvee dans la tolerance donnee.')

    plt.plot(perf_entrainement, label=metrique)
    plt.plot(perf_validation, label=f'val_{metrique}')

    if idx_intersection.size > 0:
        plt.axvline(x=idx_intersection, color='r', linestyle='--', label='Intersection')
        plt.annotate(f'Valeur Optimale: {valeur_intersection:.4f}',
                     xy=(idx_intersection, valeur_intersection),
                     xycoords='data',
                     fontsize=10,
                     color='green')

    plt.xlabel('Epoch')
    plt.ylabel(etiquette_metrique)
    plt.legend(loc='lower right')
    plt.show()

def entrainer_modeles(X_train, y_train, X_val, y_val):
    modeles = []

    modele_cnn_1 = creer_modele_cnn_1((64, 64, 3), 11)
    history1 = modele_cnn_1.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), validation_split=0.2, verbose=0)
    modeles.append(modele_cnn_1)

    entrainer_et_evaluer_modele(history1, 'accuracy', 'accuracy')
    entrainer_et_evaluer_modele(history1, 'precision', 'precision')

    modele_cnn_2 = creer_modele_cnn_2((64, 64, 3), 11)
    history2 = modele_cnn_2.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), validation_split=0.2, verbose=0)
    modeles.append(modele_cnn_2)

    entrainer_et_evaluer_modele(history2, 'accuracy', 'accuracy')
    entrainer_et_evaluer_modele(history2, 'precision', 'precision')

    return modeles

def evaluer_modele_avec_matrice_confusion(modele, X_test, y_test, noms_classes):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    classes_reelles = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
    predictions = modele.predict(X_test)
    classes_predites = np.argmax(predictions, axis=1)

    cm = confusion_matrix(classes_reelles, classes_predites)
    sommes_lignes = np.sum(cm, axis=1, keepdims=True)
    cm_pourcent = (cm / sommes_lignes) * 100

    pourcent_succes = np.diag(cm_pourcent)
    col_pourcent_succes = pourcent_succes.reshape(-1, 1)
    cm_avec_col_pourcent = np.hstack([cm_pourcent, col_pourcent_succes])

    moyenne_pourcent_succes = np.mean(pourcent_succes)
    ligne_pourcent = np.append(pourcent_succes, moyenne_pourcent_succes)

    cm_avec_resume = np.vstack([cm_avec_col_pourcent, ligne_pourcent])

    noms_classes_avec_total = noms_classes + ['Taux de Reussite']
    noms_colonnes_avec_total = noms_classes + ['Pourcentage']

    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(cm_avec_resume, annot=True, fmt='.2f', cmap='viridis',
                          xticklabels=noms_colonnes_avec_total, yticklabels=noms_classes_avec_total)

    plt.xlabel('Prediction')
    plt.ylabel('Verite')
    plt.title('Matrice de Confusion en Pourcentages avec Taux de Reussite')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

def obtenir_toutes_images_jpeg_par_dossier(rep_racine):
    images_par_dossier = defaultdict(list)
    for sousdir, _, fichiers in os.walk(rep_racine):
        for fichier in fichiers:
            if fichier.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                nom_dossier = os.path.basename(sousdir)
                images_par_dossier[nom_dossier].append(os.path.join(sousdir, fichier))
    return images_par_dossier

def selectionner_images_aleatoires_egales(images_par_dossier, nb_images_par_dossier):
    toutes_images_selectionnees = []
    for dossier, images in images_par_dossier.items():
        if len(images) <= nb_images_par_dossier:
            images_selectionnees = images
        else:
            images_selectionnees = random.sample(images, nb_images_par_dossier)
        toutes_images_selectionnees.extend(images_selectionnees)
    return toutes_images_selectionnees


def charger_donnee(images_selectionnee, grosseur_image, nom_classes_italien, nom_classes_fr):
    X, y = [], []
    nbItems = 0
    for image_chemin in images_selectionnee:
        nom_classe_italien = os.path.basename(os.path.dirname(image_chemin))
        classe_index = nom_classes_italien.index(nom_classe_italien)
        img = tf.keras.preprocessing.image.load_img(image_chemin, target_size=(grosseur_image, grosseur_image))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalisation
        X.append(img_array)
        y.append(classe_index)
        nbItems += 1

    print("Nombre d'images: " + str(nbItems))
    X = np.array(X)
    y = tf.keras.utils.to_categorical(np.array(y), num_classes=len(nom_classes_fr))
    return X, y


if __name__ == '__main__':
    data_dir = '../TP2-images'
    num_images_per_folder = 1000

    class_names_it = sorted(os.listdir(data_dir))
    translation_dict = {
        'cane': 'chien', 'cavallo': 'cheval', 'elefante': 'elephant', 'farfalla': 'papillon',
        'fursuit': 'fursuit', 'gallina': 'poule', 'gatto': 'chat', 'mucca': 'vache',
        'pecora': 'mouton', 'ragno': 'araignee', 'scoiattolo': 'ecureuil'
    }
    class_names_fr = [translation_dict[name].lower().replace('é', 'e').replace(' ', '_') for name in class_names_it]

    images_by_folder = obtenir_toutes_images_jpeg_par_dossier(data_dir)
    training_images = selectionner_images_aleatoires_egales(images_by_folder, num_images_per_folder)
    X, y = charger_donnee(training_images, grosseur_image=64, nom_classes_italien=class_names_it, nom_classes_fr=class_names_fr)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = entrainer_modeles(X_train, y_train, X_test, y_test)
    y_pred = ensemble_predict(models, X_test)
    y_true = np.argmax(y_test, axis=1)

    test_images = selectionner_images_aleatoires_egales(images_by_folder, nb_images_par_dossier=10)
    evaluer_modele_avec_matrice_confusion(models[0], X_test, y_test, class_names_fr)