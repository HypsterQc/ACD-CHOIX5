import os
import random
from collections import defaultdict

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

translation_dict = {
    'cane': 'chien',
    'cavallo': 'cheval',
    'elefante': 'elephant',
    'farfalla': 'papillon',
    'fursuit': 'fursuit',
    'gallina': 'poule',
    'gatto': 'chat',
    'mucca': 'vache',
    'pecora': 'mouton',
    'ragno': 'araignee',
    'scoiattolo': 'ecureuil'
}


def charger_donnees(images_selectionnees, taille_image, noms_classes_it, noms_classes_fr):
    X, y = [], []
    nbItems = 0
    for chemin_img in images_selectionnees:
        nom_classe_it = os.path.basename(os.path.dirname(chemin_img))
        index_classe = noms_classes_it.index(nom_classe_it)
        img = tf.keras.preprocessing.image.load_img(chemin_img, target_size=(taille_image, taille_image))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalisation
        X.append(img_array)
        y.append(index_classe)
        nbItems += 1

    print("Nombre d'images: " + str(nbItems))
    X = np.array(X)
    y = tf.keras.utils.to_categorical(np.array(y), num_classes=len(noms_classes_fr))
    return X, y

def creer_modele(taille_entree, nb_classes):
    modele = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=taille_entree),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(nb_classes, activation='softmax'),
        ]
    )
    modele.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return modele

def entrainer_et_evaluer_modele(modele, X_train, y_train, X_test, y_test, nb_epochs, taille_lot):
    modele.fit(X_train, y_train, epochs=nb_epochs, batch_size=taille_lot, validation_split=0.2)
    perte, precision = modele.evaluate(X_test, y_test)
    print(f"Précision sur les tests: {precision:.2f}")
    return modele

def predire_image(modele, chemin_image, taille_image, noms_classes):
    img = tf.keras.preprocessing.image.load_img(chemin_image, target_size=(taille_image, taille_image))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = modele.predict(img_array)
    classe_predite = np.argmax(predictions, axis=1)[0]
    return noms_classes[classe_predite]

def obtenir_toutes_images_jpeg_par_dossier(repertoire_racine):
    images_par_dossier = defaultdict(list)
    for sous_repertoire, _, fichiers in os.walk(repertoire_racine):
        for fichier in fichiers:
            if fichier.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                nom_dossier = os.path.basename(sous_repertoire)
                images_par_dossier[nom_dossier].append(os.path.join(sous_repertoire, fichier))
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


if __name__ == '__main__':
    data_dir = '../TP2-images'
    num_images_per_folder = 100

    class_names_it = sorted(os.listdir(data_dir))
    class_names_fr = [translation_dict[name].lower().replace('é', 'e').replace(' ', '_') for name in class_names_it]

    images_by_folder = obtenir_toutes_images_jpeg_par_dossier(data_dir)

    training_images = selectionner_images_aleatoires_egales(images_by_folder, num_images_per_folder)

    X, y = charger_donnees(training_images, taille_image=64, noms_classes_it=class_names_it, noms_classes_fr=class_names_fr)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = creer_modele((64, 64, 3), len(class_names_fr))
    model = entrainer_et_evaluer_modele(model, X_train, y_train, X_test, y_test, nb_epochs=5, taille_lot=32)

    model.summary()

    test_images = selectionner_images_aleatoires_egales(images_by_folder, nb_images_par_dossier=10)

    bonneReponse = 0
    for file in test_images:
        result =  predire_image(model, file, 64, class_names_fr)
        is_fursuit = 'oui' if result == 'fursuit' else 'non'
        espece = result if result != 'fursuit' else 'aucune'

        print(f"fursuit : {is_fursuit}")
        print(f"Nom image : {file}")
        print(f"Prediction: {translation_dict[file.replace('\\', '/').split('/')[1]]}")
        if translation_dict[file.replace('\\', '/').split('/')[1]] == espece:
            bonneReponse += 1

        print(f"Espece : {espece}")
    print(f"Résultat: {bonneReponse}/{len(test_images)}, {round((bonneReponse/len(test_images))*100, 2)}%")