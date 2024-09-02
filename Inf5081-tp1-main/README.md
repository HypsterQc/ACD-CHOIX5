_# INF5081-TP1 - Détermination de la Popularité d'un Profil sur un Site de Rencontre

Ce projet a pour but de déterminer si un profil sur un site de rencontre est populaire en utilisant des techniques de classification. <br>
Pour qu'un compte soit considéré populaire, il doit remplir deux critères : avoir plus de 10 visites et un taux de personnes aimant le compte après visite supérieur à 5 %. 

Nous allons mener une étude en utilisant des classificateurs vue au cours (voir [Documentations](#documentations)).

Structure fichier:
Il n'est pas obligatoire juste plus facile a comprendre lorsqu'il y a une structuration des fichiers.
- Data : la ou ce trouve la Data-warehouse et les data-marts.
- Graphe_Link : utiliser afin de garder des screenshoot des resultats des graphes et sources de cours.

Pour Repondre a la question sur l'étude en utilisant de la classification pour voir si un compte est populaire sans lemplois des critere de visites et likes et si oui comment. 
Nous avons constater qu'il y a plusieur impact minime visible a travers la [Matrisse de correlation generaliser](Graphe_Link/Dw.png). on ses appercue de l'impact que peut avoir les diverse chois de langue.
- [MartsDe.png](Graphe_Link/MartsDe.png)
- [MartsEn.png](Graphe_Link/MartsEn.png)
- [MartsEs.png](Graphe_Link/MartsEs.png)
- [MartsFr.png](Graphe_Link/MartsFr.png)
- [MartsIt.png](Graphe_Link/MartsIt.png)
- [MartsPt.png](Graphe_Link/MartsPt.png)
Pour les autres Data-Marts pair, impair et randoms on plus une pertinance sur la minimalisation des donnees afin deviter l'overfitting.
Nous nous sommes aussi servie des [attributs initiaux](Graphe_Link/attributsInitiaux.png)
## Table des Matières

1. [Introduction](#introduction)
2. [Data-Warehouse](#data-warehouse)
   1. [data-base 01 (API-Results.csv)](#data-base-01-api-resultscsv--)
   2. [data-base 02 (Instances.csv)](#data-base-02-instancescsv--)
   3. [data-base 03 (V3.db)](#data-base-03-v3db--)
3. [Normalisation](#normalisation)
   1. [Resultat Final](#resultat-final-)
   2. [Listes des refus](#listes-des-refus-)
4. [Data-marts](#data-marts)
5. [Algorithmes](#algorithmes)
   1. [Arbres de décision](#arbres-de-décision)
   2. [K-NN](#k-nn)
   3. [Classificateurs bayésiens](#classificateurs-bayésiens)
6. [Documentations](#documentations)
7. [Dépendances](#dépendances)
8. [Auteurs](#auteurs)

## Introduction

Afin de bien debuter le programme verifier dabord qu vous avez bien installer la liste de toute les dependances avant l'execution du programme.<br>
Pour plus de détails, consultez le fichier PDF : [TP1-ete2024.pdf](./TP1-ete2024.pdf)

## Data-warehouse

La comprehension et signification de chaque element de la DW.

### Data-base 01 (API-Results.csv) : 

| **Sélectionner** | **Variable**           | **Description**                                                                                 | **Option**                                                 |
|------------------|------------------------|-------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| [ ]              | gender                 | Le genre/sexe de l'utilisateur                                                                  | M = masculin, F = féminin                                  |
| [x]              | genderLooking          | Le genre/sexe que recherche l'utilisateur                                                       | M = masculin, F = féminin, none = Asexuel, both = Bisexuel |
| [x]              | age                    | L'âge                                                                                           | integer > 0 ~ min = 18                                     |
| [ ]              | name                   | Nom/Pseudonyme                                                                                  | string                                                     |
| [x]              | counts_details         | Probablement un score associé soit à quel pourcentage as-tu rempli le profil ou score de beauté | float entre 0 & 1                                          |
| [x]              | counts_pictures        | Nombre de photos de l'utilisateur                                                               | integer >= 0                                               |
| [ ]              | counts_profileVisits   | Nombre de visites de profil que l'utilisateur a reçues                                          | integer >= 0                                               |
| [ ]              | counts_kisses          | Nombre de "likes" reçus                                                                         | integer >= 0                                               |
| [x]              | counts_fans            | Nombre "d'abonnés" de l'utilisateur                                                             | integer >= 0                                               |
| [x]              | counts_g               | Nombre de ?                                                                                     | integer >= 0                                               |
| [x]              | flirtInterests_chat    | Intéressé à discuter                                                                            | true/false                                                 |
| [x]              | flirtInterests_friends | Intéressé à se faire des amis                                                                   | true/false                                                 |
| [x]              | flirtInterests_date    | Intéressé à trouver l'amour                                                                     | true/false                                                 |
| [ ]              | country                | Pays de l'utilisateur                                                                           | string                                                     |
| [ ]              | city                   | Ville de l'utilisateur                                                                          | string                                                     |
| [ ]              | location               | Localisation actuelle                                                                           | string                                                     |
| [x]              | distance               | La distance à partir d'un point de référence                                                    | km, miles, ?                                               |
| [x]              | isFlirtstar            | Indicateur si l'utilisateur est une "star de flirt"                                             | true/false                                                 |
| [x]              | isHighlighted          | Indicateur si le profil de l'utilisateur est mis en avant                                       | true/false                                                 |
| [ ]              | isInfluencer           | Indicateur si l'utilisateur est un influenceur                                                  | true/false                                                 |
| [x]              | isMobile               | Indicateur si l'utilisateur utilise une application mobile                                      | true/false                                                 |
| [x]              | isNew                  | Indicateur si l'utilisateur est nouveau                                                         | true/false                                                 |
| [x]              | isOnline               | Indicateur si l'utilisateur est actuellement en ligne                                           | true/false                                                 |
| [x]              | isVip                  | Indicateur si l'utilisateur a un statut VIP/paye pour un abonnement                             | true/false                                                 |
| [x]              | lang_count             | Nombre de langues parlées                                                                       | integer                                                    |
| [x]              | lang_fr                | Parle français                                                                                  | 0 ou 1                                                     |
| [x]              | lang_en                | Parle anglais                                                                                   | 0 ou 1                                                     |
| [x]              | lang_de                | Parle allemand                                                                                  | 0 ou 1                                                     |
| [x]              | lang_it                | Parle italien                                                                                   | 0 ou 1                                                     |
| [x]              | lang_es                | Parle espagnol                                                                                  | 0 ou 1                                                     |
| [x]              | lang_pt                | Parle portugais                                                                                 | 0 ou 1                                                     |
| [x]              | verified               | Profil vérifié                                                                                  | true/false                                                 |
| [x]              | shareProfileEnabled    | Indicateur si le partage du profil est activé                                                   | true/false                                                 |
| [ ]              | lastOnlineDate         | La dernière date en ligne                                                                       | Date                                                       |
| [x]              | lastOnlineTime         | La dernière heure en ligne                                                                      | (timestamp)                                                |
| [ ]              | birthd                 | Date de naissance                                                                               | toujours à '0'                                             |
| [ ]              | crypt                  | Probablement une valeur cryptée                                                                 | empty                                                      |
| [ ]              | freetext               | Biographie de l'utilisateur                                                                     | string                                                     |
| [x]              | whazzup                | Statut de l'utilisateur                                                                         | string                                                     |
| [x]              | userId                 | L'identifiant unique de l'utilisateur                                                           | clé primaire                                               |
| [x]              | pictureId              | L'identifiant unique de la photo de profil de l'utilisateur                                     | clé secondaire                                             |
| [ ]              | isSystemProfile        | s'il est un profil système                                                                      | empty                                                      |

### Data-base 02 (Instances.csv) : 

| **Sélectionner** | **Variable**           | **Description**                                                    | **Option**                |
|------------------|------------------------|--------------------------------------------------------------------|---------------------------|
| [ ]              | gender                 | Le genre/sexe de l'utilisateur                                     | F = féminin, M = masculin |
| [x]              | age                    | L'âge de l'utilisateur                                             | integer > 0 ~ min = 18    |
| [ ]              | name                   | Pseudonyme                                                         | string                    |
| [x]              | counts_pictures        | Nombre de photos de profil                                         | integer                   |
| [x]              | counts_profileVisits   | Nombre de visites de profil que l'utilisateur a reçues             | integer                   |
| [x]              | counts_kisses          | Le nombre de "Likes" que l'utilisateur a reçus                     | integer                   |
| [x]              | flirtInterests_chat    | Intéressé à discuter                                               | true/false                |
| [x]              | flirtInterests_friends | Intéressé à faire des amis                                         | true/false                |
| [x]              | flirtInterests_date    | Intéressé à l'amour                                                | true/false                |
| [ ]              | connectedToFacebook    | Connecté à Facebook                                                | true/false                |
| [x]              | isVIP                  | Statut VIP/Abonnement payant                                       | true/false                |
| [x]              | isVerified             | Profil vérifié                                                     | true/false                |
| [ ]              | lastOnline             | La dernière date en ligne                                          | Date                      |
| [x]              | lastOnlineTs           | La dernière heure en ligne                                         | (timestamp)               |
| [ ]              | lang_count             | Nombre de langues parlées                                          | integer >= 0              |
| [x]              | lang_fr                | Parle français                                                     | 0 ou 1                    |
| [x]              | lang_en                | Parle anglais                                                      | 0 ou 1                    |
| [x]              | lang_de                | Parle allemand                                                     | 0 ou 1                    |
| [x]              | lang_it                | Parle italien                                                      | 0 ou 1                    |
| [x]              | lang_es                | Parle espagnol                                                     | 0 ou 1                    |
| [x]              | lang_pt                | Parle portugais                                                    | 0 ou 1                    |
| [ ]              | city                   | Ville de l'utilisateur                                             | string                    |
| [ ]              | locationCity           | Localisation spécifique de l'utilisateur                           | string                    |
| [ ]              | locationCitySub        | Sous-localisation de l'utilisateur                                 | string                    |
| [ ]              | userInfo_visitDate     | La date de la dernière visite d'un autre utilisateur sur le profil |                           |
| [x]              | countDetails           | Un score lié aux détails                                           | Float entre 0 & 1         |
| [ ]              | crypt                  | Une valeur cryptée                                                 | true/false                |
| [x]              | flirtstar              | Indicateur si l'utilisateur est une "star de flirt"                | true/false                |
| [x]              | freshman               | Nouvel utilisateur                                                 | true/false                |
| [ ]              | hasBirthday            | Indicateur si l'utilisateur a donné sa date de naissance           | true/false                |
| [x]              | highlighted            | Indicateur si le profil de l'utilisateur est mis en avant          | true/false                |
| [ ]              | distance               | Distance à partir d'un point de référence                          | km, miles, ?              |
| [ ]              | locked                 | Indicateur si le profil est verrouillé/non-visible                 | true/false                |
| [ ]              | mobile                 | Utilise une application mobile                                     | true/false                |
| [ ]              | online                 | Actuellement en ligne                                              | true/false                |
| [x]              | whazzup                | Statut de l'utilisateur                                            | string                    |
| [ ]              | userId                 | L'identifiant unique de l'utilisateur                              | clé primaire              |
| [ ]              | pictureId              | L'identifiant unique de la photo de profil de l'utilisateur        | clé secondaire            |
| [ ]              | isSystemProfile        | Indicateur si le profil est un profil système                      | true/false                |


### Data-base 03 (V3.db) : 

Pareil a [data-base 01 (API-Results.csv)](#data-base-01-api-resultscsv--)

## Normalisation

Avis de Normalisation :
Suivis des etapes de normalisation et de netoyage de la Data
- Integriter : soit mettre tous les valeurs true / false, '0' / '1' en valeur numeric 0 / 1
- on a rendu compatible le fichier .db et .csv
- 'freetext' : a ete modifier de 0 pour les contenu vide et 1 si les case contenait du texte.
- toutes les colonne de donnee contenant des nombres dans un type string a ete convertie au type numeric
- 'genderLooking' : les valeur on ete modifier soit 'M': 1, 'F': 2, 'both': 3, 'none': 0.
- Les lignes avec 'isSystemProfile' = 'true' ont été supprimées
- Les lignes avec des valeurs NaN ou vides dans la colonne 'distance' ont été supprimées.
- Les lignes avec des valeurs NaN ou vides dans la colonne 'lastOnlineTs' ont été supprimées.
- Conserver lastOnlineTs car normaliser 
- Nous avons conserver chaque colonne qui est une action de correlation minime sois telle afin de garder le plus possible de colonne.

### Listes des refus :

| **Variable**                                               | **Explication**                                                                                                       |
|------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| isSystemProfile                                            | retrait des lignes complete ainsi que de la colonne car il sagis de profil systeme et non de compte utilisateur reel. |
| pictureId / userId                                         | sert uniquement d'indexation (query) pour une bd rien d'autre.                                                        |
| crypt                                                      | inutile on supose qu'il s'agis dun encryptage pour le mot de passe de lutilisateur ou autre.                          |
| city / locationCity / location / locationCitySub / country | trop de choix, trop precis, trop long a generaliser.                                                                  |
| name                                                       | aucune correlation avec la populariter.                                                                               |
| gender / isInfluencer                                      | tous des femmes (homogeneiter).                                                                                       |
| highlighted                                                | Effacer car similaire a flirstars                                                                                     |
| lastOnline / lastOnlineDate / lastOnlineTime               | pas normaliser et aucune correlation avec la populariter.                                                             |
| counts_profileVisits / counts_kisses                       | car il sagis des variable de la valeur de la class Y                                                                  | |

## Data-marts

Realisation de 3 type de data-marts, soit:
- Prend les lignes **Pairs** (datasetPair). [MartPairs.png](Graphe_Link/MartPairs.png)
- Prend les lignes **Imnpairs** (datasetImpair). [MartImpair.png](Graphe_Link/MartImpair.png)
- Prend 1/10 **Aleatoire** de la base de donnee (datasetRandom). [MartRandom.png](Graphe_Link/MartRandom.png)
- Prend les lignes de chaque **langue** individuel de la base de donnee.

## Algorithmes

Voici la liste des algorithme employer avec une explication, des resultats.

### Arbres de décision

Résultats : [ArbreDec2.png](Graphe_Link/ArbreDec2.png)

Explication : Come l'arbre de décision n'est pas assez rapide comparativement a K-NN et bayésiens on se dirigerais plus vers ces classificateur.

### K-NN

Résultats :  
- [knn1.png](Graphe_Link/knn1.png)
- [knn2.png](Graphe_Link/knn2.png)
- [knn3.png](Graphe_Link/knn3.png)
- [knn4.png](Graphe_Link/knn4.png)
- [knn5.png](Graphe_Link/knn5.png)
- [knn6.png](Graphe_Link/knn6.png)
- [knn7.png](Graphe_Link/knn7.png)
- [knn8.png](Graphe_Link/knn8.png)
- [knn9.png](Graphe_Link/knn9.png)

Explication : 
- Pour k=1, le modèle est très bien ajusté sur les données d'entraînement (overfitting), mais la précision sur les données de test est faible, indiquant une faible généralisation.
- Pour k=2, le modèle généralise mieux, car la précision du test est plus élevée que pour k=1k=1, mais la précision de l'entraînement a diminué, ce qui est souvent un bon signe d'amélioration de la généralisation.
- Pour k=3, la précision du modèle sur les données de test diminue légèrement par rapport à k=2k=2. Cependant, il y a un bon équilibre entre les True Positives et les True Negatives.
- Pour k=4, la précision du modèle sur les données de test reste la même que pour k=2k=2, mais la précision de l'entraînement continue de diminuer, ce qui indique un bon ajustement.
- Pour k=5, la précision du modèle sur les données de test est inférieure à celle de k=4k=4. Le modèle semble être moins performant pour ce choix de kk.
- Pour k=6, la précision sur les données de test augmente légèrement par rapport à k=5k=5, indiquant une amélioration.
- Pour k=7, la précision du modèle sur les données de test reste stable, mais il semble y avoir une légère détérioration par rapport à k=6k=6.
- Pour k=8, le modèle montre la meilleure précision sur les données de test parmi tous les kk testés jusqu'à présent, indiquant un bon choix pour la valeur de kk.
- Pour k=9, la précision sur les données de test diminue légèrement par rapport à k=8k=8, indiquant une performance légèrement inférieure.
la valeur optimale de kk semble se situer autour de 6 à 8 pour cet ensemble de données, offrant un bon compromis entre l'ajustement des données d'entraînement et la généralisation aux données de test.

### Classificateurs bayésiens

Résultats :  [Baye.png](Graphe_Link/Baye.png)

## Documentations

* [K-NN](Graphe_Link/5_-_Réseaux_Bayésien_et_K_plus_proches_voisins.pdf)
* [Classificateurs bayésiens Naif](Graphe_Link/5_-_Réseaux_Bayésien_et_K_plus_proches_voisins.pdf)
* [Random Forest](Graphe_Link/4_-_Arbres_de_décision-1.pdf)
* [Arbres de décision](Graphe_Link/4_-_Arbres_de_décision-1.pdf)

## Dépendances

* [Pandas](https://pandas.pydata.org/pandas-docs/stable/)
* [NumPy](https://numpy.org/doc/stable/)
* [Matplotlib](https://matplotlib.org/stable/contents.html)
* [re](https://docs.python.org/3/library/re.html)
* [Seaborn](https://seaborn.pydata.org/)
* [IPython](https://ipython.readthedocs.io/en/stable/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [io](https://docs.python.org/3/library/io.html)
* [graphviz](https://graphviz.org/documentation/)
* [pydotplus](https://pypi.org/project/pydotplus/)
* [Python 3](https://docs.python.org/3/)
* [Jupyter Notebook](https://jupyter.org/documentation)
* [datetime](https://docs.python.org/3/library/datetime.html)
* [csv](https://docs.python.org/3/library/csv.html)

## Auteurs

© 2024 Tous droits réservés;

Nom : Kevin Da Silva
CodeMS : DASK30049905

Nom : Samuel Dextraze
CodeMS : DEXS03039604

Nom : Saïd Ryan Joseph
CodeMS : JOSS92030104