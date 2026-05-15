# Airbnb Price Prediction

Projet de machine learning pour predire le prix de nuit d'un logement Airbnb a partir de ses caracteristiques. La cible du modele est `log_price`, c'est-a-dire le logarithme du prix.

## Objectif

Construire un pipeline complet capable de transformer des donnees Airbnb brutes en predictions exploitables pour un fichier de soumission.

Le projet couvre :

- exploration des donnees et visualisations ;
- nettoyage des valeurs manquantes ;
- feature engineering sur les dates, textes, equipements et coordonnees ;
- encodage des variables categorielles sans fuite de donnees ;
- comparaison de plusieurs modeles ;
- optimisation LightGBM avec Optuna ;
- validation croisee et export final des predictions.

## Structure du depot

```text
.
|-- airbnb_prediction.ipynb   # Notebook principal : EDA, features, modeles, export
|-- airbnb_train.csv          # Donnees d'entrainement avec log_price
|-- airbnb_test.csv           # Donnees de test a predire
|-- prediction_example.csv    # Format attendu pour la soumission
|-- MaPredictionFinale.csv    # Predictions finales generees par le notebook
|-- requirements.txt          # Dependances Python
`-- .gitignore
```

## Donnees

Le jeu d'entrainement contient environ 22 000 annonces Airbnb avec le prix connu. Le jeu de test contient environ 52 000 annonces pour lesquelles le notebook genere une prediction.

Variables principales utilisees :

- informations du logement : `property_type`, `room_type`, `accommodates`, `bathrooms`, `bedrooms`, `beds` ;
- localisation : `city`, `neighbourhood`, `zipcode`, `latitude`, `longitude` ;
- hote et avis : `host_since`, `host_response_rate`, `number_of_reviews`, `review_scores_rating` ;
- texte : `name`, `description`, `amenities` ;
- cible : `log_price`.

## Methodologie

Le pipeline applique plusieurs transformations avant l'entrainement :

- extraction des 50 amenities les plus frequentes ;
- conversion des dates en anciennete et ajout de flags de valeurs manquantes ;
- creation de ratios comme lits par personne ou salles de bain par chambre ;
- calcul d'une distance au centre median de chaque ville ;
- extraction de features texte avec TF-IDF puis reduction SVD ;
- target encoding K-fold pour les variables a forte cardinalite ;
- frequency encoding pour capturer la popularite de certains quartiers ou codes postaux ;
- label encoding pour les categories plus simples.

## Modeles testes

| Modele / configuration | RMSE CV |
| --- | ---: |
| Baseline moyenne | 0.7188 |
| Ridge sur 8 features brutes | 0.5755 |
| Taille + localisation | 0.4373 |
| Categories encodees | 0.4048 |
| Amenities ajoutees | 0.3940 |
| Gradient Boosting | 0.3918 |
| LightGBM toutes features | 0.3881 |
| LightGBM + Optuna + 5-fold | 0.3789 |

Le meilleur score observe dans le notebook est un RMSE OOF d'environ `0.3789` sur `log_price`.

## Installation

Creer un environnement virtuel :

```bash
python -m venv .venv
```

Activer l'environnement :

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Installer les dependances :

```bash
pip install -r requirements.txt
```

Lancer Jupyter :

```bash
jupyter lab
```

Puis ouvrir `airbnb_prediction.ipynb` et executer les cellules dans l'ordre.

## Export

Le notebook genere le fichier :

```text
MaPredictionFinale.csv
```

Le fichier respecte le format de `prediction_example.csv` et contient les predictions de `log_price` pour le jeu de test.

## Pistes d'amelioration

- Ajouter des donnees externes : transports, attractivite du quartier, prix immobilier local.
- Tester des embeddings de phrase pour mieux representer les descriptions.
- Entrainer des modeles specialises par grande ville.
- Utiliser une regression quantile pour produire des intervalles de prix.
- Gerer les gros fichiers de donnees avec Git LFS si le depot devient trop lourd.
