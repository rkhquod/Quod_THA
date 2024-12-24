# Prévision et Analyse des Ventes

Ce projet propose un mini-workflow de data science pour analyser et prédire le comportement des clients et la dynamique des produits.  
Il inclut :
- Le chargement et le nettoyage des données,
- Quelques visualisations exploratoires,
- L’entraînement **d’au moins deux modèles** (ici, **Régression Linéaire** et **Random Forest Regressor**),
- La comparaison de leurs performances et la sauvegarde des modèles entraînés.

## Structure du Projet

 ├── data_preprocessing.py # Chargement et nettoyage des données
 ├── visualizations.py # Fonctions de visualisation (exploration) 
 ├── model_train.py # Préparation des données, entraînement et comparaison de plusieurs modèles 
 ├── model_evaluate.py # Chargement et évaluation d’un modèle sauvegardé 
 ├── main.py # Point d'entrée principal qui orchestre le tout 
 ├── transactions_1.csv et transactions_2.csv # Exemple de fichier CSV contenant les transactions 
 └── README.md # Ce fichier de documentation

### Rôles des Fichiers

1. **`data_preprocessing.py`**  
   - Contient les fonctions pour charger le dataset (fichier CSV) et le nettoyer :  
     - Conversion de la date au format datetime,  
     - Suppression des lignes ou dates invalides,  
     - Gestion de la duplication, etc.

2. **`visualizations.py`**  
   - Définit plusieurs fonctions de tracés (avec `matplotlib`) pour :  
     - Afficher le nombre de transactions par client,  
     - Analyser la fréquence des transactions par mois pour un produit donné,  
     - Voir les top 5 produits sur les 6 derniers mois,  
     - Identifier une éventuelle saisonnalité mensuelle.

3. **`model_train.py`**  
   - Prépare les données pour la modélisation (groupement mensuel par client),  
   - Entraîne **plusieurs modèles** (ex. Régression Linéaire, Random Forest),  
   - Compare leurs métriques (MAE, RMSE) sur le même jeu de test,  
   - Permet de **sauvegarder** chaque modèle entraîné dans un fichier `.pkl`.

4. **`model_evaluate.py`**  
   - Charge l’un des modèles sauvegardés,  
   - Évalue ses performances globales (MAE, RMSE),  
   - Affiche éventuellement un scatter plot « valeurs réelles vs prédictions ».

5. **`main.py`**  
   - Sert de script principal. Il :  
     1. Charge et nettoie les données à partir de `transactions_x.csv`,  
     2. Lance les visualisations clés,  
     3. Entraîne plusieurs modèles et compare leurs performances,  
     4. Sauvegarde chaque modèle,  
     5. Évalue ensuite chaque modèle sur l'ensemble du dataset (ou un sous-ensemble).

6. **`transactions_x.csv`**  
   - Exemple de fichier de transactions.  

## Installation & Exécution

1. **Cloner ou copier** ce repository sur votre machine locale.

2. **Installer les dépendances** (si nécessaire) :
   ```bash
   pip install pandas numpy scikit-learn matplotlib
3. **Lancer le script principal** :
    ```bash
    python main.py
    ```

    Cela affichera les graphiques et quelques informations dans le terminal (mise en forme du DataFrame, etc.).
    Le script entraînera également plusieurs modèles (ex. LinearRegression, RandomForestRegressor), comparera leur MAE/RMSE, et sauvegardera chaque modèle (.pkl).