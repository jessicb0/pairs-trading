# Projet de Recherche - Pair Trading Quantitatif

## Présentation

Ce projet implémente un pipeline complet de recherche quantitative pour le pair trading et l’arbitrage statistique en utilisant :

- Arbitrage statistique
- Apprentissage non supervisé
- Machine Learning supervisé
- Deep Learning (LSTM)
- Analyse de sentiment NLP
- Ensemble Learning
- Backtesting et optimisation
- Analyse des coûts de transaction

Le projet est construit autour d’un workflow réaliste de pair trading sur des actions américaines.

## Rapport

Le rapport final du projet est disponible ici :

[Voir le rapport PDF](report/report.pdf)

# Fonctionnalités

## 1. Collecte des Données

Les données historiques des actions sont téléchargées via Yahoo Finance.

Fonctionnalités :
- Prix ajustés
- Rendements journaliers
- Nettoyage automatique des données manquantes



# 2. Apprentissage Non Supervisé

## Clustering Hiérarchique

Le projet regroupe les actions selon leur structure de corrélation.

Techniques utilisées :
- Matrice de corrélation
- Matrice de distance
- Clustering hiérarchique
- Dendrogramme

## Visualisation PCA

Une ACP (PCA) est utilisée afin de visualiser les clusters en 2D.



# 3. Analyse de Cointégration

Les paires sont sélectionnées grâce à :

- Filtrage par corrélation
- Test de cointégration d’Engle-Granger
- Seuil de p-value

Les meilleures paires cointégrées sont ensuite automatiquement sélectionnées.


# 4. Stratégie de Pair Trading

Le projet implémente une stratégie complète d’arbitrage statistique.

Composants principaux :

- Estimation du hedge ratio via OLS
- Construction du spread
- Normalisation via z-score
- Seuils d’entrée et de sortie
- Génération des signaux long/short
- Backtesting

## Formule du Spread

```math
Spread_t = Price_{A,t} - \beta Price_{B,t}
```

## Formule du Z-Score

```math
Z_t = \frac{Spread_t - \mu}{\sigma}
```


# 5. Tests de Stationnarité

Le projet vérifie si le spread est stationnaire.

Méthodes utilisées :
- Test Augmented Dickey-Fuller (ADF)
- Analyse ADF glissante

Cela permet de valider l’hypothèse de retour à la moyenne nécessaire au pair trading.


# 6. Analyse des Coûts de Transaction

Des coûts de transaction réalistes sont intégrés à la stratégie.

Coûts pris en compte :
- Commissions
- Bid-ask spread
- Slippage

Le projet inclut également :
- Analyse de sensibilité aux coûts
- Analyse du seuil de rentabilité



# 7. Optimisation des Seuils

La stratégie teste automatiquement plusieurs :

- Seuils d’entrée
- Seuils de sortie

Métriques d’optimisation :
- Rendement total
- Ratio de Sharpe
- Drawdown maximal
- Nombre de trades



# 8. Machine Learning Supervisé

## Random Forest Classifier

Prédit si le spread va converger.

Target :

```python
abs(Zscore_t+1) < abs(Zscore_t)
```

## Random Forest Regressor

Prédit le rendement futur de la stratégie à horizon J+1.

## XGBoost Classifier

Modèle de gradient boosting utilisé comme alternative plus avancée au Random Forest.



# 9. Deep Learning

## Réseau de Neurones LSTM

Un réseau LSTM (Long Short-Term Memory) est implémenté avec PyTorch.

Objectifs :
- Apprendre les dynamiques temporelles du spread
- Capturer les patterns séquentiels du marché
- Prédire la convergence future du spread
- Prédire le rendement futur de la stratégie

Variables utilisées :

```python
[
    "Zscore",
    "Spread",
    "Return_A",
    "Return_B"
]
```

Le LSTM utilise des séquences temporelles glissantes.



# 10. Analyse de Sentiment NLP

Le projet intègre une analyse de sentiment simplifiée.

Le système calcule :
- Nombre de mots positifs
- Nombre de mots négatifs
- Sentiment relatif entre les actifs

Ce score de sentiment est intégré au pipeline de décision.



# 11. Ensemble Learning - Meta Model

Un meta-model combine plusieurs sources d’information :

- Signal de pair trading
- Probabilité issue du machine learning
- Prédiction de rendement
- Sentiment NLP
- Prédictions deep learning

Le meta-model utilise une régression logistique afin d’estimer :


Probabilité de rendement futur positif

Cela remplace les pondérations heuristiques choisies manuellement.



# 12. Comparaison avec Buy & Hold

La stratégie est comparée à :

- Buy & Hold SHW
- Buy & Hold HD

Métriques :
- Rendement total
- Volatilité
- Ratio de Sharpe
- Drawdown maximal



# Outputs Générés

Le projet exporte automatiquement :

## CSV

- Matrices de corrélation
- Résultats de clustering
- Résultats de cointégration
- Signaux de trading
- Prédictions ML
- Prédictions LSTM
- Comparaisons benchmark
- Recommandations finales

## Figures

- Dendrogrammes
- PCA des clusters
- Graphiques du spread
- Graphiques du z-score
- Résultats du backtest
- Visualisations ML
- Courbes d’entraînement LSTM
- Comparaisons benchmark


# Dépendances Principales

pandas
numpy
matplotlib
scikit-learn
scipy
statsmodels
yfinance
xgboost
torch


# Notebook

La version notebook du projet est disponible dans :

notebooks/PairsTrading.ipynb

Le notebook suit le même pipeline que la version production.




# Technologies Utilisées

- Python
- PyTorch
- Scikit-learn
- XGBoost
- Statsmodels
- Matplotlib
- Pandas
- NumPy


# Améliorations Futures

Possibilités d’amélioration :

- Architectures Transformers
- Optimisation de portefeuille multi-paires
- Streaming temps réel
- Exécution live
- Intégration de données alternatives


