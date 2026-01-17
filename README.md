# 🏆 Ligue 1 McDonald's Predictor - Simulation 2025-2026

Une application interactive développée avec **Streamlit** pour simuler la saison de Ligue 1 en utilisant des modèles statistiques avancés.

🚀 Démo en direct : [https://ligue1byldesloges.streamlit.app]

## 📝 Présentation du projet
Ce projet utilise la Data Science pour transformer l'historique des scores de la Ligue 1 en un outil de prévision dynamique. L'objectif est de modéliser l'incertitude du sport et de projeter le classement final de la saison 2025-2026.

## ⚙️ Méthodologie Statistique

L'application repose sur trois piliers mathématiques :

1.  **Loi de Poisson :** Chaque match est simulé en calculant les espérances de buts (lambdas) pour l'équipe à domicile et à l'extérieur, basées sur leurs capacités offensives et défensives relatives.
2.  **Amortissement Exponentiel ($e^{-kt}$) :** Pour coller à la réalité du terrain, le modèle accorde un poids plus important aux résultats récents. L'état de forme actuel est ainsi mieux valorisé que les performances passées.
3.  **Simulations de Monte-Carlo :** Pour obtenir des probabilités fiables (titre, Europe, relégation), l'algorithme simule 1 000 saisons complètes. Les résultats sont ensuite agrégés sous forme de statistiques et d'une Heatmap de probabilités.

## 🛠️ Stack Technique

* **Langage :** Python 🐍
* **Analyse de données :** Pandas, NumPy
* **Statistiques :** SciPy (Poisson Distribution)
* **Visualisation :** Plotly (Graphiques dynamiques et interactifs), Matplotlib
* **Interface :** Streamlit

## 📂 Structure des fichiers

* `app.py` : Script principal de l'application.
* `data/` : Contient les fichiers CSV historiques et le calendrier 2025-2026.
* `include/` : Logos des clubs (format SVG).
* `requirements.txt` : Liste des dépendances Python nécessaires.

## 🚀 Installation locale

Si vous souhaitez exécuter ce projet sur votre machine :

1. Clonez le dépôt :
   ```bash
   git clone [https://github.com/VOTRE_NOM_UTILISATEUR/VOTRE_REPO.git](https://github.com/VOTRE_NOM_UTILISATEUR/VOTRE_REPO.git)
