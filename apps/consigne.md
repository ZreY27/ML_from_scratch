# 🌐 Application Web — Prédiction d'images (FastAPI)

Interface web pour tester les modèles de Machine Learning entraînés (codés en C++, exposés via `ML_ESGI`).
L'app se **branche automatiquement sur le dossier `models/`** : aucun fichier de configuration à éditer à la main.

## 🚀 Lancer le serveur

Depuis la **racine du projet** (le dossier qui contient `apps/`, `src/`, `models/`, …) :

```bash
pip install -r requirements.txt        # une seule fois
uvicorn apps.main:app --reload
```
Puis ouvrir 👉 **http://127.0.0.1:8000**

> ⚠️ Le module C++ `ML_ESGI` doit être compilé au préalable (voir `README.md` / `CLAUDE.md`).
> Sous Windows, l'app charge automatiquement les DLL du runtime gcc (MSYS2).

## 🧠 Utilisation
1. Choisir un **modèle** dans la première liste.
2. Choisir une **version** (la plus récente par défaut ; l'accuracy de test est affichée si disponible).
3. Charger une image (jpg/png) et cliquer sur **Analyser**.

## ⚙️ Ajouter ou mettre à jour un modèle

Rien à coder côté app : **il suffit d'entraîner**.

```bash
python scripts/train_test_mlp_classification.py       # MLP multi-classe
python scripts/train_test_linear_classification.py    # One-vs-Rest (perceptrons)
python scripts/train_test_svm_classification.py       # One-vs-Rest (SVM, Hinge loss)
```

Chaque entraînement :
- détecte les classes (= sous-dossiers de `datasets/`),
- équilibre les classes + sépare train/test, mesure l'**accuracy**,
- sauvegarde une **nouvelle version** (jamais d'écrasement) : poids `<id>_v<N>.txt`
  **+ un manifeste `<id>_v<N>.json`** (classes, taille image, type, date, accuracy),
- logge la courbe d'apprentissage dans `runs/` (TensorBoard).

Au rechargement de la page, le nouveau modèle / la nouvelle version apparaît **automatiquement** dans le sélecteur.

### Sous le capot
- **`model_registry.py`** (racine) : écriture / listing des manifestes, versioning auto, et inférence
  (`Predictor` — gère le MLP, le perceptron binaire, le **One-vs-Rest**, le SVM et le RBF).
- **`apps/main.py`** scanne `models/*.json` au démarrage ; la route `/predict` charge le manifeste
  de la version choisie et renvoie la classe prédite.

## 📊 Courbes d'entraînement (TensorBoard)
```bash
tensorboard --logdir runs        # http://localhost:6006
```
Compare **toutes les versions** (loss + accuracy) sur un même graphe — pratique pour choisir la meilleure.

## 📁 Exemple de manifeste (One-vs-Rest)
```json
{
  "id": "linear_genres",
  "name": "Perceptron - Genres (One-vs-Rest)",
  "version": 3,
  "type": "onevsrest",
  "base_type": "linear",
  "width": 32, "height": 32,
  "classes": ["Fighter", "Racing", "Platformer"],
  "weights": {
    "Fighter": "linear_genres_v3_fighter.txt",
    "Racing": "linear_genres_v3_racing.txt",
    "Platformer": "linear_genres_v3_platformer.txt"
  },
  "metrics": { "accuracy": 0.82 }
}
```

> Note : les poids (`*.txt`) et les manifestes (`models/*.json`) sont **locaux** à chaque machine
> (gitignorés). Pour partager un modèle, il faut partager ses fichiers ou ré-entraîner.
