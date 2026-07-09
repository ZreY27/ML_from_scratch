# Suivi de la répartition du travail

> Exigence du syllabus : « Le travail de chaque membre du groupe devra être clairement
> identifié (header de fichier, document de suivi, etc.) ».
> Tableau initialisé depuis l'historique git — **à corriger/compléter par l'équipe** si
> l'attribution réelle diffère (commits faits par un autre membre, pair programming, etc.).

| Partie | Fichiers principaux | Responsable |
|---|---|---|
| Modèle linéaire (Rosenblatt + régression) | `src/LinearModel.{hpp,cpp}` | Maxime Clément |
| SVM (hinge loss primal + SVR) | `src/SVM.{h,cpp}` | Maxime Clément |
| Cas de tests C++ | `src/main_test.cpp` | Maxime Clément |
| Cas de tests officiels (notebook Python) | `Cas_de_tests.ipynb` | Maxime Clément |
| Script de constitution du dataset | `pushDataset.ipynb` | Maxime Clément |
| Perceptron Multi-Couches (rétropropagation) | `src/MLP.{hpp,cpp}` | Antoine |
| Application client/serveur (FastAPI + web) | `apps/main.py`, `apps/index.html` | Antoine |
| Registre de modèles (manifestes, versioning) | `model_registry.py` | Antoine |
| RBF (K-Means + moindres carrés) | `src/RBF.{hpp,cpp}` | Équipe (Antoine, YumYumae, Maxime) |
| Scripts d'entraînement + TensorBoard | `scripts/train_test_*.py`, `scripts/training_utils.py` | À compléter |
| Chargeur d'images (STB) | `src/ImageLoader.{hpp,cpp}` | À compléter |
| Interfaçage pybind11 | `wrapper/binding.cpp` | À compléter |
| Rapport / Jupyter interactif / slides | `rapport/` | Toute l'équipe |

Note : les statistiques de commits ne reflètent pas toujours l'auteur réel du code
(merges, refactoring, commit effectué depuis la machine d'un autre membre). En cas d'écart,
c'est CE document + les headers de fichiers qui font foi — d'où l'importance de le tenir à jour.
