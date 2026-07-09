# Plan du rapport final (minimum 20 pages — exigence syllabus)

> Chaque section indique : l'exigence du syllabus qu'elle couvre, les courbes/figures à produire,
> et la source des données. Cochez au fur et à mesure. Objectif : que le jury retrouve
> **explicitement** chaque phrase du syllabus dans une section du rapport.

## 1. Introduction et problématique (1-2 pages)
- [ ] Problématique choisie (classification d'images par genre) et pourquoi c'est un problème
      « pour lequel une implémentation humaine serait difficile » (syllabus).
- [ ] Constitution du dataset : sources, méthode de collecte (`pushDataset.ipynb`), volumétrie par classe.
- [ ] Règle de Vapnik rappelée dans le README : ~10× plus d'exemples que de paramètres
      (32×32×3 = 3072 paramètres → objectif ~30 000 images, réalité assumée et discutée).

## 2. Architecture technique (2-3 pages)
- [ ] Schéma : C++ (LinearModel, MLP, RBF, SVM) → pybind11 → `ML_ESGI.pyd` → scripts Python
      → registre de manifestes JSON → API FastAPI → client web. (Exigence : « bibliothèque
      dynamique manipulée depuis des scripts python ».)
- [ ] Choix : pourquoi pybind11, pourquoi le format aplati 1D, coût des copies Python↔C++,
      pourquoi `train_from_images` (chargement des pixels côté C++).
- [ ] Zéro dépendance ML externe : STB = décodage d'image uniquement ; Gauss maison pour le RBF.

## 3. Les 4 modèles : théorie + implémentation (6-8 pages)
Pour chaque modèle : rappel théorique (slides), pseudo-code de NOTRE implémentation,
choix d'hyperparamètres, et différences assumées avec le cours.
- [ ] **Modèle linéaire** (Maxime) : règle de Rosenblatt (slide 65), labels ±1, biais séparé ;
      régression par descente de gradient et justification vs pseudo-inverse (slide 66).
- [ ] **MLP** (Antoine) : rétropropagation (slides 91-95), tanh, formules des deltas
      classification vs régression, SGD + learning rate decay.
- [ ] **RBF** (équipe) : version K centres (slides 108-112), Lloyd, moindres carrés,
      lien gamma = 1/(2σ²), estimation automatique de sigma, ridge λ=1e-4.
- [ ] **SVM** (Maxime) : marge maximale (slides 114-127), notre choix du problème PRIMAL
      (hinge loss + L2, paramètre lambda_reg) vs le dual QP du cours ; transformation
      explicite des entrées vs kernel trick ; SVR à tube epsilon.

## 4. Validation sur les cas de tests officiels (3-4 pages)
- [ ] Reprendre les figures du notebook `Cas_de_tests.ipynb` (11 cas, frontières de décision).
- [ ] Tableau récapitulatif OK/KO conforme aux annotations du cours.
- [ ] Analyse : pourquoi le linéaire échoue sur XOR/Cross/Multi Cross (linéaire en W, pas en X),
      pourquoi la transformation (x, y, x·y) suffit pour XOR, pourquoi le RBF excelle sur Cross.
      (Exigence : « démontrer la justesse de l'implémentation sur les cas de tests proposés ».)

## 5. Expérimentations sur le dataset réel (5-6 pages) — LE CŒUR DE LA NOTE
Exigence syllabus : « attention toute particulière à l'étude de l'impact des différents
paramètres des algorithmes sur la rapidité de convergence ».
- [ ] Protocole : split train/test stratifié 80/20, équilibrage par plafond, accuracy globale
      et par classe (déjà implémenté dans `training_utils.py` / manifestes).
- [ ] Courbes par modèle (TensorBoard → export matplotlib) :
      - [ ] **Sous-apprentissage** : modèle trop simple (linéaire sur images ; MLP trop petit).
      - [ ] **Sur-apprentissage** : accuracy train ↑ / accuracy test ↓ (MLP trop gros ou trop
            d'époques) → courbes train VS test superposées.
      - [ ] **Impact du learning rate** : trop grand (divergence/oscillation), trop petit (lenteur).
      - [ ] **Impact de la taille du dataset** : accuracy vs nb d'images/classe (50, 100, 200, 300).
      - [ ] **Biais dans la base d'exemples** : ex. classe déséquilibrée → l'argmax One-vs-Rest
            retombe sur la classe majoritaire (déjà observé : d'où MAX_PER_CLASS).
      - [ ] MLP : impact du decay ; RBF : impact du nombre de centres et de sigma ;
            SVM : impact de lambda_reg.
- [ ] Tableau final : accuracy test des 4 modèles + temps d'entraînement.
- [ ] « Suffisamment complexe pour apprendre, suffisamment simple pour généraliser » :
      illustrer avec la meilleure architecture MLP trouvée.

## 6. Comparaison avec TensorFlow/Keras (1-2 pages) — BONUS encouragé par le syllabus
- [ ] Même architecture MLP en Keras, mêmes données : accuracy et temps comparés aux nôtres.
- [ ] Analyse honnête des écarts (optimiseur Adam vs SGD pur, initialisations, etc.).

## 7. Application démonstrateur (1-2 pages)
- [ ] Screenshots du client web, flux upload → resize C++ → prédiction → réponse JSON.
- [ ] Versioning des modèles (manifestes), possibilité de charger toute version pré-entraînée.
      (Exigence : « sauvegarder/charger des modèles entraînés et les utiliser grâce au système
      client/serveur sur de nouvelles données ».)

## 8. Regard critique et conclusion (1-2 pages)
- [ ] Limites : dimension 3072 pour un linéaire, coût du RBF en O(n·centres), absence de
      convolutions (hors périmètre du cours), dataset réduit vs règle de Vapnik.
- [ ] Ce qu'on referait autrement. Ouverture (modèles du dernier slide du cours).

## Annexes
- [ ] Guide de build (CMake, pybind11) et d'exécution.
- [ ] Répartition du travail (voir SUIVI_EQUIPE.md) — exigence syllabus.
