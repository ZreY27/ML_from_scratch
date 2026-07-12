"""
Entraînement One-vs-Rest de SVM (Hinge loss) pour classer les images par genre.

- Classes auto-détectées depuis datasets/, équilibrage par plafond, split train/test + accuracy.
- Sauvegarde versionnée via model_registry (One-vs-Rest, base_type="svm") + log TensorBoard.

Particularité SVM : il prend les données en 2D (liste de listes) et n'a PAS de train_from_images.
On charge donc les images en mémoire côté Python (une seule fois), puis on entraîne 1 SVM binaire
par classe en réutilisant le même tableau d'images.
"""

import os
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)  # pour importer model_registry (situé à la racine)

import training_utils as tu
tu.enable_cpp_dlls()  # AVANT d'importer ML_ESGI

import ML_ESGI
import model_registry as reg
import matplotlib.pyplot as plt

# --- Hyperparamètres ---
IMAGE_WIDTH = IMAGE_HEIGHT = 32
INPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3  # 3072
LAMBDA_REG = 0.001    # force de régularisation L2 (w -= lr·2·λ·w à chaque pas) ; λ=1.0 écrasait les poids → collapse. Petit λ ≈ perceptron à marge
                      # NB : ne pas confondre avec le C du soft-margin du cours (C ~ 1/λ : il pénalise les violations de marge, pas les poids)
LEARNING_RATE = 0.001
EPOCHS = 500
N_VARIANTS = 3        # inits différentes ; rapport = moyenne ± écart-type, app = bagging des variants
MAX_PER_CLASS = 8500   # plafond par classe (équilibrage, ~max de Fighter) ; None pour tout prendre
TEST_RATIO = 0.2
SHOW_PLOT = False     # True = affiche la courbe matplotlib (BLOQUANT). Les courbes sont déjà dans TensorBoard.

DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
MODELS_DIR = os.path.join(ROOT_DIR, "models")


def load_images_2d(paths):
    """Charge et aplatit les images en liste de listes (entrée 2D du SVM).

    Retourne (X, valid_classes_indices) : les indices des images qui ont bien chargé,
    pour garder les labels alignés (les images illisibles sont ignorées).
    """
    X, valid_idx = [], []
    for i, path in enumerate(paths):
        try:
            X.append(list(ML_ESGI.load_and_resize_image(path, IMAGE_WIDTH, IMAGE_HEIGHT)))
            valid_idx.append(i)
        except Exception as e:
            print(f"Temps d'entrainement ({N_VARIANTS} variants One-vs-Rest) : {tu.format_duration(elapsed)}")

    variants_models = [models for _, (models, _) in resultats]
    _, (_, losses_by_class) = max(resultats, key=lambda r: r[0])

    if SHOW_PLOT:
        for cls, l in losses_by_class.items():
            plt.plot(l, label=f"{cls} vs Rest")

    # Le bag est le modèle déployé : évaluation détaillée
    predictor = reg.Predictor(
        {"type": "bag", "base_type": "onevsrest", "classes": classes},
        variants=[reg.Predictor({"type": "onevsrest", "classes": classes},
                                sub_models=[m[c] for c in classes])
                  for m in variants_models])
    accuracy, per_class = tu.evaluate(predictor, test, IMAGE_WIDTH, IMAGE_HEIGHT)
    print(f"\nBagging ({N_VARIANTS} variants) : {accuracy:.1%} "
          f"(meilleur variant seul : {variant_stats['accuracy_best']:.1%})")

    # Accuracy sur le TRAIN : l'écart train - test mesure le sur-apprentissage.
    accuracy_train, _ = tu.evaluate(predictor, train, IMAGE_WIDTH, IMAGE_HEIGHT)

    print(f"\nAccuracy TRAIN : {accuracy_train:.1%} | TEST : {accuracy:.1%} "
          f"(écart = {accuracy_train - accuracy:+.1%})")
    for c, a in per_class.items():
        print(f"  {c} : {a:.1%}" if a is not None else f"  {c} : (pas d'image de test)")

    # Sauvegarde versionnée : hyperparamètres (réglés) + métriques (mesurées) dans le manifeste
    hyperparams = {
        "input_size": INPUT_SIZE, "image_width": IMAGE_WIDTH, "image_height": IMAGE_HEIGHT,
        "lambda_reg": LAMBDA_REG, "learning_rate": LEARNING_RATE, "epochs": EPOCHS,
        "n_variants": N_VARIANTS,
        "max_per_class": MAX_PER_CLASS, "test_ratio": TEST_RATIO, "strategy": "onevsrest",
    }
    version, manifest = reg.save_bag(
        variants_models, "svm_genres", f"SVM - Genres (Bagging {N_VARIANTS} variants)",
        base_type="onevsrest", sub_base_type="svm",
        classes=classes, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, models_dir=MODELS_DIR,
        hyperparams=hyperparams,
        metrics={"accuracy": accuracy, "accuracy_train": accuracy_train,
                 "accuracy_per_class": per_class,
                 "variants": variant_stats,
                 "counts": tu.counts(data)},
    )
    print(f"\nSVM One-vs-Rest sauvegardé : version v{version}\n  -> {manifest}")

    # Log TensorBoard (1 courbe de hinge loss par classe + accuracy)
    tu.log_to_tensorboard(f"svm_genres_v{version}", losses=losses_by_class,
                          scalars={"accuracy": accuracy})

    if SHOW_PLOT:
        plt.title("Hinge loss (SVM One-vs-Rest)")
        plt.xlabel("Epochs")
        plt.ylabel("Hinge loss")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
