"""
Entraînement d'un réseau RBF multi-classe (K-Means + moindres carrés) pour classer
les images par genre.

- Aligné sur le pattern des autres scripts (linear/mlp/svm) : classes auto-détectées,
  équilibrage par plafond, split train/test, accuracy mesurée sur données NON vues,
  sauvegarde versionnée via model_registry (manifeste JSON -> visible dans l'app).
- Spécificité RBF : un SEUL modèle à 3 sorties (one-hot ±1), entraîné en une passe
  (Phase 1 : K-Means pour placer les centres, Phase 2 : moindres carrés pour les poids).
  Pas de courbe de loss par epoch : on logge la MSE et le taux d'erreur finaux.

Auteurs : équipe (modèle RBF) — script harmonisé avec les autres entraînements.
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

# --- Hyperparamètres ---
IMAGE_WIDTH = IMAGE_HEIGHT = 32
INPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3  # 3072
NUM_CENTERS = 50      # nombre de centres K-Means (= neurones cachés) ; doit rester <= nb d'images de train
SIGMA = 0.0           # 0.0 = estimation automatique depuis les centres (d_max / sqrt(2K))
N_VARIANTS = 3        # inits K-Means différentes ; rapport = moyenne ± écart-type, app = bagging
MAX_PER_CLASS = 8500   # plafond par classe (équilibrage, ~max de Fighter) ; None pour tout prendre
TEST_RATIO = 0.2

DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
MODELS_DIR = os.path.join(ROOT_DIR, "models")


def main():
    classes = tu.discover_classes(DATASETS_DIR)
    if len(classes) < 2:
        print(f"Temps d'entrainement ({N_VARIANTS} variants RBF) : {tu.format_duration(elapsed)}")

    variants_rbf = [m for _, (m, _) in resultats]
    _, (_, loss_history) = max(resultats, key=lambda r: r[0])  # métriques du meilleur variant
    mse = loss_history[0]
    erreur_train = loss_history[1] if len(loss_history) > 1 else None

    # Le bag est le modèle déployé : évaluation détaillée
    predictor = reg.Predictor(
        {"type": "bag", "base_type": "rbf", "classes": classes},
        variants=[reg.Predictor({"type": "rbf", "classes": classes}, model=m)
                  for m in variants_rbf])
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
        "num_centers": NUM_CENTERS, "sigma": SIGMA, "n_variants": N_VARIANTS,
        "max_per_class": MAX_PER_CLASS, "test_ratio": TEST_RATIO,
    }
    metrics = {"accuracy": accuracy, "accuracy_train": accuracy_train,
               "accuracy_per_class": per_class,
               "variants": variant_stats,
               "mse_train": mse, "counts": tu.counts(data)}
    if erreur_train is not None:
        metrics["error_rate_train"] = erreur_train

    version, manifest = reg.save_bag(
        variants_rbf, "rbf_genres", f"RBF - Genres (Bagging {N_VARIANTS} variants)",
        base_type="rbf", classes=classes, width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
        models_dir=MODELS_DIR, hyperparams=hyperparams, metrics=metrics,
    )
    print(f"\nRBF sauvegardé : version v{version}\n  -> {manifest}")

    # Log TensorBoard (valeurs finales : le RBF n'a pas de courbe par epoch)
    scalars = {"accuracy": accuracy, "mse_train": mse}
    if erreur_train is not None:
        scalars["error_rate_train"] = erreur_train
    tu.log_to_tensorboard(f"rbf_genres_v{version}", scalars=scalars)


if __name__ == "__main__":
    main()
