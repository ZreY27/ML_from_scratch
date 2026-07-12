"""
Entraînement d'un MLP multi-classe (one-hot ±1) pour classer les images par genre.

- Classes auto-détectées depuis datasets/ ; le nombre de sorties = nombre de classes.
- Équilibrage par plafond MAX_PER_CLASS (sinon le SGD ne voit presque que la classe majoritaire).
- Split train/test -> accuracy mesurée sur des données NON vues, stockée dans le manifeste.
- Sauvegarde versionnée via model_registry.
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
HIDDEN = 128
LEARNING_RATE = 0.01
DECAY = 0.00002      # décroissance inverse lr·1/(1+decay·step) ; très douce pour garder le lr vivant sur toutes les étapes
TRAINING_STEPS = 850000  # ~42 passes sur ~20 400 images de train (plafond 8500/classe, split 80%)
N_VARIANTS = 5       # le MLP est LE plus sensible à l'init (58,8 % vs 82,7 % observés à code égal !)
MAX_PER_CLASS = 8500   # plafond par classe (équilibrage, ~max de Fighter) ; None pour tout prendre
TEST_RATIO = 0.2
SHOW_PLOT = False  # True = affiche la courbe matplotlib (BLOQUANT). Les courbes sont déjà dans TensorBoard.

DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
MODELS_DIR = os.path.join(ROOT_DIR, "models")


def smooth(points, factor=0.99):
    """Lissage exponentiel pour une courbe de loss SGD très bruitée."""
    out = []
    for p in points:
        out.append(out[-1] * factor + p * (1 - factor) if out else p)
    return out


def main():
    classes = tu.discover_classes(DATASETS_DIR)
    if len(classes) < 2:
        print(f"Temps d'entrainement ({N_VARIANTS} variants MLP) : {tu.format_duration(elapsed)}")

    variants_mlp = [m for _, (m, _) in resultats]
    _, (_, loss) = max(resultats, key=lambda r: r[0])  # courbe de loss du meilleur variant

    # Le bag est le modèle déployé : évaluation détaillée
    predictor = reg.Predictor(
        {"type": "bag", "base_type": "mlp", "classes": classes},
        variants=[reg.Predictor({"type": "mlp", "classes": classes}, model=m)
                  for m in variants_mlp])
    accuracy, per_class = tu.evaluate(predictor, test, IMAGE_WIDTH, IMAGE_HEIGHT)
    print(f"\nBagging ({N_VARIANTS} variants) : {accuracy:.1%} "
          f"(meilleur variant seul : {variant_stats['accuracy_best']:.1%})")

    # Accuracy sur le TRAIN : l'écart train - test mesure le sur-apprentissage
    # (particulièrement parlant pour le MLP, qui a ~400 000 poids).
    accuracy_train, _ = tu.evaluate(predictor, train, IMAGE_WIDTH, IMAGE_HEIGHT)

    print(f"\nAccuracy TRAIN : {accuracy_train:.1%} | TEST : {accuracy:.1%} "
          f"(écart = {accuracy_train - accuracy:+.1%})")
    for c, a in per_class.items():
        print(f"  {c} : {a:.1%}" if a is not None else f"  {c} : (pas d'image de test)")

    # Sauvegarde versionnée : hyperparamètres (réglés) + métriques (mesurées) dans le manifeste
    hyperparams = {
        "input_size": INPUT_SIZE, "image_width": IMAGE_WIDTH, "image_height": IMAGE_HEIGHT,
        "architecture": [INPUT_SIZE, HIDDEN, len(classes)], "hidden": HIDDEN,
        "learning_rate": LEARNING_RATE, "decay": DECAY, "training_steps": TRAINING_STEPS,
        "n_variants": N_VARIANTS,
        "max_per_class": MAX_PER_CLASS, "test_ratio": TEST_RATIO,
    }
    version, manifest = reg.save_bag(
        variants_mlp, "mlp_genres", f"MLP - Genres (Bagging {N_VARIANTS} variants)",
        base_type="mlp", classes=classes,
        width=IMAGE_WIDTH, height=IMAGE_HEIGHT, models_dir=MODELS_DIR,
        hyperparams=hyperparams,
        metrics={"accuracy": accuracy, "accuracy_train": accuracy_train,
                 "accuracy_per_class": per_class,
                 "variants": variant_stats,
                 "counts": tu.counts(data),
                 "final_loss": loss[-1] if loss else None},
    )
    print(f"\nMLP sauvegardé : version v{version}\n  -> {manifest}")

    # Log TensorBoard (courbe de loss + accuracy)
    tu.log_to_tensorboard(f"mlp_genres_v{version}", losses=loss, scalars={"accuracy": accuracy})

    if SHOW_PLOT:
        plt.plot(smooth(loss), color="blue", label="MSE lissé (99%)")
        plt.plot(loss, color="lightblue", alpha=0.2, label="MSE brut")
        plt.title("Courbe d'apprentissage du MLP")
        plt.xlabel("Étapes d'entraînement (SGD)")
        plt.ylabel("Erreur quadratique moyenne (MSE)")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
