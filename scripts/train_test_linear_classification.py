"""
Entraînement One-vs-Rest de perceptrons linéaires pour classer les images par genre.

- Classes auto-détectées depuis les sous-dossiers de datasets/.
- Équilibrage par plafond MAX_PER_CLASS (sinon une classe majoritaire écrase tout
  et l'argmax retombe toujours sur elle).
- Split train/test -> accuracy mesurée sur des données NON vues, stockée dans le manifeste.
- Sauvegarde versionnée via model_registry (One-vs-Rest : 1 binaire par classe).
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
LEARNING_RATE = 0.01
EPOCHS = 500
N_VARIANTS = 3        # entraînements complets avec inits différentes ; rapport = moyenne ± écart-type, app = meilleur
MAX_PER_CLASS = 8500   # plafond par classe (équilibrage, ~max de Fighter) ; None pour tout prendre
TEST_RATIO = 0.2
SHOW_PLOT = False  # True = affiche la courbe matplotlib (BLOQUANT). Les courbes sont déjà dans TensorBoard.

DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
MODELS_DIR = os.path.join(ROOT_DIR, "models")


def main():
    classes = tu.discover_classes(DATASETS_DIR)
    if len(classes) < 2:
        print(f"Il faut au moins 2 classes (sous-dossiers d'images) dans {DATASETS_DIR}. Trouvé : {classes}")
        return

    data = tu.load_dataset(DATASETS_DIR, classes, max_per_class=MAX_PER_CLASS)
    print(f"Classes : {classes}")
    print(f"Images par classe (plafond {MAX_PER_CLASS}) : {tu.counts(data)}")

    train, test = tu.train_test_split(data, test_ratio=TEST_RATIO)

    # Liste ordonnée (chemin, classe réelle) pour le train
    train_paths, train_classes = [], []
    for cls in classes:
        for path in train[cls]:
            train_paths.append(path)
            train_classes.append(cls)
    print(f"Train : {len(train_paths)} images | Test : {sum(len(v) for v in test.values())} images")

    # Un variant = un classifieur One-vs-Rest complet (1 perceptron binaire par classe).
    # Chaque appel repart d'initialisations aléatoires différentes.
    def entrainer_un_variant():
        models, losses = {}, {}
        for cls in classes:
            print(f"  {cls} vs RESTE")
            labels = [1.0 if c == cls else -1.0 for c in train_classes]
            model = ML_ESGI.LinearModel(INPUT_SIZE, is_classification=True)
            losses[cls] = model.train_from_images(train_paths, labels, IMAGE_WIDTH, IMAGE_HEIGHT,
                                                  LEARNING_RATE, EPOCHS)
            models[cls] = model
        return models, losses

    def evaluer_un_variant(variant):
        models, _ = variant
        predictor = reg.Predictor({"type": "onevsrest", "classes": classes},
                                  sub_models=[models[c] for c in classes])
        acc, _ = tu.evaluate(predictor, test, IMAGE_WIDTH, IMAGE_HEIGHT)
        return acc

    # N variants -> moyenne ± écart-type (rapport), puis BAGGING : à l'inférence on
    # moyenne les scores des N variants (moyenne des SORTIES, comme le
    # bag_y_pred = np.mean(folds_y_pred) du cours), et argmax.
    t0 = time.perf_counter()
    resultats, variant_stats = tu.entrainer_variants(
        N_VARIANTS, entrainer_un_variant, evaluer_un_variant)
    elapsed = time.perf_counter() - t0
    print(f"Temps d'entrainement ({N_VARIANTS} variants) : {tu.format_duration(elapsed)}")

    variants_models = [models for _, (models, _) in resultats]     # les N dicts {classe: modèle}
    _, (_, losses_by_class) = max(resultats, key=lambda r: r[0])   # courbes du meilleur variant

    if SHOW_PLOT:
        for cls, loss in losses_by_class.items():
            plt.plot(loss, label=f"{cls} vs Rest")

    # Le bag est le modèle déployé : évaluation détaillée (globale + par classe)
    predictor = reg.Predictor(
        {"type": "bag", "base_type": "onevsrest", "classes": classes},
        variants=[reg.Predictor({"type": "onevsrest", "classes": classes},
                                sub_models=[m[c] for c in classes])
                  for m in variants_models])
    accuracy, per_class = tu.evaluate(predictor, test, IMAGE_WIDTH, IMAGE_HEIGHT)
    print(f"\nBagging ({N_VARIANTS} variants) : {accuracy:.1%} "
          f"(meilleur variant seul : {variant_stats['accuracy_best']:.1%})")

    # Accuracy sur le TRAIN (mêmes images que l'entraînement) : l'écart train - test
    # révèle le sur-apprentissage (train >> test = par-coeur) ou le sous-apprentissage
    # (les deux basses = modèle trop simple). C'est la mesure clé pour le rapport.
    accuracy_train, _ = tu.evaluate(predictor, train, IMAGE_WIDTH, IMAGE_HEIGHT)

    print(f"\nAccuracy TRAIN : {accuracy_train:.1%} | TEST : {accuracy:.1%} "
          f"(écart = {accuracy_train - accuracy:+.1%})")
    for c, a in per_class.items():
        print(f"  {c} : {a:.1%}" if a is not None else f"  {c} : (pas d'image de test)")

    # Sauvegarde versionnée : hyperparamètres (réglés) + métriques (mesurées) dans le manifeste
    hyperparams = {
        "input_size": INPUT_SIZE, "image_width": IMAGE_WIDTH, "image_height": IMAGE_HEIGHT,
        "learning_rate": LEARNING_RATE, "epochs": EPOCHS, "n_variants": N_VARIANTS,
        "max_per_class": MAX_PER_CLASS, "test_ratio": TEST_RATIO, "strategy": "onevsrest",
    }
    version, manifest = reg.save_bag(
        variants_models, "linear_genres", f"Perceptron - Genres (Bagging {N_VARIANTS} variants)",
        base_type="onevsrest", sub_base_type="linear",
        classes=classes, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, models_dir=MODELS_DIR,
        hyperparams=hyperparams,
        metrics={"accuracy": accuracy, "accuracy_train": accuracy_train,
                 "accuracy_per_class": per_class,
                 "variants": variant_stats,   # moyenne ± écart-type des N runs (pour le rapport)
                 "counts": tu.counts(data)},
    )
    print(f"\nClassifieur One-vs-Rest sauvegardé : version v{version}\n  -> {manifest}")

    # Log TensorBoard (1 courbe de loss par classe + accuracy)
    tu.log_to_tensorboard(f"linear_genres_v{version}", losses=losses_by_class,
                          scalars={"accuracy": accuracy})

    if SHOW_PLOT:
        plt.title("Erreurs d'entraînement (One-vs-Rest)")
        plt.xlabel("Epochs")
        plt.ylabel("Ratio d'erreurs")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
