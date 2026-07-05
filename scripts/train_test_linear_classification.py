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
MAX_PER_CLASS = 4500   # plafond par classe (équilibrage, ~max de Fighter) ; None pour tout prendre
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

    # Un perceptron binaire par classe (One-vs-Rest)
    models = {}
    losses_by_class = {}
    for cls in classes:
        print(f"\n--- {cls} vs RESTE ---")
        labels = [1.0 if c == cls else -1.0 for c in train_classes]
        model = ML_ESGI.LinearModel(INPUT_SIZE, is_classification=True)
        loss = model.train_from_images(train_paths, labels, IMAGE_WIDTH, IMAGE_HEIGHT,
                                       LEARNING_RATE, EPOCHS)
        models[cls] = model
        losses_by_class[cls] = loss
        if SHOW_PLOT:
            plt.plot(loss, label=f"{cls} vs Rest")

    # Évaluation sur le test (réutilise la logique d'inférence du Predictor, comme l'app)
    predictor = reg.Predictor({"type": "onevsrest", "classes": classes},
                              sub_models=[models[c] for c in classes])
    accuracy, per_class = tu.evaluate(predictor, test, IMAGE_WIDTH, IMAGE_HEIGHT)
    print(f"\nAccuracy test (One-vs-Rest) : {accuracy:.1%}")
    for c, a in per_class.items():
        print(f"  {c} : {a:.1%}" if a is not None else f"  {c} : (pas d'image de test)")

    # Sauvegarde versionnée : hyperparamètres (réglés) + métriques (mesurées) dans le manifeste
    hyperparams = {
        "input_size": INPUT_SIZE, "image_width": IMAGE_WIDTH, "image_height": IMAGE_HEIGHT,
        "learning_rate": LEARNING_RATE, "epochs": EPOCHS,
        "max_per_class": MAX_PER_CLASS, "test_ratio": TEST_RATIO, "strategy": "onevsrest",
    }
    version, manifest = reg.save_onevsrest(
        models, "linear_genres", "Perceptron - Genres (One-vs-Rest)",
        base_type="linear", width=IMAGE_WIDTH, height=IMAGE_HEIGHT, models_dir=MODELS_DIR,
        hyperparams=hyperparams,
        metrics={"accuracy": accuracy, "accuracy_per_class": per_class,
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
