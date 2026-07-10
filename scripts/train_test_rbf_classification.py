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
MAX_PER_CLASS = 4500   # plafond par classe (équilibrage, ~max de Fighter) ; None pour tout prendre
TEST_RATIO = 0.2

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

    # Chemins + labels one-hot (±1) alignés, aplatis comme attendu par le C++
    train_paths, labels_flat = [], []
    for k, cls in enumerate(classes):
        onehot = [-1.0] * len(classes)
        onehot[k] = 1.0
        for path in train[cls]:
            train_paths.append(path)
            labels_flat.extend(onehot)
    print(f"Train : {len(train_paths)} images | Test : {sum(len(v) for v in test.values())} images")

    if NUM_CENTERS > len(train_paths):
        print(f"NUM_CENTERS ({NUM_CENTERS}) > images de train ({len(train_paths)}) : impossible.")
        return

    # Un seul RBF multi-sorties (une sortie par classe, prédiction = argmax)
    model = ML_ESGI.RBF(INPUT_SIZE, NUM_CENTERS, output_size=len(classes),
                        sigma=SIGMA, is_classification=True)

    print(f"\nEntraînement RBF ({NUM_CENTERS} centres, sigma auto)...")
    # loss_history = [MSE finale] + [taux d'erreur train] (pas d'epochs : une seule passe)
    t0 = time.perf_counter()
    loss_history = model.train_from_images(train_paths, labels_flat, IMAGE_WIDTH, IMAGE_HEIGHT)
    elapsed = time.perf_counter() - t0
    print(f"⏱  Temps d'entraînement (RBF, {NUM_CENTERS} centres) : {tu.format_duration(elapsed)}")
    mse = loss_history[0]
    erreur_train = loss_history[1] if len(loss_history) > 1 else None

    # Évaluation sur le test (réutilise la logique d'inférence du Predictor, comme l'app)
    predictor = reg.Predictor({"type": "rbf", "classes": classes}, model=model)
    accuracy, per_class = tu.evaluate(predictor, test, IMAGE_WIDTH, IMAGE_HEIGHT)
    print(f"\nAccuracy test (RBF) : {accuracy:.1%}")
    for c, a in per_class.items():
        print(f"  {c} : {a:.1%}" if a is not None else f"  {c} : (pas d'image de test)")

    # Sauvegarde versionnée : hyperparamètres (réglés) + métriques (mesurées) dans le manifeste
    hyperparams = {
        "input_size": INPUT_SIZE, "image_width": IMAGE_WIDTH, "image_height": IMAGE_HEIGHT,
        "num_centers": NUM_CENTERS, "sigma": SIGMA,
        "max_per_class": MAX_PER_CLASS, "test_ratio": TEST_RATIO,
    }
    metrics = {"accuracy": accuracy, "accuracy_per_class": per_class,
               "mse_train": mse, "counts": tu.counts(data)}
    if erreur_train is not None:
        metrics["error_rate_train"] = erreur_train

    version, manifest = reg.save_single(
        model, "rbf_genres", "RBF - Genres (K-Means + moindres carrés)",
        model_type="rbf", classes=classes, width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
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
