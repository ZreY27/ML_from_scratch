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
HIDDEN = 32          # 32 > 128 en test (84,7 % vs 84,2 % à 8500 img/classe) avec 4x moins de poids
                     # et un entraînement 3x plus rapide — cf. revision/Experience_Standardisation.md §5
LEARNING_RATE = 0.01
DECAY = 0.00002      # décroissance inverse lr·1/(1+decay·step) ; très douce pour garder le lr vivant sur toutes les étapes
TRAINING_STEPS = 1200000  # ~42 passes sur ~20 400 images de train (plafond 8500/classe, split 80%)
N_VARIANTS = 5       # le MLP est LE plus sensible à l'init (58,8 % vs 82,7 % observés à code égal !)
MAX_PER_CLASS = 12000   # plafond par classe (équilibrage, ~max de Fighter) ; None pour tout prendre
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
        print(f"Il faut au moins 2 classes (sous-dossiers d'images) dans {DATASETS_DIR}. Trouvé : {classes}")
        return

    data = tu.load_dataset(DATASETS_DIR, classes, max_per_class=MAX_PER_CLASS)
    print(f"Classes : {classes}")
    print(f"Images par classe (plafond {MAX_PER_CLASS}) : {tu.counts(data)}")

    train, test = tu.train_test_split(data, test_ratio=TEST_RATIO)

    # Chemins + labels one-hot (±1) aplatis, alignés
    train_paths, labels_flat = [], []
    for cls in classes:
        onehot = [1.0 if c == cls else -1.0 for c in classes]
        for path in train[cls]:
            train_paths.append(path)
            labels_flat.extend(onehot)
    print(f"Train : {len(train_paths)} images | Test : {sum(len(v) for v in test.values())} images")

    # Un variant = un MLP complet (3072 -> couche cachée -> 1 sortie par classe),
    # avec une initialisation aléatoire différente à chaque appel.
    def entrainer_un_variant():
        model = ML_ESGI.MLP([INPUT_SIZE, HIDDEN, len(classes)], is_classification=True)
        loss = model.train_from_images(train_paths, labels_flat, IMAGE_WIDTH, IMAGE_HEIGHT,
                                       TRAINING_STEPS, LEARNING_RATE, DECAY)
        return model, loss

    def evaluer_un_variant(variant):
        model, _ = variant
        p = reg.Predictor({"type": "mlp", "classes": classes}, model=model)
        acc, _ = tu.evaluate(p, test, IMAGE_WIDTH, IMAGE_HEIGHT)
        return acc

    # N variants -> moyenne ± écart-type (rapport), puis BAGGING (moyenne des sorties)
    t0 = time.perf_counter()
    resultats, variant_stats = tu.entrainer_variants(
        N_VARIANTS, entrainer_un_variant, evaluer_un_variant)
    elapsed = time.perf_counter() - t0
    print(f"Temps d'entrainement ({N_VARIANTS} variants) : {tu.format_duration(elapsed)}")

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

    # Log TensorBoard : les courbes de loss de TOUS les variants, pas seulement le
    # meilleur -> la dispersion entre inits (58,8 % vs 82,7 % !) se lit sur un même graphe.
    losses_tb = {f"var{i + 1}": courbe for i, (_, (_, courbe)) in enumerate(resultats)}
    tu.log_to_tensorboard(f"mlp_genres_v{version}", losses=losses_tb,
                          scalars={"accuracy": accuracy, "accuracy_train": accuracy_train})

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
