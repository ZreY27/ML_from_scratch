"""
Sweep MLP — balayage de la TAILLE DE COUCHE CACHÉE.

5 runs, un par architecture. Chaque run entraîne N_CLONES MLP (initialisations
aléatoires différentes → une courbe de loss par clone) et les BAGGE (moyenne des
sorties, comme les "variants" du script train_test_mlp_classification.py).

Nouveauté : chaque run logge aussi une COURBE DE LOSS DE TEST par clone, évaluée
pendant l'entraînement tous les EVAL_EVERY steps (support C++ via eval_every).
Dans TensorBoard, un run = 5 courbes `train/clone_*` + 5 courbes `test/clone_*`,
alignées sur les mêmes steps réels.

Démarrer par la PASSE DE VALIDATION (STEPS/EVAL_EVERY réduits) pour vérifier tout
le pipeline, puis passer aux valeurs "run final" (voir plus bas).
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
LEARNING_RATE = 0.01
DECAY = 0.00002
N_CLONES = 3          # modèles par run (une courbe de loss chacun + bagging), comme les "variants"
MAX_PER_CLASS = 8500  # plafond par classe (équilibrage)
TEST_RATIO = 0.2

# STEPS = nombre d'étapes SGD par clone. 100k était déjà quasi convergent (~78 %, peu
# d'overfit) -> on vise 300k pour le run final (bon compromis convergence / temps).
# ⚠ Coût : ~1,5 h par run à 100k -> compter ~1 à 2 JOURS pour le sweep complet à 300k
# (5 configs × 5 clones, les gros réseaux h512/h256x128 dominent). Lancer la nuit / au calme.
STEPS = 300000
EVAL_EVERY = 3000     # évalue le test tous les EVAL_EVERY steps (→ STEPS/EVAL_EVERY = 100 points)

# L'axe qui varie : la couche cachée. La couche d'entrée (3072) et de sortie
# (n_classes) sont ajoutées automatiquement dans build_arch().
HIDDEN_CONFIGS = [
    ([64], "h64"),
    ([128], "h128"),
    ([256], "h256"),
    ([512], "h512"),
    ([256, 128], "h256x128"),
]

DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
MODELS_DIR = os.path.join(ROOT_DIR, "models")


def build_arch(hidden, n_classes):
    """[3072] + couches cachées + [n_classes] -> architecture complète du MLP."""
    return [INPUT_SIZE] + list(hidden) + [n_classes]


def prepare_test_flat(test_by_class, classes):
    """Pré-aplatit le set de test UNE fois (réutilisé par les 25 clones).

    Retourne (test_flat, test_labels_flat) : vecteurs plats alignés, labels one-hot ±1.
    Les images illisibles sont ignorées (labels gardés alignés).
    """
    test_flat, test_labels_flat = [], []
    for cls in classes:
        onehot = [1.0 if c == cls else -1.0 for c in classes]
        for path in test_by_class[cls]:
            try:
                img = ML_ESGI.load_and_resize_image(path, IMAGE_WIDTH, IMAGE_HEIGHT)
            except Exception:
                continue
            test_flat.extend(img)
            test_labels_flat.extend(onehot)
    return test_flat, test_labels_flat


def train_curve_windowed(loss_history, eval_steps, window):
    """Courbe de train alignée sur les steps d'éval : moyenne de la loss brute sur
    la fenêtre [s-window, s] (la loss SGD par step est trop bruitée pour un point isolé).

    Retourne [(step, mse_moyenne), ...] aux mêmes steps que la courbe de test.
    """
    points = []
    n = len(loss_history)
    for s in eval_steps:
        hi = min(s + 1, n)
        lo = max(0, hi - window)
        window_vals = loss_history[lo:hi]
        if window_vals:
            points.append((s, sum(window_vals) / len(window_vals)))
    return points


def main():
    classes = tu.discover_classes(DATASETS_DIR)
    if len(classes) < 2:
        print(f"Il faut au moins 2 classes dans {DATASETS_DIR}. Trouvé : {classes}")
        return

    data = tu.load_dataset(DATASETS_DIR, classes, max_per_class=MAX_PER_CLASS)
    print(f"Classes : {classes}")
    print(f"Images par classe (plafond {MAX_PER_CLASS}) : {tu.counts(data)}")

    train, test = tu.train_test_split(data, test_ratio=TEST_RATIO)

    # Chemins de train + labels one-hot (±1) aplatis, alignés (train chargé côté C++)
    train_paths, train_labels_flat = [], []
    for cls in classes:
        onehot = [1.0 if c == cls else -1.0 for c in classes]
        for path in train[cls]:
            train_paths.append(path)
            train_labels_flat.extend(onehot)

    # Set de test pré-aplati UNE fois, réutilisé par tous les runs/clones (pour la courbe de loss)
    test_flat, test_labels_flat = prepare_test_flat(test, classes)
    n_test = len(test_flat) // INPUT_SIZE
    print(f"Train : {len(train_paths)} images | Test : {n_test} images "
          f"(pré-aplaties pour la courbe de loss)")
    print(f"Config : STEPS={STEPS}, EVAL_EVERY={EVAL_EVERY} "
          f"(~{STEPS // EVAL_EVERY} points de test/clone), N_CLONES={N_CLONES}")

    recap = []  # (tag, arch, stats, acc_bag, acc_train, version)
    t_global = time.perf_counter()

    for hidden, tag in HIDDEN_CONFIGS:
        arch = build_arch(hidden, len(classes))
        print(f"\n{'=' * 70}\n### RUN {tag} — architecture {arch}\n{'=' * 70}")

        # Un clone = un MLP complet (init aléatoire différente à chaque construction).
        def entrainer_un_clone():
            model = ML_ESGI.MLP(arch, is_classification=True)
            loss = model.train_from_images(
                train_paths, train_labels_flat, IMAGE_WIDTH, IMAGE_HEIGHT,
                STEPS, LEARNING_RATE, DECAY,
                test_flat, test_labels_flat, EVAL_EVERY)
            # On renvoie tout ce qu'il faut pour les courbes + le bagging.
            return (model, list(loss), list(model.test_loss_history), list(model.eval_steps))

        def evaluer_un_clone(clone):
            model = clone[0]
            p = reg.Predictor({"type": "mlp", "classes": classes}, model=model)
            acc, _ = tu.evaluate(p, test, IMAGE_WIDTH, IMAGE_HEIGHT)
            return acc

        t0 = time.perf_counter()
        resultats, stats = tu.entrainer_variants(N_CLONES, entrainer_un_clone, evaluer_un_clone)
        print(f"Temps run {tag} ({N_CLONES} clones) : {tu.format_duration(time.perf_counter() - t0)}")

        clones = [c[0] for _, c in resultats]

        # Courbes TensorBoard : train/clone_i + test/clone_i, alignées sur les steps réels
        series = {}
        for i, (_, (_, loss, test_loss, eval_steps)) in enumerate(resultats, start=1):
            series[f"train/clone_{i}"] = train_curve_windowed(loss, eval_steps, EVAL_EVERY)
            series[f"test/clone_{i}"] = list(zip(eval_steps, test_loss))

        # Bag des N clones = modèle déployé
        predictor = reg.Predictor(
            {"type": "bag", "base_type": "mlp", "classes": classes},
            variants=[reg.Predictor({"type": "mlp", "classes": classes}, model=m) for m in clones])
        acc_bag, per_class = tu.evaluate(predictor, test, IMAGE_WIDTH, IMAGE_HEIGHT)
        acc_train, _ = tu.evaluate(predictor, train, IMAGE_WIDTH, IMAGE_HEIGHT)

        print(f"\n[{tag}] Bagging TEST : {acc_bag:.1%} | TRAIN : {acc_train:.1%} "
              f"(écart = {acc_train - acc_bag:+.1%}) | meilleur clone : {stats['accuracy_best']:.1%}")
        for c, a in per_class.items():
            print(f"  {c} : {a:.1%}" if a is not None else f"  {c} : (pas d'image de test)")

        tu.log_run_curves(f"mlp_{tag}", series,
                          scalars={"accuracy_bag": acc_bag, "accuracy_train": acc_train})

        # Sauvegarde versionnée du bag
        hyperparams = {
            "input_size": INPUT_SIZE, "image_width": IMAGE_WIDTH, "image_height": IMAGE_HEIGHT,
            "architecture": arch, "hidden": hidden,
            "learning_rate": LEARNING_RATE, "decay": DECAY, "training_steps": STEPS,
            "eval_every": EVAL_EVERY, "n_clones": N_CLONES,
            "max_per_class": MAX_PER_CLASS, "test_ratio": TEST_RATIO,
        }
        version, manifest = reg.save_bag(
            clones, f"mlp_sweep_{tag}", f"MLP hidden {arch} (bag {N_CLONES} clones)",
            base_type="mlp", classes=classes,
            width=IMAGE_WIDTH, height=IMAGE_HEIGHT, models_dir=MODELS_DIR,
            hyperparams=hyperparams,
            metrics={"accuracy": acc_bag, "accuracy_train": acc_train,
                     "accuracy_per_class": per_class,
                     "variants": stats,
                     "counts": tu.counts(data)},
        )
        print(f"[{tag}] sauvegardé : version v{version}\n  -> {manifest}")
        recap.append((tag, arch, stats, acc_bag, acc_train, version))

    # --- Tableau récapitulatif ---
    print(f"\n{'=' * 70}\nRÉCAP SWEEP — temps total : {tu.format_duration(time.perf_counter() - t_global)}\n{'=' * 70}")
    print(f"{'run':<10} {'architecture':<24} {'clones (moy±σ)':<18} {'bag test':<9} {'écart tr-te':<11}")
    for tag, arch, stats, acc_bag, acc_train, _ in recap:
        moy = f"{stats['accuracy_mean']:.1%}±{stats['accuracy_std']:.1%}"
        print(f"{tag:<10} {str(arch):<24} {moy:<18} {acc_bag:<9.1%} {acc_train - acc_bag:<+11.1%}")
    print("\nVisualiser les courbes : tensorboard --logdir runs")


if __name__ == "__main__":
    main()
