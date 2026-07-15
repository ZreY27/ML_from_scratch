"""
train_one.py — Entraîne UNE config MLP (1 clone) et logge sa courbe dans TensorBoard.

Version "exploratoire parallélisable" du sweep : contrairement à sweep_mlp_hidden.py
(qui enchaîne 5 configs × N clones dans un seul process), ce script entraîne UNE seule
architecture avec UN clone, ce qui permet de lancer plusieurs configs EN PARALLÈLE
(un process par config) pour balayer rapidement profondeur / largeur / learning rate /
nombre de steps.

Le split train/test utilise la même graine (42) que le sweep -> les runs restent
comparables entre eux (seul l'axe balayé change).

Sortie :
  - runs/<name>/            : courbes TensorBoard (train/clone_1, test/clone_1) + scalars
                              accuracy_bag (= test, 1 clone) et accuracy_train.
  - <results_dir>/<name>.json : récap machine (arch, lr, steps, accuracy...) pour la synthèse.

Exemple :
  python train_one.py --name exp_h128 --arch 128 --lr 0.01 --steps 100000
"""

import argparse
import json
import os
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)      # model_registry (racine)
sys.path.insert(0, os.path.dirname(__file__))  # training_utils + ML_ESGI (scripts/)

import training_utils as tu
tu.enable_cpp_dlls()  # AVANT d'importer ML_ESGI

import ML_ESGI
import model_registry as reg

# --- Constantes fixes (identiques au sweep, pour rester comparable) ---
IMAGE_WIDTH = IMAGE_HEIGHT = 32
INPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3  # 3072
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")


def train_curve_windowed(loss_history, eval_steps, window):
    """Courbe de train alignée sur les steps d'éval : moyenne de la loss brute sur
    la fenêtre [s-window, s] (la loss SGD par step est trop bruitée pour un point isolé)."""
    points = []
    n = len(loss_history)
    for s in eval_steps:
        hi = min(s + 1, n)
        lo = max(0, hi - window)
        window_vals = loss_history[lo:hi]
        if window_vals:
            points.append((s, sum(window_vals) / len(window_vals)))
    return points


def prepare_test_flat(test_by_class, classes):
    """Pré-aplatit le set de test une fois (labels one-hot ±1, images illisibles ignorées)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="nom du run (sous-dossier runs/<name>)")
    ap.add_argument("--arch", required=True,
                    help="couches cachées séparées par des virgules, ex '128' ou '256,128,64'")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--decay", type=float, default=0.00002)
    ap.add_argument("--max-per-class", type=int, default=8500)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=0, help="0 -> steps//100")
    ap.add_argument("--logdir", default=os.path.join(ROOT_DIR, "runs"))
    ap.add_argument("--results-dir", default=None, help="où écrire <name>.json (défaut: pas de json)")
    args = ap.parse_args()

    hidden = [int(x) for x in args.arch.split(",") if x.strip()]
    eval_every = args.eval_every if args.eval_every > 0 else max(1, args.steps // 100)

    classes = tu.discover_classes(DATASETS_DIR)
    if len(classes) < 2:
        print(f"[{args.name}] Il faut au moins 2 classes dans {DATASETS_DIR}. Trouvé : {classes}")
        return

    arch = [INPUT_SIZE] + hidden + [len(classes)]

    data = tu.load_dataset(DATASETS_DIR, classes, max_per_class=args.max_per_class)
    train, test = tu.train_test_split(data, test_ratio=args.test_ratio)

    train_paths, train_labels_flat = [], []
    for cls in classes:
        onehot = [1.0 if c == cls else -1.0 for c in classes]
        for path in train[cls]:
            train_paths.append(path)
            train_labels_flat.extend(onehot)

    test_flat, test_labels_flat = prepare_test_flat(test, classes)

    print(f"[{args.name}] arch={arch} lr={args.lr} steps={args.steps} decay={args.decay} "
          f"| train={len(train_paths)} test={len(test_flat)//INPUT_SIZE} | eval_every={eval_every}",
          flush=True)

    t0 = time.perf_counter()
    model = ML_ESGI.MLP(arch, is_classification=True)
    loss = model.train_from_images(
        train_paths, train_labels_flat, IMAGE_WIDTH, IMAGE_HEIGHT,
        args.steps, args.lr, args.decay,
        test_flat, test_labels_flat, eval_every)
    train_secs = time.perf_counter() - t0

    eval_steps = list(model.eval_steps)
    test_loss = list(model.test_loss_history)

    predictor = reg.Predictor({"type": "mlp", "classes": classes}, model=model)
    acc_test, per_class = tu.evaluate(predictor, test, IMAGE_WIDTH, IMAGE_HEIGHT)
    acc_train, _ = tu.evaluate(predictor, train, IMAGE_WIDTH, IMAGE_HEIGHT)

    series = {
        "train/clone_1": train_curve_windowed(list(loss), eval_steps, eval_every),
        "test/clone_1": list(zip(eval_steps, test_loss)),
    }
    # accuracy_bag pour se superposer aux runs du sweep (1 clone -> bag == modèle seul)
    tu.log_run_curves(args.name, series,
                      scalars={"accuracy_bag": acc_test, "accuracy_train": acc_train},
                      logdir=args.logdir)

    print(f"[{args.name}] TEST={acc_test:.1%} TRAIN={acc_train:.1%} "
          f"(écart={acc_train-acc_test:+.1%}) | temps={tu.format_duration(train_secs)}", flush=True)

    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        with open(os.path.join(args.results_dir, f"{args.name}.json"), "w") as f:
            json.dump({
                "name": args.name, "arch": arch, "hidden": hidden,
                "lr": args.lr, "steps": args.steps, "decay": args.decay,
                "accuracy_test": acc_test, "accuracy_train": acc_train,
                "accuracy_per_class": per_class,
                "train_secs": train_secs,
                "final_train_loss": series["train/clone_1"][-1][1] if series["train/clone_1"] else None,
                "final_test_loss": test_loss[-1] if test_loss else None,
                "min_test_loss": min(test_loss) if test_loss else None,
            }, f, indent=2)


if __name__ == "__main__":
    main()
