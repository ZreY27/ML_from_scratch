"""
visualize_resized.py — Visualise ce que les modèles "voient" réellement :
l'image originale à côté de sa version passée par le pipeline C++
(`ML_ESGI.load_and_resize_image` : resize nearest-neighbor + normalisation /255).

Utilise le VRAI loader C++ (pas une réimplémentation Python) : ce qui s'affiche
est exactement le vecteur 32x32x3 donné aux modèles.

Usage (depuis la racine du projet) :
    python scripts/visualize_resized.py                          # 3 images par classe, 32x32
    python scripts/visualize_resized.py --per-class 5            # 5 images par classe
    python scripts/visualize_resized.py --sizes 16 32 64         # compare plusieurs tailles cibles
    python scripts/visualize_resized.py --image datasets/Racing/xxx.jpg   # une ou plusieurs images précises
    python scripts/visualize_resized.py --save apercu.png        # sauvegarde en PNG sans ouvrir de fenêtre
    python scripts/visualize_resized.py --seed 7                 # autre tirage aléatoire
"""

import os
import sys
import argparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import training_utils as tu
tu.enable_cpp_dlls()  # AVANT d'importer ML_ESGI

import ML_ESGI

DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Affiche les images redimensionnées telles que les modèles les reçoivent (pipeline C++).")
    parser.add_argument("--image", nargs="+", default=None, metavar="CHEMIN",
                        help="Chemin(s) d'image(s) précis à inspecter (sinon : tirage dans datasets/)")
    parser.add_argument("--per-class", type=int, default=3,
                        help="Nombre d'images tirées par classe (défaut : 3)")
    parser.add_argument("--classes", nargs="+", default=None, metavar="CLASSE",
                        help="Limite le tirage à ces classes (ex: --classes Fighter) ; défaut : toutes")
    parser.add_argument("--sizes", type=int, nargs="+", default=[32], metavar="N",
                        help="Tailles cibles NxN à comparer (défaut : 32)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Graine du tirage aléatoire (défaut : 42, comme les scripts d'entraînement)")
    parser.add_argument("--save", default=None, metavar="FICHIER.png",
                        help="Sauvegarde la planche en PNG au lieu d'ouvrir une fenêtre")
    parser.add_argument("--show", action="store_true",
                        help="Force l'ouverture de la fenêtre même avec --save")
    return parser.parse_args()


def load_resized(path, size):
    """Vecteur aplati du loader C++ -> image numpy [size, size, 3] (valeurs 0-1)."""
    import numpy as np
    vec = ML_ESGI.load_and_resize_image(path, size, size)
    return np.asarray(vec).reshape(size, size, 3)


def load_original(path):
    """Image originale pleine résolution (Pillow). Retourne (array, 'LxH')."""
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        return np.asarray(rgb), f"{rgb.width}x{rgb.height}"


def collect_samples(args):
    """Liste de (étiquette, chemin) : images passées en argument, ou tirage par classe."""
    if args.image:
        return [(os.path.basename(p), p) for p in args.image]

    classes = tu.discover_classes(DATASETS_DIR)
    if args.classes:  # filtre optionnel, insensible à la casse ("fighter" -> "Fighter")
        wanted = {c.lower() for c in args.classes}
        classes = [c for c in classes if c.lower() in wanted]
    if not classes:
        print(f"Aucune classe correspondante (sous-dossier d'images) trouvée dans {DATASETS_DIR}.")
        sys.exit(1)
    data = tu.load_dataset(DATASETS_DIR, classes, max_per_class=args.per_class, seed=args.seed)
    return [(cls, path) for cls in classes for path in data[cls]]


def main():
    args = parse_args()

    # Backend sans fenêtre si on ne fait que sauvegarder (utilisable en SSH/CI)
    import matplotlib
    if args.save and not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = collect_samples(args)

    # Une ligne par image : [originale | taille 1 | taille 2 | ...]
    rows = []
    for label, path in samples:
        try:
            original, dims = load_original(path)
            resized = [load_resized(path, s) for s in args.sizes]
            rows.append((label, path, original, dims, resized))
        except Exception as e:
            print(f"  Image ignorée : {path} ({e})")  # même politique que train_from_images
    if not rows:
        print("Aucune image lisible à afficher.")
        sys.exit(1)

    n_cols = 1 + len(args.sizes)
    fig, axes = plt.subplots(len(rows), n_cols,
                             figsize=(2.6 * n_cols, 2.6 * len(rows)), squeeze=False)
    fig.suptitle("Pipeline C++ load_and_resize_image — nearest-neighbor + normalisation /255",
                 fontsize=11)

    for r, (label, path, original, dims, resized) in enumerate(rows):
        ax = axes[r][0]
        ax.imshow(original)
        ax.set_title(f"Originale ({dims})", fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for c, (size, img) in enumerate(zip(args.sizes, resized), start=1):
            ax = axes[r][c]
            # interpolation="nearest" : les pixels 32x32 restent nets à l'écran
            ax.imshow(img, interpolation="nearest")
            ax.set_title(f"{size}x{size}  ({size * size * 3} entrées)", fontsize=8)
            ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Planche sauvegardée : {os.path.abspath(args.save)}")
    if args.show or not args.save:
        plt.show()


if __name__ == "__main__":
    main()
