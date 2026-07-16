"""
courbes_overfitting.py — Figure sur/sous-apprentissage pour LES 4 MODÈLES.

Pour chaque modèle on trace l'accuracy TRAIN et l'accuracy TEST quand la "capacité"
du modèle augmente. L'écart qui se creuse entre les deux = le sur-apprentissage.

Subtilité (à savoir expliquer) : la "capacité" n'a pas le même axe selon le modèle.
  - Linéaire, SVM, MLP : modèles ITÉRATIFS -> axe X = temps d'entraînement (epochs/steps).
    On les entraîne par paliers (train() ne réinitialise pas les poids : appels répétés
    = on continue l'entraînement), et on mesure train/test entre chaque palier.
  - RBF : PAS itératif (K-Means + moindres carrés en un coup) -> axe X = NOMBRE DE CENTRES.
    Peu de centres = sous-apprentissage ; 1 centre par exemple = sur-apprentissage
    (cf. cours, slides 104-107 : "Nombre de wi = nombre d'exemples ! Mauvais signe").

Choix pédagogique : PEU de données (300/classe) face à des modèles à forte capacité
-> le sur-apprentissage est net (règle de Vapnik : ~10x plus d'exemples que de paramètres).

Aucune modification C++. Sortie : results/courbes_overfitting.png + valeurs affichées.
Auteur : Maxime Clément.
"""
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import training_utils as tu
tu.enable_cpp_dlls()

import ML_ESGI
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Réglages communs ---
W = H = 32
INPUT_SIZE = 3072
MAX_PER_CLASS = 300     # peu de données -> overfitting visible
TEST_RATIO = 0.2
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

BLEU, ROUGE = "#2B5FD9", "#D9542B"


def charger(split, classes):
    imgs, vraies = [], []
    for k, cls in enumerate(classes):
        for path in split[cls]:
            try:
                imgs.append(ML_ESGI.load_and_resize_image(path, W, H))
                vraies.append(k)
            except Exception:
                pass
    return imgs, np.array(vraies)


def acc_ovr(models, imgs, vraies):
    """Accuracy One-vs-Rest : argmax des scores bruts des binaires."""
    preds = np.array([int(np.argmax([m.predict_raw(x) for m in models])) for x in imgs])
    return float(np.mean(preds == vraies))


def acc_multi(model, imgs, vraies):
    """Accuracy d'un modèle multi-sorties (MLP/RBF) : argmax de la sortie."""
    preds = np.array([int(np.argmax(model.predict(x))) for x in imgs])
    return float(np.mean(preds == vraies))


# --------------------------------------------------------------------------- #
def courbe_lineaire(classes, tr_imgs, tr_v, te_imgs, te_v):
    print("\n[Linéaire] paliers d'epochs...")
    flat = [v for img in tr_imgs for v in img]
    labels = {k: [1.0 if t == k else -1.0 for t in tr_v] for k in range(len(classes))}
    models = [ML_ESGI.LinearModel(INPUT_SIZE, is_classification=True) for _ in classes]
    xs, at, ae = [], [], []
    for p in range(1, 26):              # 25 paliers de 20 epochs = 500 epochs
        for k, m in enumerate(models):
            # Appel répété = on CONTINUE l'entraînement (les poids persistent entre appels)
            m.train(flat, labels[k], 0.01, 20)
        xs.append(p * 20)
        at.append(acc_ovr(models, tr_imgs, tr_v))
        ae.append(acc_ovr(models, te_imgs, te_v))
        print(f"  {xs[-1]:>4} epochs : train {at[-1]:.1%} | test {ae[-1]:.1%}")
    return ("Linéaire (One-vs-Rest)", "Epochs", xs, at, ae)


def courbe_svm(classes, tr_imgs, tr_v, te_imgs, te_v):
    print("\n[SVM] paliers d'epochs...")
    labels = {k: [1.0 if t == k else -1.0 for t in tr_v] for k in range(len(classes))}
    models = [ML_ESGI.SVM(INPUT_SIZE, lambda_reg=0.001) for _ in classes]
    xs, at, ae = [], [], []
    for p in range(1, 26):
        for k, s in enumerate(models):
            s.train(tr_imgs, labels[k], 0.001, 20)   # SVM prend X en 2D (liste de listes)
        xs.append(p * 20)
        at.append(acc_ovr(models, tr_imgs, tr_v))
        ae.append(acc_ovr(models, te_imgs, te_v))
        print(f"  {xs[-1]:>4} epochs : train {at[-1]:.1%} | test {ae[-1]:.1%}")
    return ("SVM (One-vs-Rest)", "Epochs", xs, at, ae)


def courbe_mlp(classes, tr_imgs, tr_v, te_imgs, te_v):
    print("\n[MLP] paliers de steps...")
    flat, flab = [], []
    for x, k in zip(tr_imgs, tr_v):
        flat.extend(x)
        oh = [-1.0] * len(classes); oh[k] = 1.0
        flab.extend(oh)
    model = ML_ESGI.MLP([INPUT_SIZE, 128, len(classes)], is_classification=True)
    xs, at, ae = [], [], []
    for p in range(1, 31):              # 30 paliers de 20000 steps = 600k
        model.train(flat, flab, 20000, 0.01, 0.0)
        xs.append(p * 20000)
        at.append(acc_multi(model, tr_imgs, tr_v))
        ae.append(acc_multi(model, te_imgs, te_v))
        print(f"  {xs[-1]:>7} steps : train {at[-1]:.1%} | test {ae[-1]:.1%}")
    return ("MLP (128 neurones cachés)", "Étapes SGD", xs, at, ae)


def courbe_rbf(classes, tr_imgs, tr_v, te_imgs, te_v):
    print("\n[RBF] balayage du nombre de centres...")
    flat, flab = [], []
    for x, k in zip(tr_imgs, tr_v):
        flat.extend(x)
        oh = [-1.0] * len(classes); oh[k] = 1.0
        flab.extend(oh)
    n_samples = len(tr_imgs)
    centres = [c for c in [5, 10, 20, 40, 80, 150, 300, 500, n_samples] if c <= n_samples]
    xs, at, ae = [], [], []
    for c in centres:
        # gamma=0.01 : valeur validée sur le dataset réel (cf. train_test_rbf, 79,7 % à 150 centres)
        model = ML_ESGI.RBF(INPUT_SIZE, c, output_size=len(classes), gamma=0.01, is_classification=True)
        model.train(flat, flab, n_samples)
        xs.append(c)
        at.append(acc_multi(model, tr_imgs, tr_v))
        ae.append(acc_multi(model, te_imgs, te_v))
        print(f"  {c:>4} centres : train {at[-1]:.1%} | test {ae[-1]:.1%}")
    return ("RBF (K-Means)", "Nombre de centres", xs, at, ae)


# --------------------------------------------------------------------------- #
def main():
    classes = tu.discover_classes(DATASETS_DIR)
    data = tu.load_dataset(DATASETS_DIR, classes, max_per_class=MAX_PER_CLASS)
    train, test = tu.train_test_split(data, test_ratio=TEST_RATIO)

    print("Chargement des images (une seule fois)...")
    tr_imgs, tr_v = charger(train, classes)
    te_imgs, te_v = charger(test, classes)
    print(f"Train : {len(tr_imgs)} | Test : {len(te_imgs)}")

    resultats = [
        courbe_lineaire(classes, tr_imgs, tr_v, te_imgs, te_v),
        courbe_svm(classes, tr_imgs, tr_v, te_imgs, te_v),
        courbe_rbf(classes, tr_imgs, tr_v, te_imgs, te_v),
        courbe_mlp(classes, tr_imgs, tr_v, te_imgs, te_v),
    ]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (titre, xlabel, xs, at, ae) in zip(axes.ravel(), resultats):
        ax.plot(xs, [a * 100 for a in at], "o-", color=BLEU, label="TRAIN")
        ax.plot(xs, [a * 100 for a in ae], "s-", color=ROUGE, label="TEST")
        i = int(np.argmax(ae))
        ax.axvline(xs[i], color="gray", ls="--", alpha=0.5)
        ax.set_title(titre)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Sur / sous-apprentissage des 4 modèles ({MAX_PER_CLASS} images/classe)",
                 fontsize=14)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "courbes_overfitting.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nFigure sauvegardée : {out}")


if __name__ == "__main__":
    main()
