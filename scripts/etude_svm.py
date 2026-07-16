"""
etude_svm.py — Étude approfondie du comportement du SVM (partie individuelle : Maxime).

CE QU'ON MESURE (et pourquoi c'est LA matière d'oral sur le SVM) :

1. LAMBDA (régularisation L2) de 0.0001 à 1.0 :
   - accuracy train/test : le compromis biais/variance piloté par lambda ;
   - la NORME des poids ||W|| et donc la MARGE géométrique 2/||W|| (slide SVM) :
     lambda pousse ||W|| vers le bas -> la marge S'ÉLARGIT ;
   - le NOMBRE DE VECTEURS SUPPORTS : les exemples avec y·(W·x+b) <= 1
     (sur la marge ou du mauvais côté). Marge plus large = plus d'exemples
     dedans = plus de vecteurs supports ;
   - lambda = 1.0 reproduit VOLONTAIREMENT le collapse historique du projet
     (poids écrasés vers 0 -> le modèle ne sépare plus rien).
   NB cours : notre lambda joue le rôle inverse du C du soft-margin (C ~ 1/lambda).

2. LEARNING RATE (0.0001 / 0.001 / 0.01) à lambda fixé :
   stabilité de la descente de sous-gradient (la hinge loss n'est pas dérivable
   en 1, on descend un sous-gradient) — trop grand = oscillations, trop petit =
   convergence lente.

3. CONVERGENCE : la hinge loss par epoch (loss_history du C++) pour chaque
   config -> on VOIT la descente et les plateaux.

PROTOCOLE : mêmes 2000 images/classe et même split 80/20 (graines 42) que
etude_hyperparametres.py -> les deux études sont comparables entre elles.
Le SVM prend les entrées en 2D : on charge train ET test UNE fois en RAM,
toutes les configs réutilisent les mêmes tableaux (aucune relecture disque).

COMMANDE POUR REPRODUIRE (racine du projet, build Release) :
    python -u scripts/etude_svm.py

SORTIES :
    results/etude_svm.csv   (toutes les mesures : normes, marges, SV, accuracies)
    results/etude_svm.png   (6 graphes)

Auteur : Maxime Clément.
"""

import csv
import math
import os
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import training_utils as tu
tu.enable_cpp_dlls()  # AVANT d'importer ML_ESGI

import ML_ESGI
import matplotlib
matplotlib.use("Agg")  # rendu fichier uniquement
import matplotlib.pyplot as plt

# --- Protocole commun (identique à etude_hyperparametres.py) ---
IMAGE_WIDTH = IMAGE_HEIGHT = 32
INPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3
MAX_PER_CLASS = 2000
TEST_RATIO = 0.2

# --- Grilles ---
LAMBDAS = [0.0001, 0.001, 0.01, 0.1, 1.0]   # 1.0 = le collapse historique, exprès
LR_ETUDE = [0.0001, 0.001, 0.01]            # à lambda fixé
LAMBDA_FIXE = 0.001                          # la valeur retenue en prod
LR_FIXE = 0.001
EPOCHS = 500                                 # comme le script de prod -> comparable

DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
TMP_SAVE = os.path.join(RESULTS_DIR, "_tmp_svm_poids.txt")  # pour lire ||W|| via save()


def charger_en_ram(split, classes):
    """Charge un split {classe: [chemins]} en (X 2D, labels classe) — une seule fois."""
    X, y = [], []
    for cls in classes:
        for path in split[cls]:
            try:
                X.append(list(ML_ESGI.load_and_resize_image(path, IMAGE_WIDTH, IMAGE_HEIGHT)))
                y.append(cls)
            except Exception:
                pass
    return X, y


def entrainer_ovr(X, y, classes, lam, lr, epochs):
    """Entraîne 1 SVM binaire par classe (One-vs-Rest). Retourne {classe: svm}."""
    models = {}
    for cls in classes:
        Y = [1.0 if c == cls else -1.0 for c in y]
        svm = ML_ESGI.SVM(INPUT_SIZE, lambda_reg=lam)
        svm.train(X, Y, lr, epochs)
        models[cls] = svm
    return models


def accuracy_ovr(models, classes, X, y):
    """Accuracy One-vs-Rest : classe prédite = celle du SVM au score brut max."""
    correct = 0
    for x, vrai in zip(X, y):
        scores = [models[c].predict_raw(x) for c in classes]
        if classes[scores.index(max(scores))] == vrai:
            correct += 1
    return correct / len(X) if X else 0.0


def norme_poids(svm):
    """||W|| (sans le biais), lue en sauvegardant le modèle puis en parsant le fichier.

    Le binding n'expose pas les poids directement : save() écrit
    "svm" / biais / lambda_reg / poids... -> on parse. La marge géométrique
    du cours vaut 2/||W|| : c'est la mesure clé de l'effet de lambda.
    """
    svm.save(TMP_SAVE)
    with open(TMP_SAVE, encoding="utf-8") as f:
        tokens = f.read().split()
    if tokens and tokens[0] == "svm":
        tokens = tokens[1:]
    poids = [float(t) for t in tokens[2:]]  # après biais et lambda_reg
    return math.sqrt(sum(w * w for w in poids))


def vecteurs_supports(svm, X, Y):
    """Nombre d'exemples avec y·(W·x+b) <= 1 : SUR la marge ou du mauvais côté.

    Ce sont eux qui « portent » l'hyperplan : les exemples à marge > 1 ont un
    sous-gradient de hinge nul -> ils n'influencent que via la régularisation.
    """
    n = 0
    for x, yi in zip(X, Y):
        if yi * svm.predict_raw(x) <= 1.0 + 1e-9:
            n += 1
    return n


def main():
    classes = tu.discover_classes(DATASETS_DIR)
    data = tu.load_dataset(DATASETS_DIR, classes, max_per_class=MAX_PER_CLASS)
    train, test = tu.train_test_split(data, test_ratio=TEST_RATIO)
    print(f"Classes : {classes} | plafond étude : {MAX_PER_CLASS}/classe")

    print("Chargement train + test en RAM (une seule fois)...")
    X_tr, y_tr = charger_en_ram(train, classes)
    X_te, y_te = charger_en_ram(test, classes)
    print(f"Train : {len(X_tr)} | Test : {len(X_te)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    lignes = []          # -> CSV
    courbes_lambda = {}  # lambda -> hinge loss moyenne des 3 classes, par epoch
    courbes_lr = {}      # lr -> idem
    stats_lambda = []    # [(lam, acc_tr, acc_te, marge_moy, norme_moy, pct_sv_moy)]
    stats_lr = []        # [(lr, acc_te)]

    # ================= Étude 1 : LAMBDA =================
    for lam in LAMBDAS:
        t0 = time.perf_counter()
        models = entrainer_ovr(X_tr, y_tr, classes, lam, LR_FIXE, EPOCHS)
        acc_tr = accuracy_ovr(models, classes, X_tr, y_tr)
        acc_te = accuracy_ovr(models, classes, X_te, y_te)

        normes, marges, pct_svs, histos = [], [], [], []
        for cls in classes:
            Y = [1.0 if c == cls else -1.0 for c in y_tr]
            nw = norme_poids(models[cls])
            nsv = vecteurs_supports(models[cls], X_tr, Y)
            normes.append(nw)
            marges.append(2.0 / nw if nw > 0 else float("inf"))
            pct_svs.append(nsv / len(X_tr))
            histos.append(list(models[cls].loss_history))
            lignes.append(["lambda", lam, cls, round(nw, 4), round(2.0 / nw, 4) if nw > 0 else "",
                           nsv, round(nsv / len(X_tr), 4),
                           round(histos[-1][-1], 6) if histos[-1] else "", acc_tr, acc_te])

        courbes_lambda[lam] = [sum(vals) / len(vals) for vals in zip(*histos)]
        stats_lambda.append((lam, acc_tr, acc_te,
                             sum(marges) / len(marges), sum(normes) / len(normes),
                             sum(pct_svs) / len(pct_svs)))
        print(f"[lambda={lam:<7}] TRAIN {acc_tr:.1%} | TEST {acc_te:.1%} | "
              f"||W||moy {sum(normes)/len(normes):.3f} | marge moy {sum(marges)/len(marges):.3f} | "
              f"SV {sum(pct_svs)/len(pct_svs):.1%} | {tu.format_duration(time.perf_counter()-t0)}")

    # ================= Étude 2 : LEARNING RATE =================
    for lr in LR_ETUDE:
        t0 = time.perf_counter()
        models = entrainer_ovr(X_tr, y_tr, classes, LAMBDA_FIXE, lr, EPOCHS)
        acc_te = accuracy_ovr(models, classes, X_te, y_te)
        histos = [list(models[cls].loss_history) for cls in classes]
        courbes_lr[lr] = [sum(vals) / len(vals) for vals in zip(*histos)]
        stats_lr.append((lr, acc_te))
        lignes.append(["learning_rate", lr, "(3 classes)", "", "", "", "", "", "", acc_te])
        print(f"[lr={lr:<7}] TEST {acc_te:.1%} | {tu.format_duration(time.perf_counter()-t0)}")

    if os.path.exists(TMP_SAVE):
        os.remove(TMP_SAVE)  # fichier temporaire de lecture des poids

    # ================= Sorties =================
    csv_path = os.path.join(RESULTS_DIR, "etude_svm.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["etude", "valeur", "classe", "norme_W", "marge_2_sur_W",
                    "nb_vecteurs_supports", "pct_vecteurs_supports",
                    "hinge_finale", "acc_train", "acc_test"])
        w.writerows(lignes)
    print(f"\nMesures brutes -> {csv_path}")

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(f"Étude SVM ({MAX_PER_CLASS} img/classe, split 80/20, lr={LR_FIXE}, {EPOCHS} epochs)")

    lams = [s[0] for s in stats_lambda]
    ax = axes[0][0]
    ax.semilogx(lams, [s[1] for s in stats_lambda], "o-", label="train")
    ax.semilogx(lams, [s[2] for s in stats_lambda], "o-", label="test")
    ax.set_title("Accuracy vs lambda")
    ax.set_xlabel("lambda (log)")
    ax.set_ylabel("accuracy")
    ax.legend()

    ax = axes[0][1]
    ax.loglog(lams, [s[4] for s in stats_lambda], "o-", color="tab:red", label="||W|| moyen")
    ax.loglog(lams, [s[3] for s in stats_lambda], "o-", color="tab:green", label="marge 2/||W||")
    ax.set_title("Régularisation : lambda écrase ||W||, la marge s'élargit")
    ax.set_xlabel("lambda (log)")
    ax.legend()

    ax = axes[0][2]
    ax.semilogx(lams, [s[5] * 100 for s in stats_lambda], "o-", color="tab:purple")
    ax.set_title("% de vecteurs supports (marge <= 1) vs lambda")
    ax.set_xlabel("lambda (log)")
    ax.set_ylabel("% du train")

    ax = axes[1][0]
    for lam, courbe in courbes_lambda.items():
        ax.plot(courbe, label=f"lambda={lam}")
    ax.set_title("Hinge loss par epoch (moyenne des 3 classes)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("hinge loss")
    ax.legend()

    ax = axes[1][1]
    for lr, courbe in courbes_lr.items():
        ax.plot(courbe, label=f"lr={lr}")
    ax.set_title(f"Hinge loss par epoch selon le learning rate (lambda={LAMBDA_FIXE})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("hinge loss")
    ax.legend()

    ax = axes[1][2]
    ax.semilogx([s[0] for s in stats_lr], [s[1] for s in stats_lr], "o-")
    ax.set_title("Accuracy test vs learning rate")
    ax.set_xlabel("learning rate (log)")
    ax.set_ylabel("accuracy test")

    fig.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "etude_svm.png")
    fig.savefig(png_path, dpi=130)
    print(f"Figure -> {png_path}")

    # Synthèse console (reprise ensuite dans le rapport)
    print("\n================= SYNTHÈSE SVM =================")
    for lam, acc_tr, acc_te, marge, norme, pct in stats_lambda:
        print(f"lambda={lam:<7} : TRAIN {acc_tr:.1%} | TEST {acc_te:.1%} | "
              f"||W|| {norme:.3f} | marge {marge:.3f} | SV {pct:.1%}")
    for lr, acc_te in stats_lr:
        print(f"lr={lr:<7}     : TEST {acc_te:.1%}")


if __name__ == "__main__":
    main()
