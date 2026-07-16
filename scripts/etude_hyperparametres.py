"""
etude_hyperparametres.py — Étude d'ablation des hyperparamètres des 3 modèles "réglables".

POURQUOI CE SCRIPT : à l'oral, dire "on a pris 128 neurones" ne suffit pas — il faut
montrer qu'on a ESSAYÉ autre chose et mesuré l'effet. Ce script fait varier UN
hyperparamètre à la fois (toutes choses égales par ailleurs : mêmes données, même
split, mêmes graines) et trace accuracy TRAIN vs TEST pour chaque valeur :

1. MLP  : architecture de la couche cachée — rien, 32, 128, 256 neurones, puis
          2 couches (128+64). Question : plus gros = mieux ? (spoiler : capacité
          -> sur-apprentissage, l'écart train-test se creuse)
2. RBF  : nombre de centres K-Means (10 -> 300) à gamma fixé, puis gamma
          (0.001 -> 0.1) à 150 centres. Centres = capacité ; gamma = largeur des
          gaussiennes (slide 103 : les deux interagissent).
3. SVM  : lambda de régularisation (0.0001 -> 0.1). Petit lambda = marge peu
          contrainte (proche perceptron) ; grand lambda = poids écrasés -> collapse.

PROTOCOLE : sous-ensemble de 2000 images/classe (l'étude compare des CONFIGS entre
elles, pas besoin des 36 000 images : ce sont les ÉCARTS qui nous intéressent, et
le dataset complet multiplierait la durée par ~6). Split 80/20 stratifié, graines 42
partout -> exactement reproductible.

COMMANDE POUR REPRODUIRE (depuis la racine du projet, build Release obligatoire) :
    python scripts/etude_hyperparametres.py

SORTIES :
    results/etude_hyperparametres.csv  (toutes les mesures brutes)
    results/etude_hyperparametres.png  (4 graphes train/test)

Auteur : Maxime Clément.
"""

import csv
import os
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)  # pour importer model_registry (situé à la racine)

import training_utils as tu
tu.enable_cpp_dlls()  # AVANT d'importer ML_ESGI

import ML_ESGI
import model_registry as reg
import matplotlib
matplotlib.use("Agg")  # rendu fichier uniquement (pas de fenêtre bloquante)
import matplotlib.pyplot as plt

# --- Protocole commun (identique pour toutes les configs : seule l'étude varie) ---
IMAGE_WIDTH = IMAGE_HEIGHT = 32
INPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3  # 3072
MAX_PER_CLASS = 2000   # sous-ensemble d'étude (voir docstring) ; graine 42 -> mêmes images à chaque run
TEST_RATIO = 0.2

# --- Grilles étudiées ---
# MLP : liste des couches cachées ([] = aucun neurone caché -> équivalent d'un
# modèle linéaire à sortie tanh : bon point de référence "sans profondeur").
MLP_ARCHIS = [[], [32], [128], [256], [128, 64]]
MLP_PASSES = 50          # nb de passes sur le train (steps = passes * nb d'images)
MLP_LEARNING_RATE = 0.01
MLP_DECAY = 0.00002
MLP_VARIANTS = 2         # le MLP est très sensible à l'init (58,8 % vs 82,7 % observés !)
                         # -> 2 inits par archi, on trace la moyenne

RBF_CENTRES = [10, 50, 150, 300]      # à gamma fixé (0.01, la valeur retenue en prod)
RBF_GAMMAS = [0.001, 0.01, 0.1]       # à 150 centres fixés (la valeur retenue en prod)
RBF_GAMMA_FIXE = 0.01
RBF_CENTRES_FIXE = 150

SVM_LAMBDAS = [0.0001, 0.001, 0.01, 0.1]
SVM_LEARNING_RATE = 0.001
SVM_EPOCHS = 500         # identiques au script de prod -> comparables

DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")


def mesurer(predictor, train, test):
    """Accuracy TRAIN et TEST du predictor (l'écart mesure le sur-apprentissage)."""
    acc_test, _ = tu.evaluate(predictor, test, IMAGE_WIDTH, IMAGE_HEIGHT)
    acc_train, _ = tu.evaluate(predictor, train, IMAGE_WIDTH, IMAGE_HEIGHT)
    return acc_train, acc_test


def main():
    classes = tu.discover_classes(DATASETS_DIR)
    if len(classes) < 2:
        print(f"Il faut au moins 2 classes dans {DATASETS_DIR}. Trouvé : {classes}")
        return
    data = tu.load_dataset(DATASETS_DIR, classes, max_per_class=MAX_PER_CLASS)
    train, test = tu.train_test_split(data, test_ratio=TEST_RATIO)
    print(f"Classes : {classes} | plafond étude : {MAX_PER_CLASS}/classe")
    print(f"Train : {sum(len(v) for v in train.values())} | Test : {sum(len(v) for v in test.values())}")

    # Chemins + labels one-hot (±1) aplatis (format attendu par MLP/RBF.train_from_images)
    train_paths, labels_flat, train_classes = [], [], []
    for k, cls in enumerate(classes):
        onehot = [-1.0] * len(classes)
        onehot[k] = 1.0
        for path in train[cls]:
            train_paths.append(path)
            labels_flat.extend(onehot)
            train_classes.append(cls)
    n_train = len(train_paths)

    lignes = []  # toutes les mesures -> CSV (modele, config, variant, acc_train, acc_test, duree_s)

    # ================= 1. MLP : taille et profondeur de la partie cachée =================
    steps = MLP_PASSES * n_train
    mlp_courbe = []  # [(label, acc_train_moy, acc_test_moy)]
    for archi_cachee in MLP_ARCHIS:
        label = "aucune" if not archi_cachee else "+".join(str(n) for n in archi_cachee)
        accs_tr, accs_te = [], []
        for v in range(MLP_VARIANTS):
            t0 = time.perf_counter()
            npl = [INPUT_SIZE] + archi_cachee + [len(classes)]
            model = ML_ESGI.MLP(npl, is_classification=True)
            model.train_from_images(train_paths, labels_flat, IMAGE_WIDTH, IMAGE_HEIGHT,
                                    steps, MLP_LEARNING_RATE, MLP_DECAY)
            p = reg.Predictor({"type": "mlp", "classes": classes}, model=model)
            acc_tr, acc_te = mesurer(p, train, test)
            duree = time.perf_counter() - t0
            print(f"[MLP {label:>7}] variant {v + 1}/{MLP_VARIANTS} : "
                  f"TRAIN {acc_tr:.1%} | TEST {acc_te:.1%} | {tu.format_duration(duree)}")
            lignes.append(["mlp", f"cachee={label}", v + 1, acc_tr, acc_te, round(duree, 1)])
            accs_tr.append(acc_tr)
            accs_te.append(acc_te)
        mlp_courbe.append((label, sum(accs_tr) / len(accs_tr), sum(accs_te) / len(accs_te)))

    # ================= 2a. RBF : nombre de centres (gamma fixé) =================
    rbf_centres_courbe = []
    for k in RBF_CENTRES:
        if k > n_train:
            print(f"[RBF] {k} centres > {n_train} images de train : ignoré.")
            continue
        t0 = time.perf_counter()
        model = ML_ESGI.RBF(INPUT_SIZE, k, output_size=len(classes),
                            gamma=RBF_GAMMA_FIXE, is_classification=True)
        model.train_from_images(train_paths, labels_flat, IMAGE_WIDTH, IMAGE_HEIGHT)
        p = reg.Predictor({"type": "rbf", "classes": classes}, model=model)
        acc_tr, acc_te = mesurer(p, train, test)
        duree = time.perf_counter() - t0
        print(f"[RBF {k:>3} centres] TRAIN {acc_tr:.1%} | TEST {acc_te:.1%} | {tu.format_duration(duree)}")
        lignes.append(["rbf", f"centres={k} gamma={RBF_GAMMA_FIXE}", 1, acc_tr, acc_te, round(duree, 1)])
        rbf_centres_courbe.append((k, acc_tr, acc_te))

    # ================= 2b. RBF : gamma (nombre de centres fixé) =================
    rbf_gamma_courbe = []
    for g in RBF_GAMMAS:
        t0 = time.perf_counter()
        model = ML_ESGI.RBF(INPUT_SIZE, RBF_CENTRES_FIXE, output_size=len(classes),
                            gamma=g, is_classification=True)
        model.train_from_images(train_paths, labels_flat, IMAGE_WIDTH, IMAGE_HEIGHT)
        p = reg.Predictor({"type": "rbf", "classes": classes}, model=model)
        acc_tr, acc_te = mesurer(p, train, test)
        duree = time.perf_counter() - t0
        print(f"[RBF gamma={g}] TRAIN {acc_tr:.1%} | TEST {acc_te:.1%} | {tu.format_duration(duree)}")
        lignes.append(["rbf", f"centres={RBF_CENTRES_FIXE} gamma={g}", 1, acc_tr, acc_te, round(duree, 1)])
        rbf_gamma_courbe.append((g, acc_tr, acc_te))

    # ================= 3. SVM : lambda de régularisation =================
    # Le SVM prend les images en 2D : on les charge UNE fois en RAM, réutilisées
    # pour les 4 lambdas x 3 classifieurs binaires (One-vs-Rest).
    print("\n[SVM] chargement des images de train en RAM...")
    X_train, valid_idx = [], []
    for i, path in enumerate(train_paths):
        try:
            X_train.append(list(ML_ESGI.load_and_resize_image(path, IMAGE_WIDTH, IMAGE_HEIGHT)))
            valid_idx.append(i)
        except Exception:
            pass
    classes_valides = [train_classes[i] for i in valid_idx]

    svm_courbe = []
    for lam in SVM_LAMBDAS:
        t0 = time.perf_counter()
        sub_models = []
        for cls in classes:
            Y = [1.0 if c == cls else -1.0 for c in classes_valides]
            svm = ML_ESGI.SVM(INPUT_SIZE, lambda_reg=lam)
            svm.train(X_train, Y, SVM_LEARNING_RATE, SVM_EPOCHS)
            sub_models.append(svm)
        p = reg.Predictor({"type": "onevsrest", "classes": classes}, sub_models=sub_models)
        acc_tr, acc_te = mesurer(p, train, test)
        duree = time.perf_counter() - t0
        print(f"[SVM lambda={lam}] TRAIN {acc_tr:.1%} | TEST {acc_te:.1%} | {tu.format_duration(duree)}")
        lignes.append(["svm", f"lambda={lam}", 1, acc_tr, acc_te, round(duree, 1)])
        svm_courbe.append((lam, acc_tr, acc_te))

    # ================= Sorties : CSV + figure =================
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "etude_hyperparametres.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modele", "config", "variant", "acc_train", "acc_test", "duree_s"])
        w.writerows(lignes)
    print(f"\nMesures brutes -> {csv_path}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Étude d'hyperparamètres ({MAX_PER_CLASS} img/classe, split 80/20, graines 42)")

    # MLP : barres train/test par architecture
    ax = axes[0][0]
    labels = [l for l, _, _ in mlp_courbe]
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], [tr for _, tr, _ in mlp_courbe], width=0.4, label="train")
    ax.bar([i + 0.2 for i in x], [te for _, _, te in mlp_courbe], width=0.4, label="test")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title(f"MLP : couche(s) cachée(s) (moyenne de {MLP_VARIANTS} inits)")
    ax.set_xlabel("neurones cachés")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.legend()

    # RBF : centres (capacité)
    ax = axes[0][1]
    ax.plot([k for k, _, _ in rbf_centres_courbe], [tr for _, tr, _ in rbf_centres_courbe], "o-", label="train")
    ax.plot([k for k, _, _ in rbf_centres_courbe], [te for _, _, te in rbf_centres_courbe], "o-", label="test")
    ax.set_title(f"RBF : nombre de centres (gamma={RBF_GAMMA_FIXE})")
    ax.set_xlabel("centres K-Means")
    ax.set_ylabel("accuracy")
    ax.legend()

    # RBF : gamma (largeur des gaussiennes), échelle log
    ax = axes[1][0]
    ax.semilogx([g for g, _, _ in rbf_gamma_courbe], [tr for _, tr, _ in rbf_gamma_courbe], "o-", label="train")
    ax.semilogx([g for g, _, _ in rbf_gamma_courbe], [te for _, _, te in rbf_gamma_courbe], "o-", label="test")
    ax.set_title(f"RBF : gamma ({RBF_CENTRES_FIXE} centres)")
    ax.set_xlabel("gamma (log)")
    ax.set_ylabel("accuracy")
    ax.legend()

    # SVM : lambda, échelle log
    ax = axes[1][1]
    ax.semilogx([l for l, _, _ in svm_courbe], [tr for _, tr, _ in svm_courbe], "o-", label="train")
    ax.semilogx([l for l, _, _ in svm_courbe], [te for _, _, te in svm_courbe], "o-", label="test")
    ax.set_title("SVM : lambda de régularisation")
    ax.set_xlabel("lambda (log)")
    ax.set_ylabel("accuracy")
    ax.legend()

    fig.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "etude_hyperparametres.png")
    fig.savefig(png_path, dpi=130)
    print(f"Figure -> {png_path}")


if __name__ == "__main__":
    main()
