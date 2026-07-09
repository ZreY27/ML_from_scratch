"""
compare_keras.py — Comparaison de NOTRE MLP C++ avec un MLP équivalent Keras.

Cadre syllabus : l'utilisation d'implémentations externes est proscrite pour les modèles
du projet, MAIS « ils pourront tout à fait faire usage de la bibliothèque fournie de
tensorflow/keras à titre de comparaison de leurs implémentations personnelles ».
Ce script sert UNIQUEMENT à cette comparaison (section 6 du rapport).

Protocole : mêmes images, même split (train_test_split de training_utils, seed fixe),
même architecture (3072 -> HIDDEN -> nb_classes, activation tanh), même métrique.
On compare : accuracy test + temps d'entraînement + courbe de loss.

Prérequis : pip install tensorflow  (non ajouté à requirements.txt : dépendance lourde,
réservée à cette étude comparative).

Auteur : Maxime Clément.
"""

import os
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import training_utils as tu
tu.enable_cpp_dlls()  # AVANT d'importer ML_ESGI

import ML_ESGI
import numpy as np

# --- Hyperparamètres : ALIGNÉS sur train_test_mlp_classification.py pour comparer ---
IMAGE_WIDTH = IMAGE_HEIGHT = 32
INPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3
HIDDEN = 128
LEARNING_RATE = 0.01
TRAINING_STEPS = 60000       # notre MLP : 1 step = 1 exemple (SGD pur)
MAX_PER_CLASS = 300
TEST_RATIO = 0.2

DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")


def charger_donnees():
    """Images aplaties + labels one-hot (+1/-1), mêmes conventions que notre MLP."""
    classes = tu.discover_classes(DATASETS_DIR)
    if len(classes) < 2:
        raise SystemExit(f"Il faut au moins 2 classes dans {DATASETS_DIR}. Trouvé : {classes}")
    data = tu.load_dataset(DATASETS_DIR, classes, max_per_class=MAX_PER_CLASS)
    train, test = tu.train_test_split(data, test_ratio=TEST_RATIO)

    def vectorise(split):
        X, Y = [], []
        for k, cls in enumerate(classes):
            for path in split[cls]:
                try:
                    X.append(ML_ESGI.load_and_resize_image(path, IMAGE_WIDTH, IMAGE_HEIGHT))
                except Exception:
                    continue
                onehot = [-1.0] * len(classes)
                onehot[k] = 1.0
                Y.append(onehot)
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

    X_tr, Y_tr = vectorise(train)
    X_te, Y_te = vectorise(test)
    print(f"Classes : {classes} | train {len(X_tr)} images, test {len(X_te)} images")
    return classes, X_tr, Y_tr, X_te, Y_te


def notre_mlp(classes, X_tr, Y_tr, X_te, Y_te):
    """Notre MLP C++ (rétropropagation maison)."""
    print("\n=== NOTRE MLP C++ ===")
    model = ML_ESGI.MLP([INPUT_SIZE, HIDDEN, len(classes)], True)
    t0 = time.time()
    model.train(X_tr.flatten().tolist(), Y_tr.flatten().tolist(),
                TRAINING_STEPS, LEARNING_RATE)
    duree = time.time() - t0
    preds = np.array([int(np.argmax(model.predict(x))) for x in X_te.tolist()])
    acc = float(np.mean(preds == Y_te.argmax(axis=1)))
    print(f"Accuracy test : {acc:.1%} | temps d'entraînement : {duree:.1f}s")
    return acc, duree


def mlp_keras(classes, X_tr, Y_tr, X_te, Y_te):
    """MLP Keras d'architecture équivalente (SGD pur, pour comparer à armes égales,
    puis Adam, pour montrer l'apport d'un optimiseur moderne)."""
    import tensorflow as tf

    resultats = {}
    # Notre MLP fait TRAINING_STEPS tirages d'exemples ; Keras travaille en epochs :
    # equivalent epochs = steps / nb_exemples.
    epochs = max(1, TRAINING_STEPS // len(X_tr))

    for nom_opt, opt in [("SGD", tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)),
                         ("Adam", tf.keras.optimizers.Adam())]:
        print(f"\n=== KERAS MLP ({nom_opt}) — {epochs} epochs ===")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(INPUT_SIZE,)),
            tf.keras.layers.Dense(HIDDEN, activation="tanh"),
            tf.keras.layers.Dense(len(classes), activation="tanh"),  # sorties ±1 comme chez nous
        ])
        model.compile(optimizer=opt, loss="mse")  # même loss que notre implémentation
        t0 = time.time()
        model.fit(X_tr, Y_tr, epochs=epochs, batch_size=1, verbose=0)  # batch_size=1 = SGD pur
        duree = time.time() - t0
        preds = model.predict(X_te, verbose=0).argmax(axis=1)
        acc = float(np.mean(preds == Y_te.argmax(axis=1)))
        print(f"Accuracy test : {acc:.1%} | temps d'entraînement : {duree:.1f}s")
        resultats[nom_opt] = (acc, duree)
    return resultats


if __name__ == "__main__":
    classes, X_tr, Y_tr, X_te, Y_te = charger_donnees()
    acc_cpp, t_cpp = notre_mlp(classes, X_tr, Y_tr, X_te, Y_te)
    res_keras = mlp_keras(classes, X_tr, Y_tr, X_te, Y_te)

    print("\n===== BILAN COMPARATIF =====")
    print(f"  Notre MLP C++   : acc={acc_cpp:.1%}  temps={t_cpp:.1f}s")
    for nom, (acc, t) in res_keras.items():
        print(f"  Keras ({nom:<4})    : acc={acc:.1%}  temps={t:.1f}s")
    print("À commenter dans le rapport : écart d'accuracy (optimiseur, init), écart de temps.")
