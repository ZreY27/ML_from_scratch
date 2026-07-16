"""
compare_keras.py — Comparaison de NOTRE MLP C++ avec un MLP equivalent Keras.

Cadre syllabus : les implementations externes sont proscrites pour les modeles du
projet, MAIS "ils pourront tout a fait faire usage de tensorflow/keras a titre de
comparaison de leurs implementations personnelles". Ce script ne sert qu'a cela.

ARCHITECTURE DE LA COMPARAISON (2 environnements Python) :
- Notre MLP C++ tourne sous Python 3.14 (ML_ESGI.pyd compile en cp314) ; ses
  resultats sont deja mesures et stockes dans les manifestes models/mlp_genres_*.json.
  Ce script LIT ces manifestes (pur JSON) au lieu de re-entrainer.
- TensorFlow n'existe pas pour Python 3.14 -> ce script s'execute sous Python 3.12
  (venv .venv-tf). Il ne peut donc PAS importer ML_ESGI : les images sont chargees
  avec PIL (resize 32x32 + normalisation /255, equivalent du loader C++).

PROTOCOLE IDENTIQUE au script C++ : memes dossiers datasets/, meme plafond
d'equilibrage, meme split 80/20 stratifie aux memes graines (42) -> les listes de
fichiers train/test sont EXACTEMENT les memes que celles du run C++.

Difference documentee : notre MLP fait du SGD pur (batch = 1 exemple) ; a batch=1,
l'overhead TensorFlow rend l'entrainement prohibitif -> batch_size=32 ici, en
gardant le meme nombre de PASSES sur les donnees (~42 epochs). A commenter dans
le rapport. On compare SGD (a armes egales) puis Adam (apport d'un optimiseur moderne).

COMMANDES POUR REPRODUIRE (depuis la racine du projet) :
    py -3.12 -m venv .venv-tf
    .venv-tf\\Scripts\\python -m pip install tensorflow pillow numpy
    .venv-tf\\Scripts\\python scripts\\compare_keras.py

Auteur : Maxime Clement.
"""

import os
import sys
import json
import glob
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

# training_utils n'importe ML_ESGI que dans evaluate() (import tardif) :
# on peut donc reutiliser discover_classes / load_dataset / train_test_split
# sous Python 3.12 sans toucher au .pyd.
import training_utils as tu

import numpy as np
from PIL import Image

# --- Hyperparametres : ALIGNES sur train_test_mlp_classification.py ---
IMAGE_WIDTH = IMAGE_HEIGHT = 32
INPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3
HIDDEN = 128
LEARNING_RATE = 0.01
MAX_PER_CLASS = 12000
TEST_RATIO = 0.2
EPOCHS = 42          # ~= 1.2M steps C++ / 28 800 images = 42 passes sur le train
BATCH_SIZE = 32      # difference documentee vs notre SGD pur (batch=1)

DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
MODELS_DIR = os.path.join(ROOT_DIR, "models")


def charger_image(path):
    """Equivalent PIL du loader C++ : resize 32x32, aplati, normalise 0-1."""
    img = Image.open(path).convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    return np.asarray(img, dtype=np.float32).flatten() / 255.0


def charger_split(split, classes):
    X, Y = [], []
    for k, cls in enumerate(classes):
        onehot = [-1.0] * len(classes)
        onehot[k] = 1.0
        for path in split[cls]:
            try:
                X.append(charger_image(path))
                Y.append(onehot)
            except Exception:
                pass
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def reference_cpp():
    """Lit le dernier manifeste du MLP C++ (models/mlp_genres_*.json)."""
    manifests = []
    for p in glob.glob(os.path.join(MODELS_DIR, "mlp_genres_v*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                manifests.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    if not manifests:
        return None
    return max(manifests, key=lambda m: m.get("version", 0))


def main():
    # 1. Memes donnees, meme split que le run C++ (graines identiques)
    classes = tu.discover_classes(DATASETS_DIR)
    data = tu.load_dataset(DATASETS_DIR, classes, max_per_class=MAX_PER_CLASS)
    train, test = tu.train_test_split(data, test_ratio=TEST_RATIO)
    print(f"Classes : {classes} | plafond {MAX_PER_CLASS}/classe")

    print("Chargement des images (PIL)...")
    X_tr, Y_tr = charger_split(train, classes)
    X_te, Y_te = charger_split(test, classes)
    print(f"Train : {len(X_tr)} | Test : {len(X_te)}")

    # 2. MLP Keras d'architecture identique : 3072 -> 128 (tanh) -> 3 (tanh), loss MSE
    import tensorflow as tf
    resultats = {}
    for nom, make_opt in [("SGD",  lambda: tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)),
                          ("Adam", lambda: tf.keras.optimizers.Adam())]:
        print(f"\n=== Keras {nom} : {EPOCHS} epochs, batch {BATCH_SIZE} ===")
        tf.keras.utils.set_random_seed(42)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(INPUT_SIZE,)),
            tf.keras.layers.Dense(HIDDEN, activation="tanh"),
            tf.keras.layers.Dense(len(classes), activation="tanh"),  # sorties ±1 comme le C++
        ])
        model.compile(optimizer=make_opt(), loss="mse")
        t0 = time.time()
        model.fit(X_tr, Y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)
        duree = time.time() - t0
        acc_te = float(np.mean(model.predict(X_te, verbose=0).argmax(1) == Y_te.argmax(1)))
        acc_tr = float(np.mean(model.predict(X_tr, verbose=0).argmax(1) == Y_tr.argmax(1)))
        print(f"Keras {nom} : TEST {acc_te:.1%} | TRAIN {acc_tr:.1%} | {duree/60:.1f} min")
        resultats[nom] = (acc_te, acc_tr, duree)

    # 3. Bilan face au C++ (chiffres lus dans le manifeste, mesures en conditions reelles)
    ref = reference_cpp()
    print("\n================= BILAN COMPARATIF =================")
    if ref:
        m = ref.get("metrics", {})
        v = m.get("variants", {})
        print(f"Notre MLP C++ (manifeste v{ref['version']}) :")
        print(f"  TEST bag {m.get('accuracy', 0):.1%} | TRAIN {m.get('accuracy_train', 0):.1%} "
              f"| meilleur variant {v.get('accuracy_best', 0):.1%} "
              f"| moyenne {v.get('accuracy_mean', 0):.1%} +/- {v.get('accuracy_std', 0):.1%}")
    for nom, (te, tr, d) in resultats.items():
        print(f"Keras {nom:<5} : TEST {te:.1%} | TRAIN {tr:.1%} | {d/60:.1f} min")
    print("\nA commenter : ecarts d'accuracy (optimiseur, batch, init) et de temps "
          "(C++ SGD pur batch=1 vs TF batch=32 vectorise).")


if __name__ == "__main__":
    main()
