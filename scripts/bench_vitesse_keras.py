"""
bench_vitesse_keras.py — Vitesse d'entraînement A ARMES EGALES : Keras vs notre C++.

LA QUESTION : dire « Keras est 15x plus rapide » compare notre SGD batch=1 a un
batch=32 vectorise — deux algorithmes differents. Pour comparer les PERFS, il
faut le meme travail des deux cotes. Ce banc mesure donc le DEBIT Keras (images
traitees / seconde, forward + backward + mise a jour) :
  - a batch = 1  (strictement le meme algorithme que notre C++) ;
  - a batch = 32 (le mode "normal" de Keras, pour situer le facteur batch).

Reference C++ (mesuree sur CETTE machine, results/etude_hyperparametres.csv,
config MLP 128, mono-thread, batch=1) : 240 000 pas en ~431 s -> ~557 images/s.

Le debit ne depend pas du CONTENU des images -> donnees synthetiques de meme
forme (4800 x 3072, cibles one-hot +/-1), aucune lecture disque, mesure pure.
Un epoch de chauffe est exclu du chrono (compilation/tracage TensorFlow).

COMMANDE (environnement TF, Python 3.12) :
    .venv-tf\\Scripts\\python scripts\\bench_vitesse_keras.py

Auteur : Maxime Clement.
"""

import time

import numpy as np

N, D, H, C = 4800, 3072, 128, 3   # memes dimensions que le vrai MLP (3072 -> 128 -> 3)
CPP_IMGS_PAR_SEC = 557            # notre C++ : 240 000 pas / 431 s (etude_hyperparametres.csv)

import tensorflow as tf


def bench(batch_size, epochs_chronometres):
    """Debit Keras (images/s) pour un batch donne, hors epoch de chauffe."""
    tf.keras.utils.set_random_seed(42)
    modele = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(D,)),
        tf.keras.layers.Dense(H, activation="tanh"),
        tf.keras.layers.Dense(C, activation="tanh"),
    ])
    modele.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss="mse")

    modele.fit(X, Y, epochs=1, batch_size=batch_size, verbose=0)  # chauffe (non chronometree)
    t0 = time.perf_counter()
    modele.fit(X, Y, epochs=epochs_chronometres, batch_size=batch_size, verbose=0)
    duree = time.perf_counter() - t0
    return N * epochs_chronometres / duree


rng = np.random.default_rng(42)
X = rng.random((N, D), dtype=np.float32)
Y = np.full((N, C), -1.0, dtype=np.float32)
Y[np.arange(N), rng.integers(0, C, size=N)] = 1.0

print(f"Banc de vitesse — MLP {D}->{H}->{C}, {N} exemples synthetiques\n")
debit_b1 = bench(batch_size=1, epochs_chronometres=1)
print(f"Keras batch=1   : {debit_b1:7.0f} images/s   (MEME algorithme que notre C++)")
debit_b32 = bench(batch_size=32, epochs_chronometres=3)
print(f"Keras batch=32  : {debit_b32:7.0f} images/s   (mode normal de Keras, vectorise)")
print(f"Notre C++ b=1   : {CPP_IMGS_PAR_SEC:7.0f} images/s   (mesure etude_hyperparametres.csv, 1 coeur)")

print("\n=== LECTURE ===")
print(f"A armes egales (batch=1)     : C++ / Keras = x{CPP_IMGS_PAR_SEC / debit_b1:.1f}")
print(f"Effet du batch chez Keras    : x{debit_b32 / debit_b1:.1f} entre batch=1 et batch=32")
print(f"Keras batch=32 vs C++ batch=1: x{debit_b32 / CPP_IMGS_PAR_SEC:.1f}")
print("\nConclusion attendue : le facteur de vitesse vient du BATCH (vectorisation),")
print("pas du langage — a batch=1, l'overhead par pas de TensorFlow le penalise.")
