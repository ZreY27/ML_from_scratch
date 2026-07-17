"""
bench_vitesse_keras.py — Vitesse d'entraînement : Keras vs notre C++, A CONDITIONS
STRICTEMENT EGALES.

Notre C++ : SGD batch=1, UN seul thread, calculs en DOUBLE (64 bits).
Keras, par defaut : mini-batch 32, TOUS les coeurs (meme a batch=1 !), float32.
-> Comparer les perfs exige d'enlever ces avantages UN PAR UN. Quatre modes :

  batch32       : Keras tel qu'on l'utilise (batch=32, multi-coeurs, float32)
  batch1        : meme algorithme que nous (batch=1), mais multi-coeurs + float32
  batch1_1cpu   : batch=1 + UN seul thread (reste float32)
  batch1_exact  : batch=1 + UN thread + FLOAT64 -> le clone exact de notre C++

Reference C++ : ~1800 img/s, mesuree par scripts/bench_vitesse_cpp.py MACHINE AU
REPOS (methode de la difference : le cout fixe copie pybind11 + init s'annule).
NB : une premiere reference de 557 img/s etait FAUSSE — elle divisait le temps
total d'une config de l'etude d'hyperparametres (chargement disque + deux
evaluations completes INCLUS) par le nombre de pas. Lecon de benchmark.

TensorFlow fige sa configuration de threads a l'initialisation -> chaque mode
tourne dans un PROCESS SEPARE (le script se relance lui-meme avec le nom du
mode en argument). Donnees synthetiques de meme forme (le debit ne depend pas
du contenu des pixels) ; un epoch de chauffe est exclu du chrono.

COMMANDE (environnement TF, Python 3.12) :
    .venv-tf\\Scripts\\python scripts\\bench_vitesse_keras.py

Auteur : Maxime Clement.
"""

import subprocess
import sys
import time

N, D, H, C = 4800, 3072, 128, 3   # memes dimensions que le vrai MLP (3072 -> 128 -> 3)
CPP_IMGS_PAR_SEC = 1800           # notre C++ : bench_vitesse_cpp.py, machine au repos (cf. docstring)

MODES = {
    #  nom          (batch, un_seul_thread, float64, epochs chronometres)
    "batch32":      (32, False, False, 3),
    "batch1":       (1,  False, False, 1),
    "batch1_1cpu":  (1,  True,  False, 1),
    "batch1_exact": (1,  True,  True,  1),
}


def bench_un_mode(nom):
    """Execute UN mode (dans le process courant) et imprime 'RESULT <nom> <img/s>'."""
    batch, un_thread, en_double, epochs = MODES[nom]

    import tensorflow as tf
    if un_thread:  # a poser AVANT toute operation TF
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    if en_double:
        tf.keras.backend.set_floatx("float64")

    import numpy as np
    rng = np.random.default_rng(42)
    dtype = np.float64 if en_double else np.float32
    X = rng.random((N, D), dtype=np.float64).astype(dtype)
    Y = np.full((N, C), -1.0, dtype=dtype)
    Y[np.arange(N), rng.integers(0, C, size=N)] = 1.0

    tf.keras.utils.set_random_seed(42)
    modele = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(D,)),
        tf.keras.layers.Dense(H, activation="tanh"),
        tf.keras.layers.Dense(C, activation="tanh"),
    ])
    modele.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss="mse")

    modele.fit(X, Y, epochs=1, batch_size=batch, verbose=0)  # chauffe (tracage/compilation)
    t0 = time.perf_counter()
    modele.fit(X, Y, epochs=epochs, batch_size=batch, verbose=0)
    duree = time.perf_counter() - t0
    print(f"RESULT {nom} {N * epochs / duree:.0f}", flush=True)


def main():
    print(f"Banc de vitesse — MLP {D}->{H}->{C}, {N} exemples synthetiques")
    print("(chaque mode dans un process separe : TF fige ses threads au demarrage)\n")

    resultats = {}
    for nom in MODES:
        sortie = subprocess.run([sys.executable, __file__, nom],
                                capture_output=True, text=True)
        for ligne in sortie.stdout.splitlines():
            if ligne.startswith("RESULT "):
                resultats[nom] = float(ligne.split()[2])
        if nom not in resultats:
            print(f"ECHEC du mode {nom} :\n{sortie.stderr[-800:]}")
            return

    libelles = {
        "batch32":      "Keras batch=32, tous coeurs, float32 (mode normal)",
        "batch1":       "Keras batch=1,  tous coeurs, float32",
        "batch1_1cpu":  "Keras batch=1,  UN coeur,    float32",
        "batch1_exact": "Keras batch=1,  UN coeur,    FLOAT64  <- clone de notre C++",
    }
    for nom, debit in resultats.items():
        print(f"{libelles[nom]:<58} : {debit:7.0f} img/s")
    print(f"{'Notre C++   batch=1,  UN coeur,    double (reference)':<58} : {CPP_IMGS_PAR_SEC:7.0f} img/s")

    exact = resultats["batch1_exact"]
    print("\n=== LECTURE (decomposition de l'ecart, avantage par avantage) ===")
    print(f"Mini-batch 32          : x{resultats['batch32'] / resultats['batch1']:.1f}")
    print(f"Multi-coeurs (batch=1) : x{resultats['batch1'] / resultats['batch1_1cpu']:.1f}")
    print(f"float32 vs float64     : x{resultats['batch1_1cpu'] / exact:.1f}")
    print(f"A CONDITIONS IDENTIQUES (batch=1, 1 coeur, 64 bits) :")
    print(f"  Keras {exact:.0f} img/s vs notre C++ {CPP_IMGS_PAR_SEC} img/s "
          f"-> ratio C++/Keras = x{CPP_IMGS_PAR_SEC / exact:.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        bench_un_mode(sys.argv[1])
    else:
        main()
