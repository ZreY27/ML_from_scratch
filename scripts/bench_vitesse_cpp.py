"""
bench_vitesse_cpp.py — Debit d'entrainement de NOTRE MLP C++ (batch=1, 1 coeur, double).

Pendant du banc Keras (bench_vitesse_keras.py) : les deux se lancent MACHINE AU
REPOS pour etre comparables — une reference mesuree pendant qu'un autre calcul
tourne est faussee (on l'a constate : ±10-35 % selon la charge).

MESURE PROPRE : on chronometre deux appels train() complets (S1 puis S2 pas)
et on prend la DIFFERENCE (S2-S1)/(t2-t1) -> le cout fixe (copie pybind11 des
donnees, initialisation) s'annule, il reste le debit pur forward+backward+
mise a jour par exemple. Donnees synthetiques de meme forme que le vrai MLP
(3072 -> 128 -> 3) : le debit ne depend pas du contenu des pixels.

COMMANDE (Python 3.14, a la racine, build Release requis) :
    python scripts\\bench_vitesse_cpp.py

Auteur : Maxime Clement.
"""

import random
import time

import training_utils as tu
tu.enable_cpp_dlls()  # AVANT d'importer ML_ESGI

import ML_ESGI

N, D, H, C = 4800, 3072, 128, 3   # memes dimensions ET meme taille de dataset que le banc Keras
S1, S2 = 10000, 30000             # deux durees d'entrainement -> la difference annule le cout fixe

random.seed(42)
inputs_flat = [random.random() for _ in range(N * D)]
labels_flat = []
for _ in range(N):
    onehot = [-1.0] * C
    onehot[random.randrange(C)] = 1.0
    labels_flat.extend(onehot)


def duree_train(steps):
    """Duree totale d'un train() complet de `steps` pas (copie des donnees incluse)."""
    modele = ML_ESGI.MLP([D, H, C], is_classification=True)
    t0 = time.perf_counter()
    modele.train(inputs_flat, labels_flat, steps, 0.01, 0.0)
    return time.perf_counter() - t0


print(f"Banc C++ — MLP {D}->{H}->{C}, {N} exemples synthetiques, batch=1, 1 coeur, double")
t1 = duree_train(S1)
t2 = duree_train(S2)
debit = (S2 - S1) / (t2 - t1)
print(f"\n{S1} pas : {t1:.1f} s | {S2} pas : {t2:.1f} s")
print(f"Debit pur (difference, cout fixe annule) : {debit:.0f} images/s")
print("\nA comparer au mode 'batch1_exact' de bench_vitesse_keras.py")
print("(Keras batch=1, 1 coeur, float64), mesure lui aussi machine au repos.")
