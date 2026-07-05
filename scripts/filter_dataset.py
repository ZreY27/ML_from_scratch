"""
filter_dataset.py — Filtrage automatique du dataset d'images (nettoyage pré-entraînement).

Problème : les frames extraites des vidéos de gameplay contiennent du bruit :
écrans de chargement, menus/paramètres, fondus au noir, et quasi-doublons
(deux frames à 5 s d'intervalle pendant une pause).

Approche : des HEURISTIQUES simples et explicables (pas de ML, donc aucun souci
avec la règle "pas d'implémentation externe" du syllabus — c'est de la préparation
de données, et on sait justifier chaque seuil au jury) :

  1. "uniforme"    : contraste très faible (écart-type des gris) -> chargement, fondu noir/blanc.
  2. "aplats"      : très peu de contours (gradients faibles)    -> menus, écrans de pause.
  3. "monochrome"  : image quasi sans couleur ET peu contrastée  -> loading, splash screens.
  4. "doublon"     : quasi-identique à une image déjà gardée (hash perceptuel dHash)
                     -> frames redondantes qui gonflent artificiellement le dataset
                     (et fausseraient le split train/test !).

Usage :
  python scripts/filter_dataset.py                # RAPPORT seulement (ne touche à rien)
  python scripts/filter_dataset.py --appliquer    # déplace les rejets vers datasets_rejetees/

Les rejets ne sont JAMAIS supprimés : ils sont déplacés dans datasets_rejetees/<classe>/,
ce qui permet un contrôle visuel rapide (et de récupérer les faux positifs).
Un rapport CSV (rapport_filtrage.csv) liste chaque image avec ses métriques.

Auteur : Maxime Clément (partie individuelle : dataset).
"""

import os
import csv
import glob
import shutil
import argparse

import numpy as np
from PIL import Image

# --- Seuils (échelle 0-255) : calibrés sur nos screenshots de gameplay. ---
# À ajuster si besoin : lancer d'abord en mode rapport et regarder le CSV.
SEUIL_CONTRASTE  = 18.0   # en dessous : écran uniforme (chargement, fondu)
SEUIL_BORDS      = 2.5    # en dessous : gros aplats (menu, pause)
SEUIL_COULEUR    = 8.0    # en dessous (ET contraste < 35) : quasi monochrome
SEUIL_CONTRASTE2 = 35.0
SEUIL_HAMMING    = 4      # distance dHash max pour considérer deux images identiques
FENETRE_DOUBLONS = 80     # on compare chaque image aux N dernières gardées (frames voisines)

IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png")


def metriques(path):
    """Calcule (contraste, bords, couleur, dhash) sur une miniature 64x64."""
    img = Image.open(path).convert("RGB").resize((64, 64))
    a = np.asarray(img, dtype=np.float32)
    gris = a.mean(axis=2)

    contraste = float(gris.std())

    # Densité de contours : moyenne des gradients absolus horizontaux + verticaux
    bords = float(np.abs(np.diff(gris, axis=0)).mean() + np.abs(np.diff(gris, axis=1)).mean()) / 2.0

    # "Colorfulness" (Hasler & Süsstrunk simplifié) : dispersion des canaux opposés
    rg = a[:, :, 0] - a[:, :, 1]
    yb = 0.5 * (a[:, :, 0] + a[:, :, 1]) - a[:, :, 2]
    couleur = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2))

    # dHash 8x8 : gradient horizontal d'une miniature 9x8 -> 64 bits
    mini = np.asarray(Image.open(path).convert("L").resize((9, 8)), dtype=np.float32)
    bits = (mini[:, 1:] > mini[:, :-1]).flatten()
    dhash = int("".join("1" if b else "0" for b in bits), 2)

    return contraste, bords, couleur, dhash


def raison_rejet(contraste, bords, couleur):
    """Retourne la raison du rejet, ou None si l'image semble être du gameplay."""
    if contraste < SEUIL_CONTRASTE:
        return "uniforme (chargement/fondu)"
    if bords < SEUIL_BORDS:
        return "aplats (menu/pause)"
    if couleur < SEUIL_COULEUR and contraste < SEUIL_CONTRASTE2:
        return "quasi monochrome (loading/splash)"
    return None


def hamming(a, b):
    return bin(a ^ b).count("1")


def filtrer(datasets_dir, rejets_dir, appliquer):
    lignes_csv = []
    total_garde = total_rejet = 0

    classes = [d for d in sorted(os.listdir(datasets_dir))
               if os.path.isdir(os.path.join(datasets_dir, d))]
    if not classes:
        print(f"Aucune classe (sous-dossier) trouvée dans {datasets_dir}")
        return

    for cls in classes:
        dossier = os.path.join(datasets_dir, cls)
        paths = []
        for ext in IMAGE_EXTS:
            paths += glob.glob(os.path.join(dossier, ext))
        # Tri par nom : les frames d'une même vidéo se suivent -> la fenêtre de
        # comparaison des doublons reste petite et le scan reste rapide.
        paths.sort()

        compteurs = {}
        hashes_gardes = []  # fenêtre glissante des dHash des images gardées

        for path in paths:
            try:
                contraste, bords, couleur, dhash = metriques(path)
            except Exception as e:
                raison = f"illisible ({e})"
                contraste = bords = couleur = float("nan")
            else:
                raison = raison_rejet(contraste, bords, couleur)
                # Détection de quasi-doublon UNIQUEMENT si l'image passe les autres filtres
                if raison is None:
                    for h in hashes_gardes[-FENETRE_DOUBLONS:]:
                        if hamming(dhash, h) <= SEUIL_HAMMING:
                            raison = "doublon (dHash)"
                            break
                if raison is None:
                    hashes_gardes.append(dhash)

            lignes_csv.append({
                "classe": cls, "fichier": os.path.basename(path),
                "contraste": f"{contraste:.1f}", "bords": f"{bords:.2f}",
                "couleur": f"{couleur:.1f}",
                "verdict": raison or "garde",
            })

            if raison is None:
                total_garde += 1
                continue

            total_rejet += 1
            compteurs[raison.split(" (")[0]] = compteurs.get(raison.split(" (")[0], 0) + 1
            if appliquer:
                dest = os.path.join(rejets_dir, cls)
                os.makedirs(dest, exist_ok=True)
                shutil.move(path, os.path.join(dest, os.path.basename(path)))

        gardees = len(paths) - sum(compteurs.values())
        detail = ", ".join(f"{k}: {v}" for k, v in sorted(compteurs.items())) or "rien à rejeter"
        print(f"  {cls:<15} {len(paths):>6} images -> {gardees} gardées | {detail}")

    # Rapport CSV complet (pour vérifier/ajuster les seuils)
    rapport = os.path.join(os.path.dirname(datasets_dir) or ".", "rapport_filtrage.csv")
    with open(rapport, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["classe", "fichier", "contraste", "bords", "couleur", "verdict"])
        w.writeheader()
        w.writerows(lignes_csv)

    print(f"\nTotal : {total_garde} gardées, {total_rejet} rejetées.")
    print(f"Rapport détaillé : {rapport}")
    if not appliquer:
        print("\nMode RAPPORT (aucun fichier déplacé). Relancer avec --appliquer pour déplacer")
        print(f"les rejets vers {rejets_dir}\\<classe>\\ (récupérables, rien n'est supprimé).")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Filtre les images non-gameplay du dataset.")
    p.add_argument("--source", default="datasets", help="dossier des classes (défaut : datasets)")
    p.add_argument("--rejets", default="datasets_rejetees", help="dossier de quarantaine")
    p.add_argument("--appliquer", action="store_true",
                   help="déplace réellement les fichiers (sinon : rapport seulement)")
    args = p.parse_args()
    filtrer(args.source, args.rejets, args.appliquer)
