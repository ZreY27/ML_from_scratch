"""
training_utils.py — Helpers partagés par les scripts d'entraînement.

- Détection automatique des classes (sous-dossiers de datasets/).
- Chargement équilibré (plafond par classe) pour éviter qu'une classe majoritaire écrase tout.
- Split train/test stratifié (par classe).
- Évaluation (accuracy globale + par classe) réutilisant la logique d'inférence de model_registry.Predictor.
"""

import os
import glob
import random

# Chemin des DLL du runtime gcc (MSYS2) — requis sous Windows avant d'importer ML_ESGI.
MSYS2_BIN = r"C:\msys64\ucrt64\bin"
IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png")


def enable_cpp_dlls():
    """À appeler AVANT `import ML_ESGI` (charge les DLL du compilateur C++ sous Windows)."""
    if hasattr(os, "add_dll_directory") and os.path.isdir(MSYS2_BIN):
        os.add_dll_directory(MSYS2_BIN)


def _list_images(folder):
    paths = []
    for ext in IMAGE_EXTS:
        paths += glob.glob(os.path.join(folder, ext))
    return paths


def discover_classes(datasets_dir):
    """Classes = sous-dossiers de datasets_dir contenant au moins une image (triés)."""
    classes = []
    if not os.path.isdir(datasets_dir):
        return classes
    for name in sorted(os.listdir(datasets_dir)):
        folder = os.path.join(datasets_dir, name)
        if os.path.isdir(folder) and _list_images(folder):
            classes.append(name)
    return classes


def load_dataset(datasets_dir, classes, max_per_class=None, seed=42):
    """Retourne {classe: [chemins]}, mélangé, avec plafond optionnel par classe (équilibrage)."""
    rng = random.Random(seed)
    data = {}
    for cls in classes:
        paths = _list_images(os.path.join(datasets_dir, cls))
        rng.shuffle(paths)
        if max_per_class:
            paths = paths[:max_per_class]
        data[cls] = paths
    return data


def train_test_split(images_by_class, test_ratio=0.2, seed=42):
    """Split stratifié (par classe). Retourne (train, test), deux dicts {classe: [chemins]}."""
    rng = random.Random(seed)
    train, test = {}, {}
    for cls, paths in images_by_class.items():
        shuffled = paths[:]
        rng.shuffle(shuffled)
        # au moins 1 image en test si la classe en a plusieurs
        n_test = max(1, int(len(shuffled) * test_ratio)) if len(shuffled) > 1 else 0
        test[cls] = shuffled[:n_test]
        train[cls] = shuffled[n_test:]
    return train, test


def counts(images_by_class):
    """{classe: nombre d'images}."""
    return {cls: len(paths) for cls, paths in images_by_class.items()}


def evaluate(predictor, test_by_class, width, height):
    """Accuracy globale + par classe sur le set de test.

    predictor : objet avec .predict(vecteur_image) -> nom_de_classe (ex: model_registry.Predictor).
    Retourne (accuracy_globale, {classe: accuracy}).
    """
    import ML_ESGI  # import tardif (suppose enable_cpp_dlls() déjà fait)

    per_class = {}
    correct = total = 0
    for cls, paths in test_by_class.items():
        c = t = 0
        for path in paths:
            try:
                img = ML_ESGI.load_and_resize_image(path, width, height)
            except Exception:
                continue
            t += 1
            if predictor.predict(img) == cls:
                c += 1
        per_class[cls] = (c / t) if t else None
        correct += c
        total += t
    return (correct / total if total else 0.0), per_class
