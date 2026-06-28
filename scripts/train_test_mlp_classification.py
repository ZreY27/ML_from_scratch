import os
import sys
import glob
import matplotlib.pyplot as plt

# Autorise Python à charger les DLLs du compilateur C++ (MSYS2)
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(r"C:\msys64\ucrt64\bin")

# Dossier racine du projet (pour accéder aux datasets)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)  # pour importer model_registry (situé à la racine)

import ML_ESGI
import model_registry as reg

# 1. Paramètres du modèle et de l'entraînement
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
INPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3 # 3072 pixels
LEARNING_RATE = 0.01
DECAY = 0.001 # Réduit progressivement le learning rate
TRAINING_STEPS = 15000 # On peut augmenter les étapes car la fin sera beaucoup plus stable

# 2. Définition des catégories et de leurs labels attendus (One-Hot Encoding)
categories = ["Fighter", "Racing", "Platformer"]
category_labels = {
    "Fighter":    [ 1.0, -1.0, -1.0], # Le neurone 0 doit s'activer
    "Racing":     [-1.0,  1.0, -1.0], # Le neurone 1 doit s'activer
    "Platformer": [-1.0, -1.0,  1.0]  # Le neurone 2 doit s'activer
}

all_paths = []
all_labels_flat = []

# 3. Parcours automatique des dossiers pour lister toutes les images
for cat in categories:
    search_path_jpg = os.path.join(ROOT_DIR, "datasets", cat, "*.jpg")
    search_path_png = os.path.join(ROOT_DIR, "datasets", cat, "*.png")
    imgs = glob.glob(search_path_jpg) + glob.glob(search_path_png)
    
    for img_path in imgs:
        all_paths.append(img_path)
        # On aplatit directement les labels pour le C++
        all_labels_flat.extend(category_labels[cat])

if len(all_paths) == 0:
    print("Aucune image trouvée. Veuillez vérifier les dossiers 'datasets/Fighter', etc.")
else:
    print(f"{len(all_paths)} images trouvées. Préparation de l'entraînement MLP...")
    
    # 4. Création du modèle MLP
    # Entrée: 3072, Couche cachée: 128 (arbitraire, peut être modifié), Sortie: 3
    model = ML_ESGI.MLP([INPUT_SIZE, 128, 3], is_classification=True)
    
    print("Début de l'entraînement...")
    loss_history = model.train_from_images(
        image_paths=all_paths,
        expected_outputs=all_labels_flat,
        target_w=IMAGE_WIDTH,
        target_h=IMAGE_HEIGHT,
        training_steps=TRAINING_STEPS,
        learning_rate=LEARNING_RATE,
        decay=DECAY
    )
    
    # 5. Sauvegarde du modèle entraîné (versionné : poids + manifeste)
    version, manifest_path = reg.save_single(
        model, "mlp_genres", "MLP - Genres", "mlp", categories,
        IMAGE_WIDTH, IMAGE_HEIGHT, models_dir=os.path.join(ROOT_DIR, "models"),
        metrics={"final_loss": loss_history[-1] if loss_history else None},
    )
    print(f"\nEntraînement terminé ! Modèle MLP sauvegardé : version v{version}\n  -> {manifest_path}")

    # 6. Affichage de la courbe d'apprentissage
    # Le SGD est très bruité (une image aléatoire à la fois), on lisse la courbe pour le rapport
    def smooth_curve(points, factor=0.99):
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points

    plt.plot(smooth_curve(loss_history), color='blue', label='MSE Lissé (99%)')
    plt.plot(loss_history, color='lightblue', alpha=0.2, label='MSE Brut')
    plt.title("Courbe d'apprentissage du MLP")
    plt.xlabel("Étapes d'entraînement (SGD Steps)")
    plt.ylabel("Erreur Quadratique Moyenne (Loss)")
    plt.legend()
    plt.show()