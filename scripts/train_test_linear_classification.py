import os
import sys

# Autorise Python à charger les DLLs du compilateur C++ (MSYS2)
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(r"C:\msys64\ucrt64\bin")

# Dossier racine du projet (pour accéder aux datasets)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)  # pour importer model_registry (situé à la racine)

import ML_ESGI
import model_registry as reg
import matplotlib.pyplot as plt
import glob

# 1. Paramètres de l'image (doivent correspondre aux entrées du modèle)
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
INPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3
LEARNING_RATE = 0.01
EPOCHS = 500

# 2. Définition des 3 dossiers
categories = ["Fighter", "Racing", "Platformer"]
images_dict = {}
all_paths = []

# Parcours automatique des dossiers
for cat in categories:
    # Utilisation du chemin absolu pour trouver les images indépendamment du répertoire d'exécution du script
    search_path_jpg = os.path.join(ROOT_DIR, "datasets", cat, "*.jpg")
    search_path_png = os.path.join(ROOT_DIR, "datasets", cat, "*.png")
    imgs = glob.glob(search_path_jpg) + glob.glob(search_path_png)
    images_dict[cat] = imgs
    all_paths.extend(imgs)

if len(all_paths) == 0:
    print("Veuillez placer les images dans les dossiers datasets/Fighter, datasets/Racing, etc.")
else:
    print(f"{len(all_paths)} images trouvées. Début du Multi-classes (One-vs-Rest)...")
    
    # Dictionnaire pour stocker les 3 modèles entraînés
    trained_models = {}
    final_errors = {}

    # 3. Boucle principale : Un modèle par catégorie
    for target_cat in categories:
        print(f"\n--- Modèle : {target_cat.upper()} vs RESTE ---")
        
        # Création des labels dynamiquement pour ce modèle spécifique
        labels = []
        for cat in categories:
            if cat == target_cat:
                labels.extend([1.0] * len(images_dict[cat])) # Cible = 1.0
            else:
                labels.extend([-1.0] * len(images_dict[cat])) # Reste = -1.0
                
        # Création et Entraînement
        model = ML_ESGI.LinearModel(INPUT_SIZE, is_classification=True)
        loss_history = model.train_from_images(all_paths, labels, IMAGE_WIDTH, IMAGE_HEIGHT, LEARNING_RATE, EPOCHS)
        
        # Conservation du modèle (sauvegarde groupée en One-vs-Rest après la boucle)
        trained_models[target_cat] = model
        final_errors[target_cat] = loss_history[-1] if loss_history else None

        # Ajout de la courbe au graphique global
        plt.plot(loss_history, label=f"{target_cat} vs Rest")
        print(f"Entraînement de {target_cat} terminé (erreur finale : {final_errors[target_cat]:.3f}).")

    # Sauvegarde groupée : un classifieur One-vs-Rest versionné (3 binaires + manifeste)
    version, manifest_path = reg.save_onevsrest(
        trained_models, "linear_genres", "Perceptron - Genres (One-vs-Rest)",
        base_type="linear", width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
        models_dir=os.path.join(ROOT_DIR, "models"),
        metrics={"train_error_final": final_errors},
    )
    print(f"\nClassifieur One-vs-Rest sauvegardé : version v{version}\n  -> {manifest_path}")

    # 4. Affichage du graphique final avec les 3 courbes
    plt.title("Évolution des erreurs (Stratégie One-vs-Rest)")
    plt.xlabel("Epochs")
    plt.ylabel("Ratio d'erreurs")
    plt.legend()
    plt.show()
